/* Kokkos port of test_sync: build linear test functions on the local
 * grid, corrupt the ghost zones, run sync_vars, verify ghosts are
 * restored. Drives both EVOLVED and AUX paths. */
#include "comm.hpp"
#include "gf.hpp"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "domain.h"
}

#define TWO_PI 6.283185307179586476925

KOKKOS_INLINE_FUNCTION
double test_func(int v, double x, double y, double z)
{
    const int kx = v + 1, ky = v + 2, kz = v + 3;
    return Kokkos::sin(TWO_PI * kx * x)
         + Kokkos::cos(TWO_PI * ky * y)
         + Kokkos::sin(TWO_PI * kz * z);
}

/* Kokkos kernel: write test_func into a Field3D over the full local box
 * (ghost zones included). For EVOLVED writes go to state; for AUX to
 * state as well (the verify path will see both). */
static void fill_view(Field3D V, int v_global,
                      double x0, double y0, double z0,
                      double dx, double dy, double dz)
{
    const int64_t nx = V.extent(0), ny = V.extent(1), nz = V.extent(2);
    using Range3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
    Kokkos::parallel_for("fill_view",
        Range3D({0, 0, 0}, {nx, ny, nz}),
        KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
            const double x = x0 + i * dx;
            const double y = y0 + j * dy;
            const double z = z0 + k * dz;
            V(i, j, k) = test_func(v_global, x, y, z);
        });
}

/* Stamp sentinel values into ghost zones whose neighbour exists and is
 * a different rank. (Self-periodic axes share their interior with
 * their own ghost, so corrupting them would also corrupt the source of
 * the sync data.) */
static void corrupt_ghost(Field3D V, int gs, const NGFS *gfs)
{
    const int64_t nx = V.extent(0), ny = V.extent(1), nz = V.extent(2);
    const int self = gfs->domain.rank;
    const int lx = gfs->domain.lower_x_rank;
    const int ux = gfs->domain.upper_x_rank;
    const int ly = gfs->domain.lower_y_rank;
    const int uy = gfs->domain.upper_y_rank;
    const int lz = gfs->domain.lower_z_rank;
    const int uz = gfs->domain.upper_z_rank;
    using Range3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    if (lx != INVALID_RANK && lx != self)
        Kokkos::parallel_for("corrupt_lx",
            Range3D({0, 0, 0}, {gs, ny, nz}),
            KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                V(i, j, k) = 42.0;
            });
    if (ux != INVALID_RANK && ux != self)
        Kokkos::parallel_for("corrupt_ux",
            Range3D({nx - gs, 0, 0}, {nx, ny, nz}),
            KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                V(i, j, k) = 45.0;
            });
    if (ly != INVALID_RANK && ly != self)
        Kokkos::parallel_for("corrupt_ly",
            Range3D({0, 0, 0}, {nx, gs, nz}),
            KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                V(i, j, k) = 43.0;
            });
    if (uy != INVALID_RANK && uy != self)
        Kokkos::parallel_for("corrupt_uy",
            Range3D({0, ny - gs, 0}, {nx, ny, nz}),
            KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                V(i, j, k) = 46.0;
            });
    if (lz != INVALID_RANK && lz != self)
        Kokkos::parallel_for("corrupt_lz",
            Range3D({0, 0, 0}, {nx, ny, gs}),
            KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                V(i, j, k) = 44.0;
            });
    if (uz != INVALID_RANK && uz != self)
        Kokkos::parallel_for("corrupt_uz",
            Range3D({0, 0, nz - gs}, {nx, ny, nz}),
            KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                V(i, j, k) = 47.0;
            });
}

static int verify_view(Field3D V, int v_global,
                       double x0, double y0, double z0,
                       double dx, double dy, double dz, int rank)
{
    auto h = Kokkos::create_mirror_view(V);
    Kokkos::deep_copy(h, V);
    const int64_t nx = V.extent(0), ny = V.extent(1), nz = V.extent(2);
    const double tol = 1.0e-12;
    double max_err = 0.0;
    for (int64_t k = 0; k < nz; k++)
    {
        const double z = z0 + k * dz;
        for (int64_t j = 0; j < ny; j++)
        {
            const double y = y0 + j * dy;
            for (int64_t i = 0; i < nx; i++)
            {
                const double x = x0 + i * dx;
                const int kx = v_global + 1, ky = v_global + 2, kz = v_global + 3;
                const double expected = std::sin(TWO_PI * kx * x)
                                      + std::cos(TWO_PI * ky * y)
                                      + std::sin(TWO_PI * kz * z);
                const double err = std::fabs(h(i, j, k) - expected);
                if (err > max_err) max_err = err;
                if (err > tol)
                {
                    std::fprintf(stderr,
                        "Rank %d: var %d mismatch at (%lld,%lld,%lld): "
                        "got %.16e expected %.16e (err %.3e)\n",
                        rank, v_global,
                        (long long)i, (long long)j, (long long)k,
                        h(i, j, k), expected, err);
                    return -1;
                }
            }
        }
    }
    if (rank == 0)
        std::fprintf(stderr, "  var %d max error: %.3e\n", v_global, max_err);
    return 0;
}

static void output_json_var0(const NGFS *gfs)
{
    auto h = Kokkos::create_mirror_view(gfs->evol[0].state);
    Kokkos::deep_copy(h, gfs->evol[0].state);
    char fname[128];
    std::snprintf(fname, sizeof fname, "Var0_rank_%d.json", gfs->domain.rank);
    FILE *f = std::fopen(fname, "w");
    std::fprintf(f, "{\n");
    std::fprintf(f, "    \"nx\": %ld,\n", (long)gfs->nx);
    std::fprintf(f, "    \"ny\": %ld,\n", (long)gfs->ny);
    std::fprintf(f, "    \"nz\": %ld,\n", (long)gfs->nz);
    std::fprintf(f, "    \"dx\": %20.16e,\n", gfs->dx);
    std::fprintf(f, "    \"dy\": %20.16e,\n", gfs->dy);
    std::fprintf(f, "    \"dz\": %20.16e,\n", gfs->dz);
    std::fprintf(f, "    \"x0\": %20.16e,\n", gfs->x0);
    std::fprintf(f, "    \"y0\": %20.16e,\n", gfs->y0);
    std::fprintf(f, "    \"z0\": %20.16e,\n", gfs->z0);
    std::fprintf(f, "    \"local_i0\": %ld,\n", (long)gfs->domain.local_i0);
    std::fprintf(f, "    \"local_j0\": %ld,\n", (long)gfs->domain.local_j0);
    std::fprintf(f, "    \"local_k0\": %ld,\n", (long)gfs->domain.local_k0);
    std::fprintf(f, "    \"global_ni\": %ld,\n", (long)gfs->domain.global_ni);
    std::fprintf(f, "    \"global_nj\": %ld,\n", (long)gfs->domain.global_nj);
    std::fprintf(f, "    \"global_nk\": %ld,\n", (long)gfs->domain.global_nk);
    std::fprintf(f, "    \"global_x0\": %20.16e,\n", gfs->domain.global_x0);
    std::fprintf(f, "    \"global_y0\": %20.16e,\n", gfs->domain.global_y0);
    std::fprintf(f, "    \"global_z0\": %20.16e,\n", gfs->domain.global_z0);
    std::fprintf(f, "    \"rank\": %d,\n", gfs->domain.rank);
    std::fprintf(f, "    \"mpi_size\": %d,\n", gfs->domain.mpi_size);
    std::fprintf(f, "    \"gs\": %d,\n", gfs->gs);
    std::fprintf(f, "    \"lower_x_ghost\": %s,\n",
                 (gfs->domain.lower_x_rank != INVALID_RANK) ? "true" : "false");
    std::fprintf(f, "    \"upper_x_ghost\": %s,\n",
                 (gfs->domain.upper_x_rank != INVALID_RANK) ? "true" : "false");
    std::fprintf(f, "    \"lower_y_ghost\": %s,\n",
                 (gfs->domain.lower_y_rank != INVALID_RANK) ? "true" : "false");
    std::fprintf(f, "    \"upper_y_ghost\": %s,\n",
                 (gfs->domain.upper_y_rank != INVALID_RANK) ? "true" : "false");
    std::fprintf(f, "    \"lower_z_ghost\": %s,\n",
                 (gfs->domain.lower_z_rank != INVALID_RANK) ? "true" : "false");
    std::fprintf(f, "    \"upper_z_ghost\": %s,\n",
                 (gfs->domain.upper_z_rank != INVALID_RANK) ? "true" : "false");
    std::fprintf(f, "    \"data\": [ ");
    for (int64_t k = 0; k < gfs->nz; k++)
    {
        std::fprintf(f, "%s[\n", k ? "," : "");
        for (int64_t j = 0; j < gfs->ny; j++)
        {
            std::fprintf(f, "%s[\n", j ? "," : "");
            for (int64_t i = 0; i < gfs->nx; i++)
            {
                std::fprintf(f, "%s%20.16e", i ? "," : "", h(i, j, k));
            }
            std::fprintf(f, "]");
        }
        std::fprintf(f, "]");
    }
    std::fprintf(f, "]\n}\n");
    std::fclose(f);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    int rc = EXIT_SUCCESS;
    {
        int mpi_size, mpi_rank;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

        if (argc < 4 || argc > 7)
        {
            if (mpi_rank == 0)
                std::fprintf(stderr,
                             "Usage: %s NX NY NZ "
                             "[periodic_x periodic_y periodic_z]\n",
                             argv[0]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        const int64_t global_nx = std::atoi(argv[1]);
        const int64_t global_ny = std::atoi(argv[2]);
        const int64_t global_nz = std::atoi(argv[3]);
        int per_x = 1, per_y = 1, per_z = 0;
        if (argc >= 7)
        {
            per_x = std::atoi(argv[4]);
            per_y = std::atoi(argv[5]);
            per_z = std::atoi(argv[6]);
        }

        size_t dims[3] = { (size_t)global_nx, (size_t)global_ny,
                           (size_t)global_nz };
        size_t topology[3];
        automatic_topology(3, dims, mpi_size, topology);

        const int gs = 2;
        const int n_evol = 9;
        const int n_aux  = 5;

        NGFS gfs{};
        setup_3d_domain((int)topology[0], (int)topology[1], (int)topology[2],
                        mpi_rank, global_nx, global_ny, global_nz, gs,
                        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, &gfs.domain,
                        per_x ? PERIODIC : NON_PERIODIC,
                        per_y ? PERIODIC : NON_PERIODIC,
                        per_z ? PERIODIC : NON_PERIODIC);
        ngfs_allocate(n_evol, n_aux, &gfs);

        /* --- EVOLVED sync (state buffer, kidx = -1) --- */
        for (int v = 0; v < n_evol; v++)
            fill_view(gfs.evol[v].state, /*v_global=*/v,
                      gfs.x0, gfs.y0, gfs.z0, gfs.dx, gfs.dy, gfs.dz);
        for (int v = 0; v < n_evol; v++)
            corrupt_ghost(gfs.evol[v].state, gs, &gfs);
        sync_vars(&gfs, EVOLVED, /*kidx=*/-1);

        if (mpi_rank == 0)
            std::fprintf(stderr, "Verifying EVOLVED sync (%d vars)...\n",
                         n_evol);
        for (int v = 0; v < n_evol; v++)
        {
            if (verify_view(gfs.evol[v].state, v,
                            gfs.x0, gfs.y0, gfs.z0,
                            gfs.dx, gfs.dy, gfs.dz, mpi_rank) != 0)
            {
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        output_json_var0(&gfs);

        /* --- AUX sync (state buffer) --- */
        for (int v = 0; v < n_aux; v++)
            fill_view(gfs.aux[v].state, /*v_global=*/n_evol + v,
                      gfs.x0, gfs.y0, gfs.z0, gfs.dx, gfs.dy, gfs.dz);
        for (int v = 0; v < n_aux; v++)
            corrupt_ghost(gfs.aux[v].state, gs, &gfs);
        sync_vars(&gfs, AUX);

        if (mpi_rank == 0)
            std::fprintf(stderr, "Verifying AUX sync (%d vars)...\n", n_aux);
        for (int v = 0; v < n_aux; v++)
        {
            if (verify_view(gfs.aux[v].state, n_evol + v,
                            gfs.x0, gfs.y0, gfs.z0,
                            gfs.dx, gfs.dy, gfs.dz, mpi_rank) != 0)
            {
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }

        if (mpi_rank == 0) std::fprintf(stderr, "PASSED\n");

        ngfs_deallocate(&gfs);
        cleanup_3d_domain(&gfs.domain);
    }
    Kokkos::finalize();
    MPI_Finalize();
    return rc;
}
