/* Kokkos port of test_rk4: dy/dt = -y on a uniform grid; checks the
 * 4th-order RK4 convergence rate. */
#include "comm.hpp"
#include "gf.hpp"
#include "rk4.hpp"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" {
#include "domain.h"
}

static void rhs_exp_decay(NGFS *gfs, double /*t*/, int kidx)
{
    const int64_t nx = gfs->nx, ny = gfs->ny, nz = gfs->nz;
    using Range3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
    for (int v = 0; v < gfs->n_evol_vars; v++)
    {
        auto S = gfs->evol[v].state;
        auto K = gfs->evol[v].K[kidx];
        Kokkos::parallel_for("rhs_exp_decay",
            Range3D({0, 0, 0}, {nx, ny, nz}),
            KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                K(i, j, k) = -S(i, j, k);
            });
    }
    sync_vars(gfs, EVOLVED, kidx);
}

static double run_test(double dt, int nsteps)
{
    const int gs = 2;
    const int64_t nx = 8, ny = 8, nz = 8;
    const int n_evol = 1, n_aux = 0;

    NGFS gfs{};
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    setup_3d_domain(1, 1, 1, rank, nx, ny, nz, gs,
                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, &gfs.domain,
                    NON_PERIODIC, NON_PERIODIC, NON_PERIODIC);
    ngfs_allocate(n_evol, n_aux, &gfs);

    Kokkos::deep_copy(gfs.evol[0].state, 1.0);
    Kokkos::deep_copy(gfs.evol[0].old_,  1.0);

    double t = 0.0;
    for (int step = 0; step < nsteps; step++)
    {
        RK4_Step(&gfs, t, dt, rhs_exp_decay);
        t += dt;
    }

    const double exact = std::exp(-t);
    auto h = Kokkos::create_mirror_view(gfs.evol[0].state);
    Kokkos::deep_copy(h, gfs.evol[0].state);
    double max_err = 0.0;
    for (int64_t k = 0; k < gfs.nz; k++)
        for (int64_t j = 0; j < gfs.ny; j++)
            for (int64_t i = 0; i < gfs.nx; i++)
            {
                const double err = std::fabs(h(i, j, k) - exact);
                if (err > max_err) max_err = err;
            }

    ngfs_deallocate(&gfs);
    cleanup_3d_domain(&gfs.domain);
    return max_err;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    int rc = EXIT_SUCCESS;
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        const double dt1 = 0.01, dt2 = dt1 / 2.0;
        const int    n1 = 100,    n2 = 200;
        const double err1 = run_test(dt1, n1);
        const double err2 = run_test(dt2, n2);

        if (rank == 0)
        {
            const double ratio = err1 / err2;
            std::fprintf(stderr, "RK4 test: dy/dt = -y, t_final = 1.0\n");
            std::fprintf(stderr, "  dt = %.4f : error = %.3e\n", dt1, err1);
            std::fprintf(stderr, "  dt = %.4f : error = %.3e\n", dt2, err2);
            std::fprintf(stderr, "  ratio = %.2f (expected ~16 for 4th order)\n",
                         ratio);
            if (err1 > 1.0e-8)
            {
                std::fprintf(stderr, "FAILED: error too large at dt=%.4f\n", dt1);
                rc = EXIT_FAILURE;
            }
            else if (ratio < 12.0 || ratio > 20.0)
            {
                std::fprintf(stderr,
                             "FAILED: convergence ratio %.2f not near 16\n",
                             ratio);
                rc = EXIT_FAILURE;
            }
            else
            {
                std::fprintf(stderr, "PASSED\n");
            }
        }
        MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    Kokkos::finalize();
    if (rc != EXIT_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);
    MPI_Finalize();
    return rc;
}
