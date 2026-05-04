/* test_sync.c — Test domain decomposition and ghost-zone synchronisation
 * for the Maxwell grid infrastructure.
 *
 * Usage: mpirun -np N ./test_sync NX NY NZ [periodic_x periodic_y periodic_z]
 *   periodic flags: 1 = periodic, 0 = non-periodic (default: 1 1 0)
 *
 * The test:
 *   1. Sets up a 3D domain with automatic topology
 *   2. Allocates n_evol evolved variables and n_aux auxiliary variables
 *   3. Fills all variables with known linear test functions via ->dot
 *   4. Deliberately corrupts ghost zones with sentinel values
 *   5. Calls sync_vars() which must restore the correct values
 *   6. Verifies all data (including ghost zones) against expected values
 *   7. Outputs vars[0]->dot as JSON for Python verification
 */
#include "comm.h"
#include "domain.h"
#include "gf.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* Periodic test function with integer frequencies (period = 1 in all
 * directions).  Integer multiples of 2π ensure f(x) = f(x+1). */
#define TWO_PI 6.283185307179586476925
static double test_func(int v, double x, double y, double z)
{
    const int kx = v + 1;
    const int ky = v + 2;
    const int kz = v + 3;
    return sin(TWO_PI * kx * x)
         + cos(TWO_PI * ky * y)
         + sin(TWO_PI * kz * z);
}

static void fill_var_dot(struct ngfs *gfs, int vstart, int nvars)
{
    for (int v = 0; v < nvars; v++)
    {
        double *dot = gfs->vars[v + vstart]->dot;
        for (int64_t k = 0; k < gfs->nz; k++)
        {
            const double z = gfs->z0 + k * gfs->dz;
            for (int64_t j = 0; j < gfs->ny; j++)
            {
                const double y = gfs->y0 + j * gfs->dy;
                for (int64_t i = 0; i < gfs->nx; i++)
                {
                    const double x = gfs->x0 + i * gfs->dx;
                    dot[ijk_indx(i, j, k, gfs)] = test_func(v + vstart, x, y, z);
                }
            }
        }
    }
}

/* Corrupt ghost zones with sentinel values.
 * Only corrupt faces where the neighbor is a DIFFERENT rank.  When a rank
 * is its own periodic neighbor (1 proc per periodic axis), corrupting the
 * ghost zone would also corrupt the interior that the sync reads from,
 * since the axis-by-axis sync reads cross-axis ghost zones as part of the
 * interior face data. */
static void corrupt_ghost_dot(struct ngfs *gfs, int vstart, int nvars)
{
    const int64_t nx = gfs->nx, ny = gfs->ny, nz = gfs->nz;
    const int gs = gfs->gs;
    const int self = gfs->domain.rank;

    for (int v = 0; v < nvars; v++)
    {
        double *dot = gfs->vars[v + vstart]->dot;
        const int lx = gfs->domain.lower_x_rank;
        const int ux = gfs->domain.upper_x_rank;
        const int ly = gfs->domain.lower_y_rank;
        const int uy = gfs->domain.upper_y_rank;
        const int lz = gfs->domain.lower_z_rank;
        const int uz = gfs->domain.upper_z_rank;

        if (lx != INVALID_RANK && lx != self)
            for (int64_t k = 0; k < nz; k++)
                for (int64_t j = 0; j < ny; j++)
                    for (int i = 0; i < gs; i++)
                        dot[ijk_indx(i, j, k, gfs)] = 42;

        if (ux != INVALID_RANK && ux != self)
            for (int64_t k = 0; k < nz; k++)
                for (int64_t j = 0; j < ny; j++)
                    for (int64_t i = nx - gs; i < nx; i++)
                        dot[ijk_indx(i, j, k, gfs)] = 45;

        if (ly != INVALID_RANK && ly != self)
            for (int64_t k = 0; k < nz; k++)
                for (int j = 0; j < gs; j++)
                    for (int64_t i = 0; i < nx; i++)
                        dot[ijk_indx(i, j, k, gfs)] = 43;

        if (uy != INVALID_RANK && uy != self)
            for (int64_t k = 0; k < nz; k++)
                for (int64_t j = ny - gs; j < ny; j++)
                    for (int64_t i = 0; i < nx; i++)
                        dot[ijk_indx(i, j, k, gfs)] = 46;

        if (lz != INVALID_RANK && lz != self)
            for (int k = 0; k < gs; k++)
                for (int64_t j = 0; j < ny; j++)
                    for (int64_t i = 0; i < nx; i++)
                        dot[ijk_indx(i, j, k, gfs)] = 44;

        if (uz != INVALID_RANK && uz != self)
            for (int64_t k = nz - gs; k < nz; k++)
                for (int64_t j = 0; j < ny; j++)
                    for (int64_t i = 0; i < nx; i++)
                        dot[ijk_indx(i, j, k, gfs)] = 47;
    }
}

static int verify_dot(struct ngfs *gfs, int vstart, int nvars)
{
    const double tol = 1.0e-12;
    double max_err = 0.0;

    for (int v = 0; v < nvars; v++)
    {
        double *dot = gfs->vars[v + vstart]->dot;
        for (int64_t k = 0; k < gfs->nz; k++)
        {
            const double z = gfs->z0 + k * gfs->dz;
            for (int64_t j = 0; j < gfs->ny; j++)
            {
                const double y = gfs->y0 + j * gfs->dy;
                for (int64_t i = 0; i < gfs->nx; i++)
                {
                    const double x = gfs->x0 + i * gfs->dx;
                    const double expected = test_func(v + vstart, x, y, z);
                    const double got = dot[ijk_indx(i, j, k, gfs)];
                    const double err = fabs(got - expected);
                    if (err > max_err)
                        max_err = err;
                    if (err > tol)
                    {
                        fprintf(stderr,
                                "Rank %d: var %d mismatch at (%lld,%lld,%lld): "
                                "got %.16e expected %.16e (err %.3e)\n",
                                gfs->domain.rank, v + vstart,
                                (long long)i, (long long)j, (long long)k,
                                got, expected, err);
                        return -1;
                    }
                }
            }
        }
    }
    if (gfs->domain.rank == 0)
        fprintf(stderr, "  vars [%d..%d] max error: %.3e\n",
                vstart, vstart + nvars - 1, max_err);
    return 0;
}

/* Output vars[0]->dot as JSON for Python verification */
static void output_json(struct ngfs *gfs)
{
    char fname[128];
    snprintf(fname, sizeof(fname), "Var0_rank_%d.json", gfs->domain.rank);
    FILE *f = fopen(fname, "w");

    fprintf(f, "{\n");
    fprintf(f, "    \"nx\": %ld,\n", (long)gfs->nx);
    fprintf(f, "    \"ny\": %ld,\n", (long)gfs->ny);
    fprintf(f, "    \"nz\": %ld,\n", (long)gfs->nz);
    fprintf(f, "    \"dx\": %20.16e,\n", gfs->dx);
    fprintf(f, "    \"dy\": %20.16e,\n", gfs->dy);
    fprintf(f, "    \"dz\": %20.16e,\n", gfs->dz);
    fprintf(f, "    \"x0\": %20.16e,\n", gfs->x0);
    fprintf(f, "    \"y0\": %20.16e,\n", gfs->y0);
    fprintf(f, "    \"z0\": %20.16e,\n", gfs->z0);
    fprintf(f, "    \"local_i0\": %ld,\n", (long)gfs->domain.local_i0);
    fprintf(f, "    \"local_j0\": %ld,\n", (long)gfs->domain.local_j0);
    fprintf(f, "    \"local_k0\": %ld,\n", (long)gfs->domain.local_k0);
    fprintf(f, "    \"global_ni\": %ld,\n", (long)gfs->domain.global_ni);
    fprintf(f, "    \"global_nj\": %ld,\n", (long)gfs->domain.global_nj);
    fprintf(f, "    \"global_nk\": %ld,\n", (long)gfs->domain.global_nk);
    fprintf(f, "    \"global_x0\": %20.16e,\n", gfs->domain.global_x0);
    fprintf(f, "    \"global_y0\": %20.16e,\n", gfs->domain.global_y0);
    fprintf(f, "    \"global_z0\": %20.16e,\n", gfs->domain.global_z0);
    fprintf(f, "    \"rank\": %d,\n", gfs->domain.rank);
    fprintf(f, "    \"mpi_size\": %d,\n", gfs->domain.mpi_size);
    fprintf(f, "    \"gs\": %d,\n", gfs->gs);
    fprintf(f, "    \"lower_x_ghost\": %s,\n",
            (gfs->domain.lower_x_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"upper_x_ghost\": %s,\n",
            (gfs->domain.upper_x_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"lower_y_ghost\": %s,\n",
            (gfs->domain.lower_y_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"upper_y_ghost\": %s,\n",
            (gfs->domain.upper_y_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"lower_z_ghost\": %s,\n",
            (gfs->domain.lower_z_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"upper_z_ghost\": %s,\n",
            (gfs->domain.upper_z_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"data\": [ ");

    double *dot = gfs->vars[0]->dot;
    for (int64_t k = 0; k < gfs->nz; k++)
    {
        fprintf(f, "%s[\n", k ? "," : "");
        for (int64_t j = 0; j < gfs->ny; j++)
        {
            fprintf(f, "%s[\n", j ? "," : "");
            for (int64_t i = 0; i < gfs->nx; i++)
            {
                fprintf(f, "%s%20.16e", i ? "," : "",
                        dot[ijk_indx(i, j, k, gfs)]);
            }
            fprintf(f, "]");
        }
        fprintf(f, "]");
    }
    fprintf(f, "]\n}\n");
    fclose(f);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (argc < 4 || argc > 7)
    {
        if (mpi_rank == 0)
            fprintf(stderr, "Usage: %s NX NY NZ [periodic_x periodic_y periodic_z]\n",
                    argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int64_t global_nx = atoi(argv[1]);
    const int64_t global_ny = atoi(argv[2]);
    const int64_t global_nz = atoi(argv[3]);

    int per_x = 1, per_y = 1, per_z = 0; /* Maxwell default: periodic x,y */
    if (argc >= 7)
    {
        per_x = atoi(argv[4]);
        per_y = atoi(argv[5]);
        per_z = atoi(argv[6]);
    }

    size_t dims[3] = { global_nx, global_ny, global_nz };
    size_t topology[3];
    automatic_topology(3, dims, mpi_size, topology);

    const int gs = 2;
    const int n_evol = 9; /* same as Maxwell */
    const int n_aux = 5;

    struct ngfs gfs = { .vars = NULL, .auxvars = NULL };

    setup_3d_domain(topology[0], topology[1], topology[2], mpi_rank,
                    global_nx, global_ny, global_nz, gs,
                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, &gfs.domain,
                    per_x ? PERIODIC : NON_PERIODIC,
                    per_y ? PERIODIC : NON_PERIODIC,
                    per_z ? PERIODIC : NON_PERIODIC);

    ngfs_allocate(n_evol, n_aux, &gfs);

    /* --- Test EVOLVED sync --- */
    /* Set dot = new for testing (sync_vars syncs ->dot) */
    for (int v = 0; v < n_evol; v++)
        gfs.vars[v]->dot = gfs.vars[v]->new;

    fill_var_dot(&gfs, 0, n_evol);
    corrupt_ghost_dot(&gfs, 0, n_evol);
    sync_vars(&gfs, EVOLVED);

    if (mpi_rank == 0)
        fprintf(stderr, "Verifying EVOLVED sync (%d vars)...\n", n_evol);
    if (verify_dot(&gfs, 0, n_evol) != 0)
    {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Output vars[0] JSON for Python verification */
    output_json(&gfs);

    /* --- Test AUX sync --- */
    /* aux vars have dot = new by default (set in gf_aux_allocate) */
    fill_var_dot(&gfs, n_evol, n_aux);
    corrupt_ghost_dot(&gfs, n_evol, n_aux);
    sync_vars(&gfs, AUX);

    if (mpi_rank == 0)
        fprintf(stderr, "Verifying AUX sync (%d vars)...\n", n_aux);
    if (verify_dot(&gfs, n_evol, n_aux) != 0)
    {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (mpi_rank == 0)
        fprintf(stderr, "PASSED\n");

    ngfs_deallocate(&gfs);
    cleanup_3d_domain(&gfs.domain);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
