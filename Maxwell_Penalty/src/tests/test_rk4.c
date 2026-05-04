/* test_rk4.c — Test RK4 time integration on a simple ODE.
 *
 * Usage: mpirun -np 1 ./test_rk4
 *
 * Solves dy/dt = -y with y(0) = 1, exact solution y(t) = exp(-t).
 * Runs at two different dt values and checks:
 *   1. The error at each dt is within tolerance
 *   2. The convergence rate is ~4th order (error ratio ~ 16 for dt/2)
 */
#include "comm.h"
#include "domain.h"
#include "gf.h"
#include "rk4.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* RHS: dy/dt = -y.  Called by RK4_Step at each stage. */
static void rhs_exp_decay(struct ngfs *gfs, const double t)
{
    /* dot = -new for all evolved variables */
    for (int v = 0; v < gfs->n_evol_vars; v++)
    {
        for (int64_t idx = 0; idx < gfs->n_tot; idx++)
        {
            gfs->vars[v]->dot[idx] = -gfs->vars[v]->new[idx];
        }
    }
    /* sync is a no-op for single-process, but call it for correctness */
    sync_vars(gfs, EVOLVED);
}

static double run_test(double dt, int nsteps)
{
    const int gs = 2;
    const int64_t nx = 8, ny = 8, nz = 8;
    const int n_evol = 1, n_aux = 0;

    struct ngfs gfs = { .vars = NULL, .auxvars = NULL };

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    setup_3d_domain(1, 1, 1, rank, nx, ny, nz, gs,
                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0, &gfs.domain,
                    NON_PERIODIC, NON_PERIODIC, NON_PERIODIC);

    ngfs_allocate(n_evol, n_aux, &gfs);

    /* Initial condition: y = 1.0 everywhere */
    for (int64_t idx = 0; idx < gfs.n_tot; idx++)
    {
        gfs.vars[0]->new[idx] = 1.0;
        gfs.vars[0]->old[idx] = 1.0;
    }

    /* Evolve */
    double t = 0.0;
    for (int step = 0; step < nsteps; step++)
    {
        RK4_Step(&gfs, t, dt, rhs_exp_decay);
        t += dt;
    }

    /* Check against exact solution y(t) = exp(-t) */
    const double exact = exp(-t);
    double max_err = 0.0;
    for (int64_t idx = 0; idx < gfs.n_tot; idx++)
    {
        double err = fabs(gfs.vars[0]->new[idx] - exact);
        if (err > max_err)
            max_err = err;
    }

    ngfs_deallocate(&gfs);
    cleanup_3d_domain(&gfs.domain);

    return max_err;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Run at two resolutions to check convergence rate */
    const double dt1 = 0.01;
    const double dt2 = dt1 / 2.0;
    const int nsteps1 = 100; /* t_final = 1.0 */
    const int nsteps2 = 200; /* t_final = 1.0 */

    const double err1 = run_test(dt1, nsteps1);
    const double err2 = run_test(dt2, nsteps2);

    if (rank == 0)
    {
        const double ratio = err1 / err2;
        fprintf(stderr, "RK4 test: dy/dt = -y, t_final = 1.0\n");
        fprintf(stderr, "  dt = %.4f : error = %.3e\n", dt1, err1);
        fprintf(stderr, "  dt = %.4f : error = %.3e\n", dt2, err2);
        fprintf(stderr, "  ratio = %.2f (expected ~16 for 4th order)\n", ratio);

        /* Error should be small */
        if (err1 > 1.0e-8)
        {
            fprintf(stderr, "FAILED: error too large at dt=%.4f\n", dt1);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* Convergence rate: err1/err2 should be ~2^4 = 16 */
        if (ratio < 12.0 || ratio > 20.0)
        {
            fprintf(stderr, "FAILED: convergence ratio %.2f not near 16\n", ratio);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fprintf(stderr, "PASSED\n");
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
