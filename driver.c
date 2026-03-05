#include "comm.h"
#include "domain.h"
#include "gauss_seidel.h"
#include "gf.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (argc < 4)
    {
        if (mpi_rank == 0)
            fprintf(stderr, "Usage: %s NX NY NZ [omega] [n_smooth] [n_iters]\n",
                    argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int global_nx = atoi(argv[1]);
    const int global_ny = atoi(argv[2]);
    const int global_nz = atoi(argv[3]);
    const double omega = (argc > 4) ? atof(argv[4]) : 1.5;
    const int n_smooth = (argc > 5) ? atoi(argv[5]) : 2;
    const int n_iters  = (argc > 6) ? atoi(argv[6]) : 20;
    const double tol = 1.0e-10;

    /* ---- Domain decomposition ---- */
    size_t dims[3]     = { (size_t)global_nx, (size_t)global_ny, (size_t)global_nz };
    size_t topology[3] = { 0, 0, 0 };
    automatic_topology(3, dims, (size_t)mpi_size, topology);

    if (mpi_rank == 0)
    {
        printf("Grid:     %d x %d x %d\n", global_nx, global_ny, global_nz);
        printf("Topology: %zu x %zu x %zu  (%d ranks)\n",
               topology[0], topology[1], topology[2], mpi_size);
        printf("omega = %.4f,  n_smooth = %d,  n_iters = %d\n",
               omega, n_smooth, n_iters);
    }

    const int px = (int)topology[0];
    const int py = (int)topology[1];
    const int pz = (int)topology[2];

    /* Grid spacing for a node-based grid on [0,1]^3 */
    const double dx = 1.0 / (global_nx - 1);
    const double dy = 1.0 / (global_ny - 1);
    const double dz = 1.0 / (global_nz - 1);

    /* ---- Allocate grid functions (gs=1 for 2nd-order stencil) ---- */
    struct ngfs_3d gfs;
    gfs.vars = NULL;

    setup_3d_domain(px, py, pz, mpi_rank,
                    global_nx, global_ny, global_nz,
                    /*gs=*/1,
                    0.0, 0.0, 0.0,
                    dx, dy, dz,
                    &gfs.domain);

    ngfs_3d_allocate(/*nvars=*/3, &gfs);

    /* ---- Manufactured solution:  u_exact = sin(pi*x)*sin(pi*y)*sin(pi*z)
     *      => Laplacian u_exact = -3*pi^2 * u_exact
     *      Set VAR_RHS = -3*pi^2 * u_exact, VAR_SOL = 0                  ---- */
    const double pi   = acos(-1.0);
    const double pi2  = pi * pi;

    double *rhs = gfs.vars[VAR_RHS]->val;
    double *sol = gfs.vars[VAR_SOL]->val;

    for (int64_t k = 0; k < gfs.nz; k++)
    {
        const double z = gfs.z0 + k * gfs.dz;
        for (int64_t j = 0; j < gfs.ny; j++)
        {
            const double y = gfs.y0 + j * gfs.dy;
            for (int64_t i = 0; i < gfs.nx; i++)
            {
                const double x   = gfs.x0 + i * gfs.dx;
                const int64_t idx = gf_indx_3d(&gfs, i, j, k);
                rhs[idx] = -3.0 * pi2 * sin(pi * x) * sin(pi * y) * sin(pi * z);
                sol[idx] = 0.0;
            }
        }
    }
    apply_bc_3d(&gfs, VAR_SOL);

    /* ---- Solve with Gauss-Seidel ---- */
    double norm = calc_defect_3d(&gfs);
    if (mpi_rank == 0)
        printf("\niter %4d  |defect|_inf = %12.6e\n", 0, norm);

    for (int it = 1; it <= n_iters; it++)
    {
        gauss_seidel_3d(&gfs, n_smooth, omega);
        norm = calc_defect_3d(&gfs);
        if (mpi_rank == 0)
            printf("iter %4d  |defect|_inf = %12.6e\n", it, norm);

        if (norm < tol)
        {
            break;
        }
    }

    /* ---- Error vs exact solution ---- */
    double local_err = 0.0;
    for (int64_t k = gfs.gs; k < gfs.nz - gfs.gs; k++)
    {
        const double z = gfs.z0 + k * gfs.dz;
        for (int64_t j = gfs.gs; j < gfs.ny - gfs.gs; j++)
        {
            const double y = gfs.y0 + j * gfs.dy;
            for (int64_t i = gfs.gs; i < gfs.nx - gfs.gs; i++)
            {
                const double x      = gfs.x0 + i * gfs.dx;
                const int    idx    = gf_indx_3d(&gfs, i, j, k);
                const double u_exact = sin(pi * x) * sin(pi * y) * sin(pi * z);
                const double err    = fabs(sol[idx] - u_exact);
                if (err > local_err)
                    local_err = err;
            }
        }
    }
    double global_err;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX,
                  gfs.domain.cart_comm);
    if (mpi_rank == 0)
        printf("\n|u - u_exact|_inf = %12.6e\n", global_err);

    /* ---- Cleanup ---- */
    cleanup_3d_domain(&gfs.domain);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
