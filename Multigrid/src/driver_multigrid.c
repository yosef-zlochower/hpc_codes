#include "comm.h"
#include "domain.h"
#include "gauss_seidel.h"
#include "gf.h"
#include "io.h"
#include "multigrid.h"
#include "parameter.h"
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

    if (argc < 2)
    {
        if (mpi_rank == 0)
            fprintf(stderr, "Usage: %s <params.toml>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    struct param_st param;
    if (parse_parameter_file(&param, argv[1]) != 0)
    {
        if (mpi_rank == 0)
            fprintf(stderr, "Failed to parse parameter file '%s'\n", argv[1]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (param.global_nx_cells < 1 || param.global_ny_cells < 1 || param.global_nz_cells < 1)
    {
        if (mpi_rank == 0)
            fprintf(stderr,
                    "Error: nx_cells=%lld ny_cells=%lld nz_cells=%lld — "
                    "each dimension needs at least 1 cell.\n",
                    (long long)param.global_nx_cells,
                    (long long)param.global_ny_cells,
                    (long long)param.global_nz_cells);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Grid points = cells + 1 (node-based grid) */
    const int64_t global_nx = param.global_nx_cells + 1;
    const int64_t global_ny = param.global_ny_cells + 1;
    const int64_t global_nz = param.global_nz_cells + 1;

    /* ---- Domain decomposition ---- */
    size_t dims[3]     = { (size_t)global_nx,
                           (size_t)global_ny,
                           (size_t)global_nz };
    size_t topology[3] = { 0, 0, 0 };
    automatic_topology(3, dims, (size_t)mpi_size, topology);

    const int px = (int)topology[0];
    const int py = (int)topology[1];
    const int pz = (int)topology[2];

    /* Grid spacing for a node-based grid on [0,1]^3 */
    const double dx = 1.0 / (global_nx - 1);
    const double dy = 1.0 / (global_ny - 1);
    const double dz = 1.0 / (global_nz - 1);

    /* ---- Allocate root grid (gs=1 for 7-point stencil) ---- */
    struct ngfs_3d gfs;
    gfs.vars = NULL;

    setup_3d_domain(px, py, pz, mpi_rank,
                    global_nx, global_ny, global_nz,
                    /*gs=*/1,
                    0.0, 0.0, 0.0,
                    dx, dy, dz,
                    &gfs.domain);

    ngfs_3d_allocate(/*nvars=*/3, &gfs);

    /* ---- Build coarse-grid hierarchy (multigrid only) ---- */
    if (param.use_multigrid)
    {
        if (ngfs_3d_create_hierarchy(&gfs, param.min_cells) != 0)
        {
            if (mpi_rank == 0)
                fprintf(stderr, "Hierarchy creation failed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    if (mpi_rank == 0)
    {
        printf("Grid:     %lld x %lld x %lld\n",
               (long long)global_nx,
               (long long)global_ny,
               (long long)global_nz);
        printf("Topology: %zu x %zu x %zu  (%d ranks)\n",
               topology[0], topology[1], topology[2], mpi_size);
        printf("Solver:   %s\n",
               param.use_multigrid ? "multigrid V-cycle" : "Gauss-Seidel");
        printf("omega = %.4f,  n_smooth = %d,  n_iters = %d,  tol = %g\n",
               param.omega, param.n_smooth, param.n_iters, param.tol);

        if (param.use_multigrid)
        {
            int nlevels = 1;
            struct ngfs_3d *p = gfs.child;
            while (p) { nlevels++; p = p->child; }
            printf("subcycles = %d,  min_cells = %d,  levels = %d\n",
                   param.subcycles, param.min_cells, nlevels);
        }
    }

    /* ---- Manufactured solution:  u_exact = sin(pi*x)*sin(pi*y)*sin(pi*z)
     *      => Laplacian u_exact = -3*pi^2 * u_exact
     *      Set VAR_RHS = -3*pi^2 * u_exact, VAR_SOL = 0               ---- */
    const double pi  = acos(-1.0);
    const double pi2 = pi * pi;

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
                const double  x   = gfs.x0 + i * gfs.dx;
                const int64_t idx = gf_indx_3d(&gfs, i, j, k);
                rhs[idx] = -3.0 * pi2 * sin(pi * x) * sin(pi * y) * sin(pi * z);
                sol[idx] = 0.0;
            }
        }
    }
    apply_bc_3d(&gfs, VAR_SOL);

    /* ---- Solve ---- */
    double norm = calc_defect_3d(&gfs);
    if (mpi_rank == 0)
        printf("\niter %4d  |defect|_inf = %12.6e\n", 0, norm);

    for (int it = 1; it <= param.n_iters; it++)
    {
        if (param.use_multigrid)
        {
            norm = vcycle_3d(&gfs, param.n_smooth, param.omega,
                             param.tol, param.subcycles);
        }
        else
        {
            gauss_seidel_3d(&gfs, param.n_smooth, param.omega);
            norm = calc_defect_3d(&gfs);
        }

        if (mpi_rank == 0)
            printf("iter %4d  |defect|_inf = %12.6e\n", it, norm);

        if (norm < param.tol)
            break;
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
                const double  x       = gfs.x0 + i * gfs.dx;
                const int64_t idx     = gf_indx_3d(&gfs, i, j, k);
                const double  u_exact = sin(pi * x) * sin(pi * y) * sin(pi * z);
                const double  err     = fabs(sol[idx] - u_exact);
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

    output_3d_gf(&gfs, 0);

    /* ---- Cleanup ---- */
    ngfs_3d_deallocate(&gfs);
    cleanup_3d_domain(&gfs.domain);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
