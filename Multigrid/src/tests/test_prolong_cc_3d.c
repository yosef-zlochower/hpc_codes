/* Operator-level test for `prolong_var_cc_3d` -- the cell-centred
 * trilinear prolongation used by the V-cycle on every all-cell-
 * centred hierarchy.
 *
 * Strategy: set up an N-N-N (all-Neumann) domain, fill the *coarse*
 * grid with a linear test function g(x,y,z) at every cell (interior
 * + physical ghosts), zero the fine grid, then call
 * prolong_var_cc_3d.  The cc trilinear weights (3/4)^near (1/4)^far
 * are exact for any linear function, so prolong_var_cc_3d (which
 * does fine -= P(coarse)) leaves every interior fine cell with
 * value -g(fine_cell_centre).  Boundary fine cells (ghosts at i=0
 * and i=nx-1 on every axis) are not touched by the kernel; we don't
 * verify them.  Pass/fail in C; no Python verifier required.
 */
#include "domain.h"
#include "gf.h"
#include "comm.h"
#include "multigrid.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double linear_g(double x, double y, double z)
{
    return -0.3 * x + 1.7 * y - 0.9 * z + 1.1;
}

static void fill_linear(struct ngfs_3d *gfs, int var)
{
    for (int64_t k = 0; k < gfs->nz; k++) {
        const double z = gfs->z0 + k * gfs->dz;
        for (int64_t j = 0; j < gfs->ny; j++) {
            const double y = gfs->y0 + j * gfs->dy;
            for (int64_t i = 0; i < gfs->nx; i++) {
                const double x = gfs->x0 + i * gfs->dx;
                gfs->vars[var]->val[gf_indx_3d(gfs, i, j, k)] = linear_g(x, y, z);
            }
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int mpi_size = -1, mpi_rank = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (argc != 4) {
        if (mpi_rank == 0)
            fprintf(stderr, "Usage: %s NX_cells NY_cells NZ_cells\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int64_t nx_cells = atoll(argv[1]);
    const int64_t ny_cells = atoll(argv[2]);
    const int64_t nz_cells = atoll(argv[3]);

    size_t dims[3] = { (size_t)nx_cells, (size_t)ny_cells, (size_t)nz_cells };
    size_t topology[3] = { 0, 0, 0 };
    automatic_topology(3, dims, (size_t)mpi_size, topology);

    const bool neumann_face[6] = { true, true, true, true, true, true };
    const double dx = 1.0 / (double)nx_cells;
    const double dy = 1.0 / (double)ny_cells;
    const double dz = 1.0 / (double)nz_cells;

    struct ngfs_3d gfs;
    gfs.vars = NULL;

    if (setup_3d_domain((int)topology[0], (int)topology[1], (int)topology[2],
                        mpi_rank,
                        nx_cells, ny_cells, nz_cells,
                        neumann_face,
                        /*gs=*/1,
                        /*a_x=*/0.0, /*a_y=*/0.0, /*a_z=*/0.0,
                        dx, dy, dz, &gfs.domain) != 0) {
        if (mpi_rank == 0)
            fprintf(stderr, "setup_3d_domain failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    ngfs_3d_allocate(/*nvars=*/1, &gfs);

    if (ngfs_3d_create_hierarchy(&gfs, /*min_cells=*/4) != 0) {
        if (mpi_rank == 0)
            fprintf(stderr, "hierarchy build failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if (gfs.child == NULL) {
        if (mpi_rank == 0)
            fprintf(stderr, "no child level (grid too coarse)\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Coarse grid: fill with linear g at every cell (interior + ghost). */
    struct ngfs_3d *child = gfs.child;
    fill_linear(child, 0);
    sync_var_3d(child, 0);

    /* Fine grid: zero everywhere.  prolong_var_cc_3d will subtract
     * P(child) from the fine values; with fine = 0, the result is
     * fine = -P(child). */
    memset(gfs.vars[0]->val, 0, (size_t)gfs.n * sizeof(double));

    prolong_var_cc_3d(child, 0, &gfs, 0);

    /* Verify every interior fine cell satisfies fine ~= -g(fine_cell_centre).
     * The kernel skips ghost rows (i=0 and i=nx-1 on every axis on
     * the boundary rank, gs-wide MPI layer otherwise). */
    const int gs = gfs.gs;
    const int64_t ip_lo = (gfs.domain.lower_x_rank != INVALID_RANK) ? gs : 1;
    const int64_t ip_hi = gfs.nx - ((gfs.domain.upper_x_rank != INVALID_RANK) ? gs : 1);
    const int64_t jp_lo = (gfs.domain.lower_y_rank != INVALID_RANK) ? gs : 1;
    const int64_t jp_hi = gfs.ny - ((gfs.domain.upper_y_rank != INVALID_RANK) ? gs : 1);
    const int64_t kp_lo = (gfs.domain.lower_z_rank != INVALID_RANK) ? gs : 1;
    const int64_t kp_hi = gfs.nz - ((gfs.domain.upper_z_rank != INVALID_RANK) ? gs : 1);

    double local_err = 0.0;
    for (int64_t kp = kp_lo; kp < kp_hi; kp++) {
        const double z = gfs.z0 + kp * gfs.dz;
        for (int64_t jp = jp_lo; jp < jp_hi; jp++) {
            const double y = gfs.y0 + jp * gfs.dy;
            for (int64_t ip = ip_lo; ip < ip_hi; ip++) {
                const double x = gfs.x0 + ip * gfs.dx;
                const double expected = -linear_g(x, y, z);
                const double got      = gfs.vars[0]->val[gf_indx_3d(&gfs, ip, jp, kp)];
                const double err      = fabs(got - expected);
                if (err > local_err) local_err = err;
            }
        }
    }

    double global_err;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX,
                  gfs.domain.cart_comm);

    /* Linear functions are prolongated exactly by the cc trilinear
     * weights; only round-off remains. */
    const double tol = 1.0e-12 * (double)(nx_cells + ny_cells + nz_cells);
    if (mpi_rank == 0)
        printf("test_prolong_cc_3d: max |u_h + g(fine_centre)|_inf = %12.6e (tol %g)\n",
               global_err, tol);

    int rc = (global_err <= tol) ? EXIT_SUCCESS : EXIT_FAILURE;
    if (mpi_rank == 0) {
        if (rc == EXIT_SUCCESS) puts("PASSED");
        else                    puts("FAILED");
    }

    ngfs_3d_deallocate(&gfs);
    cleanup_3d_domain(&gfs.domain);
    MPI_Finalize();
    return rc;
}
