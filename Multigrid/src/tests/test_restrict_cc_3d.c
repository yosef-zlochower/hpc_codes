/* Operator-level test for `restrict_var_cc_3d` -- the cell-centred
 * box-average restriction used by the V-cycle on every all-cell-
 * centred hierarchy (problems where every axis has at least one
 * Neumann face).
 *
 * Strategy: set up an N-N-N (all-Neumann) domain on the unit cube,
 * fill the fine grid with a linear test function f(x,y,z) at every
 * cell (interior + physical ghosts), call restrict_var_cc_3d, and
 * verify that every coarse interior cell equals f evaluated at the
 * coarse cell centre.  Restriction is the average of the eight
 * enclosed fine cells; for a linear function this is exactly f at
 * the centroid of the eight cells -- which is the coarse cell
 * centre by construction.  Pass/fail in C; no Python verifier
 * required.
 */
#include "domain.h"
#include "gf.h"
#include "comm.h"
#include "multigrid.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static double linear_f(double x, double y, double z)
{
    return 1.5 * x - 2.0 * y + z + 0.7;
}

static void fill_linear(struct ngfs_3d *gfs, int var)
{
    for (int64_t k = 0; k < gfs->nz; k++) {
        const double z = gfs->z0 + k * gfs->dz;
        for (int64_t j = 0; j < gfs->ny; j++) {
            const double y = gfs->y0 + j * gfs->dy;
            for (int64_t i = 0; i < gfs->nx; i++) {
                const double x = gfs->x0 + i * gfs->dx;
                gfs->vars[var]->val[gf_indx_3d(gfs, i, j, k)] = linear_f(x, y, z);
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

    /* All-Neumann layout: every axis cell-centred, N+2 points,
     * origin shifted by -h/2.  Linear f -> trivially compatible
     * boundary mirror, so we can fill the whole array including
     * ghosts without bothering with apply_bc. */
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

    /* Fill the fine grid -- including physical ghosts -- with the
     * linear test function evaluated at the array's *coordinate*
     * positions (gfs->x0 already accounts for the -h/2 shift on
     * cell-centred axes, so every cell, ghost or interior, gets
     * f(x_cell, y_cell, z_cell)). */
    fill_linear(&gfs, 0);
    sync_var_3d(&gfs, 0);

    restrict_var_cc_3d(&gfs, 0, gfs.child, 0);

    /* Verify every interior coarse cell holds f at its cell centre. */
    struct ngfs_3d *child = gfs.child;
    const int gs = child->gs;
    const int64_t ic_lo = (child->domain.lower_x_rank != INVALID_RANK) ? gs : 1;
    const int64_t ic_hi = child->nx - ((child->domain.upper_x_rank != INVALID_RANK) ? gs : 1);
    const int64_t jc_lo = (child->domain.lower_y_rank != INVALID_RANK) ? gs : 1;
    const int64_t jc_hi = child->ny - ((child->domain.upper_y_rank != INVALID_RANK) ? gs : 1);
    const int64_t kc_lo = (child->domain.lower_z_rank != INVALID_RANK) ? gs : 1;
    const int64_t kc_hi = child->nz - ((child->domain.upper_z_rank != INVALID_RANK) ? gs : 1);

    double local_err = 0.0;
    for (int64_t kc = kc_lo; kc < kc_hi; kc++) {
        const double zc = child->z0 + kc * child->dz;
        for (int64_t jc = jc_lo; jc < jc_hi; jc++) {
            const double yc = child->y0 + jc * child->dy;
            for (int64_t ic = ic_lo; ic < ic_hi; ic++) {
                const double xc = child->x0 + ic * child->dx;
                const double expected = linear_f(xc, yc, zc);
                const double got      = child->vars[0]->val[gf_indx_3d(child, ic, jc, kc)];
                const double err      = fabs(got - expected);
                if (err > local_err) local_err = err;
            }
        }
    }

    double global_err;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX,
                  gfs.domain.cart_comm);

    /* Linear functions are restricted exactly (8-cell average of a
     * linear field equals the field at the centroid).  The only
     * non-zero contribution is round-off; tolerance scales with N
     * via accumulated FP additions. */
    const double tol = 1.0e-12 * (double)(nx_cells + ny_cells + nz_cells);
    if (mpi_rank == 0)
        printf("test_restrict_cc_3d: max |u_H - f(coarse_centre)|_inf = %12.6e (tol %g)\n",
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
