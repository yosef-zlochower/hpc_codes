/* Operator-level test for boundary_truncation_3d, the diagnostic
 * that estimates the local truncation error tau_h(u_h) at every
 * cell adjacent to a Neumann face owned by this rank.  This is
 * the building block for Phase 7's deferred-correction step
 * (see Plan.md sec. 7.2 + 7.3).
 *
 * Strategy: choose u^*(x,y,z) = x^3 + y^3 + z^3 -- a cubic for
 * which u_xxx = u_yyy = u_zzz = 6 everywhere, so the analytic
 * boundary truncation is known exactly:
 *
 *   tau_h |_{lower-x, i=1} = +(h_x / 24) * 6 = h_x / 4
 *   tau_h |_{upper-x, i=nx-2} = -(h_x / 24) * 6 = -h_x / 4
 *
 * and similarly for y, z.  At cells where multiple Neumann faces
 * meet (corners, edges) the per-axis contributions accumulate
 * additively.  Set up an N-N-N domain at [0,1]^3, fill u_h with
 * u^* at every cell (interior + ghost coordinates use the formula
 * directly so the ghost values are *exact* u^*(ghost coords),
 * not the simple-mirror approximation), call
 * boundary_truncation_3d, and compare against the analytic value
 * to round-off.  Pass/fail is checked in C; no Python verifier
 * required.
 *
 * This is a stronger check than convergence-rate measurement: it
 * exercises the helper's formula and corner-cell accumulation in
 * isolation, decoupling it from the subsequent V-cycle correction
 * pass.  If a future change to apply_bc_3d introduces a different
 * ghost-mirror order or sign convention, this test will fire.
 */
#include "domain.h"
#include "gf.h"
#include "comm.h"
#include "multigrid.h"
#include "bc.h"
#include "problem.h"
#include "gauss_seidel.h" /* VAR_SOL */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double u_cubic(double x, double y, double z)
{
    return x * x * x + y * y * y + z * z * z;
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

    /* All-Neumann layout, cubic exact solution, [0,1]^3. */
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

    /* boundary_truncation_3d needs gfs->bc to identify which faces
     * are Neumann.  Build a minimal homogeneous-Neumann spec; the
     * .value callbacks are unused (the helper reads only u_h, not q).
     * Must malloc -- ngfs_3d_deallocate calls free() on gfs->bc. */
    gfs.bc = malloc(sizeof(struct bc_spec_t));
    if (!gfs.bc) {
        fprintf(stderr, "malloc bc_spec_t failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    for (int f = 0; f < NUM_FACES; f++) {
        gfs.bc->face[f].kind        = BC_NEUMANN;
        gfs.bc->face[f].homogeneous = true;
        gfs.bc->face[f].value       = NULL;
    }

    /* Fill var 0 with u^* at every cell using the array's coordinate
     * formula (gfs->x0 already accounts for the -h/2 shift, so the
     * formula is exact at every cell including ghosts). */
    for (int64_t k = 0; k < gfs.nz; k++) {
        const double z = gfs.z0 + k * gfs.dz;
        for (int64_t j = 0; j < gfs.ny; j++) {
            const double y = gfs.y0 + j * gfs.dy;
            for (int64_t i = 0; i < gfs.nx; i++) {
                const double x = gfs.x0 + i * gfs.dx;
                gfs.vars[0]->val[gf_indx_3d(&gfs, i, j, k)] = u_cubic(x, y, z);
            }
        }
    }
    sync_var_3d(&gfs, 0);

    /* Allocate tau buffer and compute. */
    double *tau = calloc((size_t)gfs.n, sizeof(double));
    boundary_truncation_3d(&gfs, 0, tau);

    /* Verify.  Expected per-axis contribution at boundary cells:
     *   lower-a, i_a == 1                   :  +h_a / 4
     *   upper-a, i_a == n_a (= nx_cells)    :  -h_a / 4
     * Multiple faces contribute additively at corners/edges. */
    const int64_t Nx = gfs.domain.global_nx_cells;
    const int64_t Ny = gfs.domain.global_ny_cells;
    const int64_t Nz = gfs.domain.global_nz_cells;
    const int gs = gfs.gs;

    /* Owned interior cells in *global* indices range over [1, N_a]. */
    const int64_t i_owned_lo = (gfs.domain.lower_x_rank != INVALID_RANK) ? gs : 1;
    const int64_t i_owned_hi = gfs.nx - ((gfs.domain.upper_x_rank != INVALID_RANK) ? gs : 1);
    const int64_t j_owned_lo = (gfs.domain.lower_y_rank != INVALID_RANK) ? gs : 1;
    const int64_t j_owned_hi = gfs.ny - ((gfs.domain.upper_y_rank != INVALID_RANK) ? gs : 1);
    const int64_t k_owned_lo = (gfs.domain.lower_z_rank != INVALID_RANK) ? gs : 1;
    const int64_t k_owned_hi = gfs.nz - ((gfs.domain.upper_z_rank != INVALID_RANK) ? gs : 1);

    const bool x_lo_owned_neumann = (gfs.domain.lower_x_rank == INVALID_RANK);
    const bool x_hi_owned_neumann = (gfs.domain.upper_x_rank == INVALID_RANK);
    const bool y_lo_owned_neumann = (gfs.domain.lower_y_rank == INVALID_RANK);
    const bool y_hi_owned_neumann = (gfs.domain.upper_y_rank == INVALID_RANK);
    const bool z_lo_owned_neumann = (gfs.domain.lower_z_rank == INVALID_RANK);
    const bool z_hi_owned_neumann = (gfs.domain.upper_z_rank == INVALID_RANK);

    double local_err = 0.0;
    for (int64_t k = k_owned_lo; k < k_owned_hi; k++) {
        const int64_t gk = gfs.domain.local_k0 + k;
        for (int64_t j = j_owned_lo; j < j_owned_hi; j++) {
            const int64_t gj = gfs.domain.local_j0 + j;
            for (int64_t i = i_owned_lo; i < i_owned_hi; i++) {
                const int64_t gi = gfs.domain.local_i0 + i;
                double expected = 0.0;
                if (gi == 1 && x_lo_owned_neumann)  expected += dx / 4.0;
                if (gi == Nx && x_hi_owned_neumann) expected -= dx / 4.0;
                if (gj == 1 && y_lo_owned_neumann)  expected += dy / 4.0;
                if (gj == Ny && y_hi_owned_neumann) expected -= dy / 4.0;
                if (gk == 1 && z_lo_owned_neumann)  expected += dz / 4.0;
                if (gk == Nz && z_hi_owned_neumann) expected -= dz / 4.0;

                const double got = tau[gf_indx_3d(&gfs, i, j, k)];
                const double err = fabs(got - expected);
                if (err > local_err) local_err = err;
            }
        }
    }

    double global_err;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX,
                  gfs.domain.cart_comm);

    /* Round-off tolerance.  The 4-point FD on u = x^3 sums values
     * that are O(1) and produces (after division) the constant
     * u_xxx = 6 to round-off; tau_buf entries are O(h).  Allow a
     * generous epsilon scaled by the cell count. */
    const double tol = 1.0e-12 * (double)(nx_cells + ny_cells + nz_cells);

    if (mpi_rank == 0)
        printf("test_deferred_correction_3d: max |tau - analytic|_inf = %12.6e (tol %g)\n",
               global_err, tol);

    int rc = (global_err <= tol) ? EXIT_SUCCESS : EXIT_FAILURE;
    if (mpi_rank == 0)
        puts(rc == EXIT_SUCCESS ? "PASSED" : "FAILED");

    free(tau);
    ngfs_3d_deallocate(&gfs);
    cleanup_3d_domain(&gfs.domain);
    MPI_Finalize();
    return rc;
}
