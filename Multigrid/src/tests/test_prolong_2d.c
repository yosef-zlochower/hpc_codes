#include "comm.h"
#include "domain.h"
#include "gf.h"
#include "io.h"
#include "multigrid.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_gf_test_function(struct ngfs_2d *gfs);
static void corrupt_gf(struct ngfs_2d *gfs);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int mpi_size = -1;
    int mpi_rank = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s NX NY\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int global_nx_cells = atoi(argv[1]);
    const int global_ny_cells = atoi(argv[2]);

    size_t dims[2];
    dims[0] = global_nx_cells;
    dims[1] = global_ny_cells;
    size_t topology[2];
    automatic_topology(2, dims, mpi_size, topology);

    const int px = topology[0];
    const int py = topology[1];

    const double dx = 1.0 / global_nx_cells;
    const double dy = 1.0 / global_ny_cells;
    const double global_x0 = 0.0;
    const double global_y0 = 0.0;
    const int gs = 2;
    const int nvars = 1;
    const int min_cells = 4;

    struct ngfs_2d gfs;
    gfs.vars = NULL;

    setup_2d_domain(px, py, mpi_rank, global_nx_cells, global_ny_cells, gs,
                    global_x0, global_y0, dx, dy, &gfs.domain);
    ngfs_2d_allocate(nvars, &gfs);

    if (ngfs_2d_create_hierarchy(&gfs, min_cells) != 0)
    {
        fprintf(stderr, "Rank %d: hierarchy creation failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    struct ngfs_2d *child = gfs.child;
    if (child == NULL)
    {
        fprintf(stderr, "Rank %d: no child (grid too small?)\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Fill parent with f, sync */
    fill_gf_test_function(&gfs);
    corrupt_gf(&gfs);
    sync_var_2d(&gfs, 0);

    /* Restrict parent -> child (inject first to fill boundaries, then
     * full-weighting for interior points) */
    inject_var_2d(&gfs, 0, child, 0);
    restrict_var_2d(&gfs, 0, child, 0);

    /* Sync child */
    corrupt_gf(child);
    sync_var_2d(child, 0);

    /* Sentinel test (Finding 2): prolong must not touch parent physical
     * boundary points (those at global index 0 or N-1 per axis).
     * Overwrite them with a sentinel and verify they are unchanged after. */
    {
        const double SENTINEL = -9999.0;
        const int64_t pnx = gfs.nx;
        const int64_t pny = gfs.ny;
        if (gfs.domain.lower_x_rank == INVALID_RANK)
            for (int64_t jp = 0; jp < pny; jp++)
                gfs.vars[0]->val[gf_indx_2d(&gfs, 0, jp)] = SENTINEL;
        if (gfs.domain.upper_x_rank == INVALID_RANK)
            for (int64_t jp = 0; jp < pny; jp++)
                gfs.vars[0]->val[gf_indx_2d(&gfs, pnx - 1, jp)] = SENTINEL;
        if (gfs.domain.lower_y_rank == INVALID_RANK)
            for (int64_t ip = 0; ip < pnx; ip++)
                gfs.vars[0]->val[gf_indx_2d(&gfs, ip, 0)] = SENTINEL;
        if (gfs.domain.upper_y_rank == INVALID_RANK)
            for (int64_t ip = 0; ip < pnx; ip++)
                gfs.vars[0]->val[gf_indx_2d(&gfs, ip, pny - 1)] = SENTINEL;

        /* Prolong child -> parent (subtracts child from parent) */
        prolong_var_2d(child, 0, &gfs, 0);

        /* Verify sentinels are unchanged */
        if (gfs.domain.lower_x_rank == INVALID_RANK)
            for (int64_t jp = 0; jp < pny; jp++)
                if (gfs.vars[0]->val[gf_indx_2d(&gfs, 0, jp)] != SENTINEL)
                {
                    fprintf(stderr, "Rank %d: prolong touched lower-x boundary parent point at jp=%lld\n",
                            mpi_rank, (long long)jp);
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
        if (gfs.domain.upper_x_rank == INVALID_RANK)
            for (int64_t jp = 0; jp < pny; jp++)
                if (gfs.vars[0]->val[gf_indx_2d(&gfs, pnx - 1, jp)] != SENTINEL)
                {
                    fprintf(stderr, "Rank %d: prolong touched upper-x boundary parent point at jp=%lld\n",
                            mpi_rank, (long long)jp);
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
        if (gfs.domain.lower_y_rank == INVALID_RANK)
            for (int64_t ip = 0; ip < pnx; ip++)
                if (gfs.vars[0]->val[gf_indx_2d(&gfs, ip, 0)] != SENTINEL)
                {
                    fprintf(stderr, "Rank %d: prolong touched lower-y boundary parent point at ip=%lld\n",
                            mpi_rank, (long long)ip);
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
        if (gfs.domain.upper_y_rank == INVALID_RANK)
            for (int64_t ip = 0; ip < pnx; ip++)
                if (gfs.vars[0]->val[gf_indx_2d(&gfs, ip, pny - 1)] != SENTINEL)
                {
                    fprintf(stderr, "Rank %d: prolong touched upper-y boundary parent point at ip=%lld\n",
                            mpi_rank, (long long)ip);
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }

        /* Restore boundary to original f = 1.5x - 2y so verify_zeros.py sees
         * the expected values there (it skips them, but the JSON is cleaner). */
        if (gfs.domain.lower_x_rank == INVALID_RANK)
            for (int64_t jp = 0; jp < pny; jp++)
            {
                const double x = gfs.x0 + 0 * gfs.dx;
                const double y = gfs.y0 + jp * gfs.dy;
                gfs.vars[0]->val[gf_indx_2d(&gfs, 0, jp)] = 1.5 * x - 2.0 * y;
            }
        if (gfs.domain.upper_x_rank == INVALID_RANK)
            for (int64_t jp = 0; jp < pny; jp++)
            {
                const double x = gfs.x0 + (pnx - 1) * gfs.dx;
                const double y = gfs.y0 + jp * gfs.dy;
                gfs.vars[0]->val[gf_indx_2d(&gfs, pnx - 1, jp)] = 1.5 * x - 2.0 * y;
            }
        if (gfs.domain.lower_y_rank == INVALID_RANK)
            for (int64_t ip = 0; ip < pnx; ip++)
            {
                const double x = gfs.x0 + ip * gfs.dx;
                const double y = gfs.y0 + 0 * gfs.dy;
                gfs.vars[0]->val[gf_indx_2d(&gfs, ip, 0)] = 1.5 * x - 2.0 * y;
            }
        if (gfs.domain.upper_y_rank == INVALID_RANK)
            for (int64_t ip = 0; ip < pnx; ip++)
            {
                const double x = gfs.x0 + ip * gfs.dx;
                const double y = gfs.y0 + (pny - 1) * gfs.dy;
                gfs.vars[0]->val[gf_indx_2d(&gfs, ip, pny - 1)] = 1.5 * x - 2.0 * y;
            }
    }

    /* Sync parent */
    corrupt_gf(&gfs);
    sync_var_2d(&gfs, 0);

    /* Output parent for verification: should be 0 except at all physical
     * boundary points (lo and hi on both axes), which are never touched by
     * prolongation so they keep their original f values. */
    output_2d_gf(&gfs, 0, NULL);

    ngfs_2d_deallocate(&gfs);
    cleanup_2d_domain(&gfs.domain);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

static void fill_gf_test_function(struct ngfs_2d *gfs)
{
    for (int64_t j = 0; j < gfs->ny; j++)
    {
        const double y = gfs->y0 + j * gfs->dy;
        for (int64_t i = 0; i < gfs->nx; i++)
        {
            const double x = gfs->x0 + i * gfs->dx;
            gfs->vars[0]->val[gf_indx_2d(gfs, i, j)] = 1.5 * x - 2.0 * y;
        }
    }
}

static void corrupt_box(struct ngfs_2d *gfs, IndexBox box, double val)
{
    for (int64_t j = box.js; j < box.je; j++)
        for (int64_t i = box.is; i < box.ie; i++)
            gfs->vars[0]->val[gf_indx_2d(gfs, i, j)] = val;
}

static void corrupt_gf(struct ngfs_2d *gfs)
{
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int gs = gfs->gs;

    IndexBox x_lo = { .is = 0,      .ie = gs,      .js = 0, .je = ny, .ks = 0, .ke = 0 };
    IndexBox x_hi = { .is = nx - gs, .ie = nx,      .js = 0, .je = ny, .ks = 0, .ke = 0 };
    IndexBox y_lo = { .is = 0,      .ie = nx,      .js = 0, .je = gs, .ks = 0, .ke = 0 };
    IndexBox y_hi = { .is = 0,      .ie = nx,      .js = ny - gs, .je = ny, .ks = 0, .ke = 0 };

    if (gfs->domain.lower_x_rank != INVALID_RANK) corrupt_box(gfs, x_lo, 42);
    if (gfs->domain.upper_x_rank != INVALID_RANK) corrupt_box(gfs, x_hi, 45);
    if (gfs->domain.lower_y_rank != INVALID_RANK) corrupt_box(gfs, y_lo, 43);
    if (gfs->domain.upper_y_rank != INVALID_RANK) corrupt_box(gfs, y_hi, 46);
}

