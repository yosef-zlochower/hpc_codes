#include "comm.h"
#include "domain.h"
#include "gf.h"
#include "io.h"
#include "multigrid.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void fill_nl_test(struct ngfs_2d *gfs);
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

    const int global_nx = atoi(argv[1]);
    const int global_ny = atoi(argv[2]);

    size_t dims[2];
    dims[0] = global_nx;
    dims[1] = global_ny;
    size_t topology[2];
    automatic_topology(2, dims, mpi_size, topology);

    const double dx = 1.0 / (global_nx - 1);
    const double dy = 1.0 / (global_ny - 1);
    const int gs = 2;
    const int nvars = 1;
    const int min_cells = 4;

    struct ngfs_2d gfs;
    gfs.vars = NULL;

    setup_2d_domain(topology[0], topology[1], mpi_rank, global_nx, global_ny,
                    gs, 0.0, 0.0, dx, dy, &gfs.domain);
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

    /* Fill parent (fine) with f = x*(1-x)*y*(1-y), sync */
    fill_nl_test(&gfs);
    corrupt_gf(&gfs);
    sync_var_2d(&gfs, 0);

    /* Restrict parent -> child.
     * inject fills physical boundaries; restrict overwrites interior points
     * with the full-weighting stencil. */
    inject_var_2d(&gfs, 0, child, 0);
    restrict_var_2d(&gfs, 0, child, 0);

    /* Sync child */
    corrupt_gf(child);
    sync_var_2d(child, 0);

    /* Output child (coarse) data for verification */
    output_2d_gf(child, 0);

    ngfs_2d_deallocate(&gfs);
    cleanup_2d_domain(&gfs.domain);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

static void fill_nl_test(struct ngfs_2d *gfs)
{
    for (int64_t j = 0; j < gfs->ny; j++)
    {
        const double y = gfs->y0 + j * gfs->dy;
        for (int64_t i = 0; i < gfs->nx; i++)
        {
            const double x = gfs->x0 + i * gfs->dx;
            gfs->vars[0]->val[gf_indx_2d(gfs, i, j)] =
                x * (1.0 - x) * y * (1.0 - y);
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

    IndexBox x_lo = { .is = 0,       .ie = gs,    .js = 0, .je = ny, .ks = 0, .ke = 0 };
    IndexBox x_hi = { .is = nx - gs, .ie = nx,    .js = 0, .je = ny, .ks = 0, .ke = 0 };
    IndexBox y_lo = { .is = 0,       .ie = nx,    .js = 0, .je = gs, .ks = 0, .ke = 0 };
    IndexBox y_hi = { .is = 0,       .ie = nx,    .js = ny - gs, .je = ny, .ks = 0, .ke = 0 };

    if (gfs->domain.lower_x_rank != INVALID_RANK) corrupt_box(gfs, x_lo, 42);
    if (gfs->domain.upper_x_rank != INVALID_RANK) corrupt_box(gfs, x_hi, 45);
    if (gfs->domain.lower_y_rank != INVALID_RANK) corrupt_box(gfs, y_lo, 43);
    if (gfs->domain.upper_y_rank != INVALID_RANK) corrupt_box(gfs, y_hi, 46);
}

