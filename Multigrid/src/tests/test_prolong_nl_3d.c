#include "comm.h"
#include "domain.h"
#include "gf.h"
#include "io.h"
#include "multigrid.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void fill_nl_test(struct ngfs_3d *gfs);
static void corrupt_gf(struct ngfs_3d *gfs);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int mpi_size = -1;
    int mpi_rank = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s NX NY NZ\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int global_nx_cells = atoi(argv[1]);
    const int global_ny_cells = atoi(argv[2]);
    const int global_nz_cells = atoi(argv[3]);

    size_t dims[3];
    dims[0] = global_nx_cells;
    dims[1] = global_ny_cells;
    dims[2] = global_nz_cells;
    size_t topology[3];
    automatic_topology(3, dims, mpi_size, topology);

    const double dx = 1.0 / global_nx_cells;
    const double dy = 1.0 / global_ny_cells;
    const double dz = 1.0 / global_nz_cells;
    const int gs = 2;
    const int nvars = 1;
    const int min_cells = 4;

    struct ngfs_3d gfs;
    gfs.vars = NULL;

    setup_3d_domain(topology[0], topology[1], topology[2], mpi_rank,
                    global_nx_cells, global_ny_cells, global_nz_cells,
                    /*neumann_face=*/NULL,
                    gs, 0.0, 0.0, 0.0, dx, dy, dz, &gfs.domain);
    ngfs_3d_allocate(nvars, &gfs);

    if (ngfs_3d_create_hierarchy(&gfs, min_cells) != 0)
    {
        fprintf(stderr, "Rank %d: hierarchy creation failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    struct ngfs_3d *child = gfs.child;
    if (child == NULL)
    {
        fprintf(stderr, "Rank %d: no child (grid too small?)\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Fill child (coarse) with F = X*(1-X)*Y*(1-Y)*Z*(1-Z), sync.
     * Parent (fine) starts at zero from calloc. */
    fill_nl_test(child);
    corrupt_gf(child);
    sync_var_3d(child, 0);

    /* Prolong child -> parent: parent -= P[F].
     * Since parent is zero, result is parent = -P[F]. */
    prolong_var_3d(child, 0, &gfs, 0);

    /* Sync parent */
    corrupt_gf(&gfs);
    sync_var_3d(&gfs, 0);

    /* Output parent (fine) data for verification */
    output_3d_gf(&gfs, 0, NULL);

    ngfs_3d_deallocate(&gfs);
    cleanup_3d_domain(&gfs.domain);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

static void fill_nl_test(struct ngfs_3d *gfs)
{
    for (int64_t k = 0; k < gfs->nz; k++)
    {
        const double z = gfs->z0 + k * gfs->dz;
        for (int64_t j = 0; j < gfs->ny; j++)
        {
            const double y = gfs->y0 + j * gfs->dy;
            for (int64_t i = 0; i < gfs->nx; i++)
            {
                const double x = gfs->x0 + i * gfs->dx;
                gfs->vars[0]->val[gf_indx_3d(gfs, i, j, k)] =
                    x * (1.0 - x) * y * (1.0 - y) * z * (1.0 - z);
            }
        }
    }
}

static void corrupt_box(struct ngfs_3d *gfs, IndexBox box, double val)
{
    for (int64_t k = box.ks; k < box.ke; k++)
        for (int64_t j = box.js; j < box.je; j++)
            for (int64_t i = box.is; i < box.ie; i++)
                gfs->vars[0]->val[gf_indx_3d(gfs, i, j, k)] = val;
}

static void corrupt_gf(struct ngfs_3d *gfs)
{
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t nz = gfs->nz;
    const int gs = gfs->gs;

    IndexBox x_lo = { .is = 0,       .ie = gs,    .js = 0, .je = ny, .ks = 0,       .ke = nz };
    IndexBox x_hi = { .is = nx - gs, .ie = nx,    .js = 0, .je = ny, .ks = 0,       .ke = nz };
    IndexBox y_lo = { .is = 0,       .ie = nx,    .js = 0, .je = gs, .ks = 0,       .ke = nz };
    IndexBox y_hi = { .is = 0,       .ie = nx,    .js = ny - gs, .je = ny, .ks = 0, .ke = nz };
    IndexBox z_lo = { .is = 0,       .ie = nx,    .js = 0, .je = ny, .ks = 0,       .ke = gs };
    IndexBox z_hi = { .is = 0,       .ie = nx,    .js = 0, .je = ny, .ks = nz - gs, .ke = nz };

    if (gfs->domain.lower_x_rank != INVALID_RANK) corrupt_box(gfs, x_lo, 42);
    if (gfs->domain.upper_x_rank != INVALID_RANK) corrupt_box(gfs, x_hi, 45);
    if (gfs->domain.lower_y_rank != INVALID_RANK) corrupt_box(gfs, y_lo, 43);
    if (gfs->domain.upper_y_rank != INVALID_RANK) corrupt_box(gfs, y_hi, 46);
    if (gfs->domain.lower_z_rank != INVALID_RANK) corrupt_box(gfs, z_lo, 44);
    if (gfs->domain.upper_z_rank != INVALID_RANK) corrupt_box(gfs, z_hi, 47);
}

