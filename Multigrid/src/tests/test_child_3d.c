#include "domain.h"
#include "gf.h"
#include "multigrid.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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

    const int px = topology[0];
    const int py = topology[1];
    const int pz = topology[2];

    const double dx = 1.0 / global_nx_cells;
    const double dy = 1.0 / global_ny_cells;
    const double dz = 1.0 / global_nz_cells;
    const double global_x0 = 0.0;
    const double global_y0 = 0.0;
    const double global_z0 = 0.0;
    const int gs = 2;
    const int nvars = 1;
    const int min_cells = 4;

    struct ngfs_3d gfs;
    gfs.vars = NULL;

    setup_3d_domain(px, py, pz, mpi_rank, global_nx_cells, global_ny_cells, global_nz_cells,
                    gs, global_x0, global_y0, global_z0,
                    dx, dy, dz, &gfs.domain);
    ngfs_3d_allocate(nvars, &gfs);

    /* Build the full hierarchy */
    if (ngfs_3d_create_hierarchy(&gfs, min_cells) != 0)
    {
        fprintf(stderr, "Rank %d: ngfs_3d_create_hierarchy failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* The root must have a child (grid is large enough to coarsen at least once) */
    struct ngfs_3d *child = gfs.child;
    if (child == NULL)
    {
        fprintf(stderr, "Rank %d: expected at least one child\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* create_child must return -1 when the node already has a child */
    if (ngfs_3d_create_child(&gfs, min_cells) != -1)
    {
        fprintf(stderr, "Rank %d: expected -1 when parent->child != NULL\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Walk the full hierarchy: verify every level, not just the first child */
    {
        struct ngfs_3d *par = &gfs;
        int level = 0;
        while (par->child != NULL)
        {
            struct ngfs_3d *chd = par->child;
            level++;

            /* Spacing must double at each level */
            if (chd->dx != 2.0 * par->dx)
            {
                fprintf(stderr, "Rank %d level %d: dx mismatch: child->dx=%g 2*par->dx=%g\n",
                        mpi_rank, level, chd->dx, 2.0 * par->dx);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            if (chd->dy != 2.0 * par->dy)
            {
                fprintf(stderr, "Rank %d level %d: dy mismatch\n", mpi_rank, level);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            if (chd->dz != 2.0 * par->dz)
            {
                fprintf(stderr, "Rank %d level %d: dz mismatch\n", mpi_rank, level);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            /* Global dimensions follow the coarsening formula:
             * child cell count = parent cell count / 2 (per axis). */
            const size_t exp_cells_x = par->domain.global_nx_cells / 2;
            const size_t exp_cells_y = par->domain.global_ny_cells / 2;
            const size_t exp_cells_z = par->domain.global_nz_cells / 2;
            if (chd->domain.global_nx_cells != (int64_t)exp_cells_x)
            {
                fprintf(stderr, "Rank %d level %d: global_nx_cells mismatch: got %lu expected %lu\n",
                        mpi_rank, level, chd->domain.global_nx_cells, exp_cells_x);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            if (chd->domain.global_ny_cells != (int64_t)exp_cells_y)
            {
                fprintf(stderr, "Rank %d level %d: global_ny_cells mismatch: got %lu expected %lu\n",
                        mpi_rank, level, chd->domain.global_ny_cells, exp_cells_y);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            if (chd->domain.global_nz_cells != (int64_t)exp_cells_z)
            {
                fprintf(stderr, "Rank %d level %d: global_nz_cells mismatch: got %lu expected %lu\n",
                        mpi_rank, level, chd->domain.global_nz_cells, exp_cells_z);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            /* Ghost size must be preserved */
            if (chd->gs != gfs.gs)
            {
                fprintf(stderr, "Rank %d level %d: gs mismatch\n", mpi_rank, level);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            /* Ghost face presence must match the root at every level */
            if ((chd->domain.lower_x_rank == INVALID_RANK) !=
                (gfs.domain.lower_x_rank == INVALID_RANK) ||
                (chd->domain.upper_x_rank == INVALID_RANK) !=
                (gfs.domain.upper_x_rank == INVALID_RANK) ||
                (chd->domain.lower_y_rank == INVALID_RANK) !=
                (gfs.domain.lower_y_rank == INVALID_RANK) ||
                (chd->domain.upper_y_rank == INVALID_RANK) !=
                (gfs.domain.upper_y_rank == INVALID_RANK) ||
                (chd->domain.lower_z_rank == INVALID_RANK) !=
                (gfs.domain.lower_z_rank == INVALID_RANK) ||
                (chd->domain.upper_z_rank == INVALID_RANK) !=
                (gfs.domain.upper_z_rank == INVALID_RANK))
            {
                fprintf(stderr, "Rank %d level %d: ghost face presence mismatch\n",
                        mpi_rank, level);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            /* Parent/child pointer integrity */
            if (chd->parent != par || par->child != chd)
            {
                fprintf(stderr, "Rank %d level %d: parent/child pointer mismatch\n",
                        mpi_rank, level);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            par = chd;
        }
    }

    /* Confirm create_child returns 1 (final depth) at the leaf */
    struct ngfs_3d *leaf = &gfs;
    while (leaf->child)
        leaf = leaf->child;
    if (ngfs_3d_create_child(leaf, min_cells) != 1)
    {
        fprintf(stderr, "Rank %d: expected return 1 (final depth) at leaf\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Cleanup: deallocate root; recursively frees entire hierarchy */
    ngfs_3d_deallocate(&gfs);
    cleanup_3d_domain(&gfs.domain);

    if (mpi_rank == 0)
        printf("PASSED\n");

    MPI_Finalize();
    return EXIT_SUCCESS;
}
