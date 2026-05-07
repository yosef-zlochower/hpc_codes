#include "comm.h"
#include "domain.h"
#include "gf.h"
#include "io.h"
#include "multigrid.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_gf_test_function(struct ngfs_3d *gfs);
static void corrupt_gf(struct ngfs_3d *gfs);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int mpi_size = -1;
    int mpi_rank = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s NX NY NZ [inject|restrict]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int global_nx_cells = atoi(argv[1]);
    const int global_ny_cells = atoi(argv[2]);
    const int global_nz_cells = atoi(argv[3]);
    const int do_restrict = (strcmp(argv[4], "restrict") == 0);

    if (!do_restrict && strcmp(argv[4], "inject") != 0)
    {
        fprintf(stderr, "Mode must be 'inject' or 'restrict'\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

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
                    /*neumann_face=*/NULL,
                    gs, global_x0, global_y0, global_z0,
                    dx, dy, dz, &gfs.domain);
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

    /* Fill parent, corrupt ghost zones, sync */
    fill_gf_test_function(&gfs);
    corrupt_gf(&gfs);
    sync_var_3d(&gfs, 0);

    /* Project parent -> child.
     * inject_var_3d fills all non-ghost child points (including physical
     * boundaries).  When restricting, restrict_var_3d then overwrites the
     * interior points with the full-weighting stencil. */
    inject_var_3d(&gfs, 0, child, 0);

    if (do_restrict)
    {
        /* Sentinel test (Finding 2): physical boundary child points must not
         * be modified by restrict_var_3d.  Overwrite them now and verify
         * they are unchanged after the call. */
        const double SENTINEL = -9999.0;
        const int64_t cnx = child->nx;
        const int64_t cny = child->ny;
        const int64_t cnz = child->nz;
        if (child->domain.lower_x_rank == INVALID_RANK)
            for (int64_t kc = 0; kc < cnz; kc++)
                for (int64_t jc = 0; jc < cny; jc++)
                    child->vars[0]->val[gf_indx_3d(child, 0, jc, kc)] = SENTINEL;
        if (child->domain.upper_x_rank == INVALID_RANK)
            for (int64_t kc = 0; kc < cnz; kc++)
                for (int64_t jc = 0; jc < cny; jc++)
                    child->vars[0]->val[gf_indx_3d(child, cnx - 1, jc, kc)] = SENTINEL;
        if (child->domain.lower_y_rank == INVALID_RANK)
            for (int64_t kc = 0; kc < cnz; kc++)
                for (int64_t ic = 0; ic < cnx; ic++)
                    child->vars[0]->val[gf_indx_3d(child, ic, 0, kc)] = SENTINEL;
        if (child->domain.upper_y_rank == INVALID_RANK)
            for (int64_t kc = 0; kc < cnz; kc++)
                for (int64_t ic = 0; ic < cnx; ic++)
                    child->vars[0]->val[gf_indx_3d(child, ic, cny - 1, kc)] = SENTINEL;
        if (child->domain.lower_z_rank == INVALID_RANK)
            for (int64_t jc = 0; jc < cny; jc++)
                for (int64_t ic = 0; ic < cnx; ic++)
                    child->vars[0]->val[gf_indx_3d(child, ic, jc, 0)] = SENTINEL;
        if (child->domain.upper_z_rank == INVALID_RANK)
            for (int64_t jc = 0; jc < cny; jc++)
                for (int64_t ic = 0; ic < cnx; ic++)
                    child->vars[0]->val[gf_indx_3d(child, ic, jc, cnz - 1)] = SENTINEL;

        restrict_var_3d(&gfs, 0, child, 0);

        /* Verify sentinels are unchanged */
        if (child->domain.lower_x_rank == INVALID_RANK)
            for (int64_t kc = 0; kc < cnz; kc++)
                for (int64_t jc = 0; jc < cny; jc++)
                    if (child->vars[0]->val[gf_indx_3d(child, 0, jc, kc)] != SENTINEL)
                    {
                        fprintf(stderr, "Rank %d: restrict touched lower-x boundary at (jc=%lld,kc=%lld)\n",
                                mpi_rank, (long long)jc, (long long)kc);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }
        if (child->domain.upper_x_rank == INVALID_RANK)
            for (int64_t kc = 0; kc < cnz; kc++)
                for (int64_t jc = 0; jc < cny; jc++)
                    if (child->vars[0]->val[gf_indx_3d(child, cnx - 1, jc, kc)] != SENTINEL)
                    {
                        fprintf(stderr, "Rank %d: restrict touched upper-x boundary at (jc=%lld,kc=%lld)\n",
                                mpi_rank, (long long)jc, (long long)kc);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }
        if (child->domain.lower_y_rank == INVALID_RANK)
            for (int64_t kc = 0; kc < cnz; kc++)
                for (int64_t ic = 0; ic < cnx; ic++)
                    if (child->vars[0]->val[gf_indx_3d(child, ic, 0, kc)] != SENTINEL)
                    {
                        fprintf(stderr, "Rank %d: restrict touched lower-y boundary at (ic=%lld,kc=%lld)\n",
                                mpi_rank, (long long)ic, (long long)kc);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }
        if (child->domain.upper_y_rank == INVALID_RANK)
            for (int64_t kc = 0; kc < cnz; kc++)
                for (int64_t ic = 0; ic < cnx; ic++)
                    if (child->vars[0]->val[gf_indx_3d(child, ic, cny - 1, kc)] != SENTINEL)
                    {
                        fprintf(stderr, "Rank %d: restrict touched upper-y boundary at (ic=%lld,kc=%lld)\n",
                                mpi_rank, (long long)ic, (long long)kc);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }
        if (child->domain.lower_z_rank == INVALID_RANK)
            for (int64_t jc = 0; jc < cny; jc++)
                for (int64_t ic = 0; ic < cnx; ic++)
                    if (child->vars[0]->val[gf_indx_3d(child, ic, jc, 0)] != SENTINEL)
                    {
                        fprintf(stderr, "Rank %d: restrict touched lower-z boundary at (ic=%lld,jc=%lld)\n",
                                mpi_rank, (long long)ic, (long long)jc);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }
        if (child->domain.upper_z_rank == INVALID_RANK)
            for (int64_t jc = 0; jc < cny; jc++)
                for (int64_t ic = 0; ic < cnx; ic++)
                    if (child->vars[0]->val[gf_indx_3d(child, ic, jc, cnz - 1)] != SENTINEL)
                    {
                        fprintf(stderr, "Rank %d: restrict touched upper-z boundary at (ic=%lld,jc=%lld)\n",
                                mpi_rank, (long long)ic, (long long)jc);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }

        /* Restore boundary cells from analytic f = 1.5x - 2y + z */
        if (child->domain.lower_x_rank == INVALID_RANK)
            for (int64_t kc = 0; kc < cnz; kc++)
                for (int64_t jc = 0; jc < cny; jc++)
                {
                    const double x = child->x0 + 0 * child->dx;
                    const double y = child->y0 + jc * child->dy;
                    const double z = child->z0 + kc * child->dz;
                    child->vars[0]->val[gf_indx_3d(child, 0, jc, kc)] = 1.5*x - 2.0*y + z;
                }
        if (child->domain.upper_x_rank == INVALID_RANK)
            for (int64_t kc = 0; kc < cnz; kc++)
                for (int64_t jc = 0; jc < cny; jc++)
                {
                    const double x = child->x0 + (cnx - 1) * child->dx;
                    const double y = child->y0 + jc * child->dy;
                    const double z = child->z0 + kc * child->dz;
                    child->vars[0]->val[gf_indx_3d(child, cnx - 1, jc, kc)] = 1.5*x - 2.0*y + z;
                }
        if (child->domain.lower_y_rank == INVALID_RANK)
            for (int64_t kc = 0; kc < cnz; kc++)
                for (int64_t ic = 0; ic < cnx; ic++)
                {
                    const double x = child->x0 + ic * child->dx;
                    const double y = child->y0 + 0 * child->dy;
                    const double z = child->z0 + kc * child->dz;
                    child->vars[0]->val[gf_indx_3d(child, ic, 0, kc)] = 1.5*x - 2.0*y + z;
                }
        if (child->domain.upper_y_rank == INVALID_RANK)
            for (int64_t kc = 0; kc < cnz; kc++)
                for (int64_t ic = 0; ic < cnx; ic++)
                {
                    const double x = child->x0 + ic * child->dx;
                    const double y = child->y0 + (cny - 1) * child->dy;
                    const double z = child->z0 + kc * child->dz;
                    child->vars[0]->val[gf_indx_3d(child, ic, cny - 1, kc)] = 1.5*x - 2.0*y + z;
                }
        if (child->domain.lower_z_rank == INVALID_RANK)
            for (int64_t jc = 0; jc < cny; jc++)
                for (int64_t ic = 0; ic < cnx; ic++)
                {
                    const double x = child->x0 + ic * child->dx;
                    const double y = child->y0 + jc * child->dy;
                    const double z = child->z0 + 0 * child->dz;
                    child->vars[0]->val[gf_indx_3d(child, ic, jc, 0)] = 1.5*x - 2.0*y + z;
                }
        if (child->domain.upper_z_rank == INVALID_RANK)
            for (int64_t jc = 0; jc < cny; jc++)
                for (int64_t ic = 0; ic < cnx; ic++)
                {
                    const double x = child->x0 + ic * child->dx;
                    const double y = child->y0 + jc * child->dy;
                    const double z = child->z0 + (cnz - 1) * child->dz;
                    child->vars[0]->val[gf_indx_3d(child, ic, jc, cnz - 1)] = 1.5*x - 2.0*y + z;
                }
    }

    /* Corrupt child ghost zones, sync child */
    corrupt_gf(child);
    sync_var_3d(child, 0);

    /* Output child data for verification */
    output_3d_gf(child, 0, NULL);

    ngfs_3d_deallocate(&gfs);
    cleanup_3d_domain(&gfs.domain);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

static void fill_gf_test_function(struct ngfs_3d *gfs)
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
                    1.5 * x - 2.0 * y + z;
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

    IndexBox x_lo = { .is = 0,      .ie = gs,      .js = 0, .je = ny, .ks = 0, .ke = nz };
    IndexBox x_hi = { .is = nx - gs, .ie = nx,      .js = 0, .je = ny, .ks = 0, .ke = nz };
    IndexBox y_lo = { .is = 0,      .ie = nx,      .js = 0, .je = gs, .ks = 0, .ke = nz };
    IndexBox y_hi = { .is = 0,      .ie = nx,      .js = ny - gs, .je = ny, .ks = 0, .ke = nz };
    IndexBox z_lo = { .is = 0,      .ie = nx,      .js = 0, .je = ny, .ks = 0,      .ke = gs };
    IndexBox z_hi = { .is = 0,      .ie = nx,      .js = 0, .je = ny, .ks = nz - gs, .ke = nz };

    if (gfs->domain.lower_x_rank != INVALID_RANK) corrupt_box(gfs, x_lo, 42);
    if (gfs->domain.upper_x_rank != INVALID_RANK) corrupt_box(gfs, x_hi, 45);
    if (gfs->domain.lower_y_rank != INVALID_RANK) corrupt_box(gfs, y_lo, 43);
    if (gfs->domain.upper_y_rank != INVALID_RANK) corrupt_box(gfs, y_hi, 46);
    if (gfs->domain.lower_z_rank != INVALID_RANK) corrupt_box(gfs, z_lo, 44);
    if (gfs->domain.upper_z_rank != INVALID_RANK) corrupt_box(gfs, z_hi, 47);
}

