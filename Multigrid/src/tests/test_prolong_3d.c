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

    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s NX NY NZ\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int global_nx = atoi(argv[1]);
    const int global_ny = atoi(argv[2]);
    const int global_nz = atoi(argv[3]);

    size_t dims[3];
    dims[0] = global_nx;
    dims[1] = global_ny;
    dims[2] = global_nz;
    size_t topology[3];
    automatic_topology(3, dims, mpi_size, topology);

    const int px = topology[0];
    const int py = topology[1];
    const int pz = topology[2];

    const double dx = 1.0 / (global_nx - 1);
    const double dy = 1.0 / (global_ny - 1);
    const double dz = 1.0 / (global_nz - 1);
    const double global_x0 = 0.0;
    const double global_y0 = 0.0;
    const double global_z0 = 0.0;
    const int gs = 2;
    const int nvars = 1;
    const int min_cells = 4;

    struct ngfs_3d gfs;
    gfs.vars = NULL;

    setup_3d_domain(px, py, pz, mpi_rank, global_nx, global_ny, global_nz,
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

    /* Fill parent with f, sync */
    fill_gf_test_function(&gfs);
    corrupt_gf(&gfs);
    sync_var_3d(&gfs, 0);

    /* Restrict parent -> child */
    inject_var_3d(&gfs, 0, child, 0);
    restrict_var_3d(&gfs, 0, child, 0);

    /* Sync child */
    corrupt_gf(child);
    sync_var_3d(child, 0);

    /* Sentinel test (Finding 2): prolong must not touch parent physical
     * boundary points (those at global index 0 or N-1 per axis).
     * Overwrite them with a sentinel and verify they are unchanged after. */
    {
        const double SENTINEL = -9999.0;
        const int64_t pnx = gfs.nx;
        const int64_t pny = gfs.ny;
        const int64_t pnz = gfs.nz;
        if (gfs.domain.lower_x_rank == INVALID_RANK)
            for (int64_t kp = 0; kp < pnz; kp++)
                for (int64_t jp = 0; jp < pny; jp++)
                    gfs.vars[0]->val[gf_indx_3d(&gfs, 0, jp, kp)] = SENTINEL;
        if (gfs.domain.upper_x_rank == INVALID_RANK)
            for (int64_t kp = 0; kp < pnz; kp++)
                for (int64_t jp = 0; jp < pny; jp++)
                    gfs.vars[0]->val[gf_indx_3d(&gfs, pnx - 1, jp, kp)] = SENTINEL;
        if (gfs.domain.lower_y_rank == INVALID_RANK)
            for (int64_t kp = 0; kp < pnz; kp++)
                for (int64_t ip = 0; ip < pnx; ip++)
                    gfs.vars[0]->val[gf_indx_3d(&gfs, ip, 0, kp)] = SENTINEL;
        if (gfs.domain.upper_y_rank == INVALID_RANK)
            for (int64_t kp = 0; kp < pnz; kp++)
                for (int64_t ip = 0; ip < pnx; ip++)
                    gfs.vars[0]->val[gf_indx_3d(&gfs, ip, pny - 1, kp)] = SENTINEL;
        if (gfs.domain.lower_z_rank == INVALID_RANK)
            for (int64_t jp = 0; jp < pny; jp++)
                for (int64_t ip = 0; ip < pnx; ip++)
                    gfs.vars[0]->val[gf_indx_3d(&gfs, ip, jp, 0)] = SENTINEL;
        if (gfs.domain.upper_z_rank == INVALID_RANK)
            for (int64_t jp = 0; jp < pny; jp++)
                for (int64_t ip = 0; ip < pnx; ip++)
                    gfs.vars[0]->val[gf_indx_3d(&gfs, ip, jp, pnz - 1)] = SENTINEL;

        /* Prolong child -> parent */
        prolong_var_3d(child, 0, &gfs, 0);

        /* Verify sentinels are unchanged */
        if (gfs.domain.lower_x_rank == INVALID_RANK)
            for (int64_t kp = 0; kp < pnz; kp++)
                for (int64_t jp = 0; jp < pny; jp++)
                    if (gfs.vars[0]->val[gf_indx_3d(&gfs, 0, jp, kp)] != SENTINEL)
                    {
                        fprintf(stderr, "Rank %d: prolong touched lower-x boundary at (jp=%lld,kp=%lld)\n",
                                mpi_rank, (long long)jp, (long long)kp);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }
        if (gfs.domain.upper_x_rank == INVALID_RANK)
            for (int64_t kp = 0; kp < pnz; kp++)
                for (int64_t jp = 0; jp < pny; jp++)
                    if (gfs.vars[0]->val[gf_indx_3d(&gfs, pnx - 1, jp, kp)] != SENTINEL)
                    {
                        fprintf(stderr, "Rank %d: prolong touched upper-x boundary at (jp=%lld,kp=%lld)\n",
                                mpi_rank, (long long)jp, (long long)kp);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }
        if (gfs.domain.lower_y_rank == INVALID_RANK)
            for (int64_t kp = 0; kp < pnz; kp++)
                for (int64_t ip = 0; ip < pnx; ip++)
                    if (gfs.vars[0]->val[gf_indx_3d(&gfs, ip, 0, kp)] != SENTINEL)
                    {
                        fprintf(stderr, "Rank %d: prolong touched lower-y boundary at (ip=%lld,kp=%lld)\n",
                                mpi_rank, (long long)ip, (long long)kp);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }
        if (gfs.domain.upper_y_rank == INVALID_RANK)
            for (int64_t kp = 0; kp < pnz; kp++)
                for (int64_t ip = 0; ip < pnx; ip++)
                    if (gfs.vars[0]->val[gf_indx_3d(&gfs, ip, pny - 1, kp)] != SENTINEL)
                    {
                        fprintf(stderr, "Rank %d: prolong touched upper-y boundary at (ip=%lld,kp=%lld)\n",
                                mpi_rank, (long long)ip, (long long)kp);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }
        if (gfs.domain.lower_z_rank == INVALID_RANK)
            for (int64_t jp = 0; jp < pny; jp++)
                for (int64_t ip = 0; ip < pnx; ip++)
                    if (gfs.vars[0]->val[gf_indx_3d(&gfs, ip, jp, 0)] != SENTINEL)
                    {
                        fprintf(stderr, "Rank %d: prolong touched lower-z boundary at (ip=%lld,jp=%lld)\n",
                                mpi_rank, (long long)ip, (long long)jp);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }
        if (gfs.domain.upper_z_rank == INVALID_RANK)
            for (int64_t jp = 0; jp < pny; jp++)
                for (int64_t ip = 0; ip < pnx; ip++)
                    if (gfs.vars[0]->val[gf_indx_3d(&gfs, ip, jp, pnz - 1)] != SENTINEL)
                    {
                        fprintf(stderr, "Rank %d: prolong touched upper-z boundary at (ip=%lld,jp=%lld)\n",
                                mpi_rank, (long long)ip, (long long)jp);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }

        /* Restore boundary to original f = 1.5x - 2y + z */
        if (gfs.domain.lower_x_rank == INVALID_RANK)
            for (int64_t kp = 0; kp < pnz; kp++)
                for (int64_t jp = 0; jp < pny; jp++)
                {
                    const double x = gfs.x0, y = gfs.y0 + jp * gfs.dy, z = gfs.z0 + kp * gfs.dz;
                    gfs.vars[0]->val[gf_indx_3d(&gfs, 0, jp, kp)] = 1.5*x - 2.0*y + z;
                }
        if (gfs.domain.upper_x_rank == INVALID_RANK)
            for (int64_t kp = 0; kp < pnz; kp++)
                for (int64_t jp = 0; jp < pny; jp++)
                {
                    const double x = gfs.x0 + (pnx-1)*gfs.dx, y = gfs.y0 + jp*gfs.dy, z = gfs.z0 + kp*gfs.dz;
                    gfs.vars[0]->val[gf_indx_3d(&gfs, pnx - 1, jp, kp)] = 1.5*x - 2.0*y + z;
                }
        if (gfs.domain.lower_y_rank == INVALID_RANK)
            for (int64_t kp = 0; kp < pnz; kp++)
                for (int64_t ip = 0; ip < pnx; ip++)
                {
                    const double x = gfs.x0 + ip*gfs.dx, y = gfs.y0, z = gfs.z0 + kp*gfs.dz;
                    gfs.vars[0]->val[gf_indx_3d(&gfs, ip, 0, kp)] = 1.5*x - 2.0*y + z;
                }
        if (gfs.domain.upper_y_rank == INVALID_RANK)
            for (int64_t kp = 0; kp < pnz; kp++)
                for (int64_t ip = 0; ip < pnx; ip++)
                {
                    const double x = gfs.x0 + ip*gfs.dx, y = gfs.y0 + (pny-1)*gfs.dy, z = gfs.z0 + kp*gfs.dz;
                    gfs.vars[0]->val[gf_indx_3d(&gfs, ip, pny - 1, kp)] = 1.5*x - 2.0*y + z;
                }
        if (gfs.domain.lower_z_rank == INVALID_RANK)
            for (int64_t jp = 0; jp < pny; jp++)
                for (int64_t ip = 0; ip < pnx; ip++)
                {
                    const double x = gfs.x0 + ip*gfs.dx, y = gfs.y0 + jp*gfs.dy, z = gfs.z0;
                    gfs.vars[0]->val[gf_indx_3d(&gfs, ip, jp, 0)] = 1.5*x - 2.0*y + z;
                }
        if (gfs.domain.upper_z_rank == INVALID_RANK)
            for (int64_t jp = 0; jp < pny; jp++)
                for (int64_t ip = 0; ip < pnx; ip++)
                {
                    const double x = gfs.x0 + ip*gfs.dx, y = gfs.y0 + jp*gfs.dy, z = gfs.z0 + (pnz-1)*gfs.dz;
                    gfs.vars[0]->val[gf_indx_3d(&gfs, ip, jp, pnz - 1)] = 1.5*x - 2.0*y + z;
                }
    }

    /* Sync parent */
    corrupt_gf(&gfs);
    sync_var_3d(&gfs, 0);

    /* Output parent: interior should be 0; all physical boundary points (lo and
     * hi on every axis) remain at their original f values since prolong never
     * touches them. */
    output_3d_gf(&gfs, 0, NULL);

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

