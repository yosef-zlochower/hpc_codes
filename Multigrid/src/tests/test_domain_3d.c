#include "comm.h"
#include "domain.h"
#include "gf.h"
#include "io.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void fill_gf_test_function(struct ngfs_3d *gfs);
static void fill_gf_test_function2(struct ngfs_3d *gfs);
static void corrupt_gf(struct ngfs_3d *gfs);
static void corrupt_gf_var(struct ngfs_3d *gfs, int var);
static void verify_var1(struct ngfs_3d *gfs);

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

    if (mpi_rank == 0)
    {
        for (int i = 0; i < 3; i++)
        {
            printf("Dimension(%d) total points %lu, processes %lu, (app.) points per process %lu\n", i, dims[i], topology[i],
                   dims[i] / topology[i]);
        }
    }

    const int px = topology[0];
    const int py = topology[1];
    const int pz = topology[2];

    if (global_nx_cells <= 0 || global_ny_cells <= 0 || global_nz_cells <= 0)
    {
        // TODO: FIX
        fprintf(stderr, "NX, NY, NZ all > 0 required (%d, %d, %d)\n", global_nx_cells,
                global_ny_cells, global_nz_cells);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (px * py * pz != mpi_size)
    {
        // TODO: FIX
        fprintf(stderr, "PX * PY != MPI_SIZE (%d, %d, %d)\n", px, py, mpi_size);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const double dx = 1.0 / global_nx_cells;
    const double dy = 1.0 / global_ny_cells;
    const double dz = 1.0 / global_nz_cells;
    const double global_x0 = 0.0;
    const double global_y0 = 0.0;
    const double global_z0 = 0.0;

    const int gs = 2;

    struct ngfs_3d gfs;
    gfs.vars = NULL;

    const int nvars = 2;

    setup_3d_domain(px, py, pz, mpi_rank, global_nx_cells, global_ny_cells, global_nz_cells,
                    /*neumann_face=*/NULL,
                    gs,
                    global_x0, global_y0, global_z0, dx, dy, dz, &gfs.domain);

    ngfs_3d_allocate(nvars, &gfs);

    fill_gf_test_function(&gfs);
    corrupt_gf(&gfs);
    sync_var_3d(&gfs, 0);
    output_3d_gf(&gfs, 0, NULL);

    /* Also exercise vars[1] to catch variable-offset bugs in sync_var_3d */
    fill_gf_test_function2(&gfs);
    corrupt_gf_var(&gfs, 1);
    sync_var_3d(&gfs, 1);
    verify_var1(&gfs);

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
                const int64_t ijk = gf_indx_3d(gfs, i, j, k);
                const double x = gfs->x0 + i * gfs->dx;
                gfs->vars[0]->val[ijk] = 1.5 * x - 2 * y + z;
            }
        }
    }
}

/* Second test function: f2 = 3x + y + 2z  (distinct from vars[0]) */
static void fill_gf_test_function2(struct ngfs_3d *gfs)
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
                gfs->vars[1]->val[gf_indx_3d(gfs, i, j, k)] = 3.0 * x + y + 2.0 * z;
            }
        }
    }
}

/* corrupt_gf_var: same sentinel pattern as corrupt_gf but for an arbitrary var */
static void corrupt_gf_var(struct ngfs_3d *gfs, int var)
{
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t nz = gfs->nz;
    const int gs = gfs->gs;

    if (gfs->domain.lower_x_rank != INVALID_RANK)
        for (int64_t k = 0; k < nz; k++)
            for (int64_t j = 0; j < ny; j++)
                for (int i = 0; i < gs; i++)
                    gfs->vars[var]->val[gf_indx_3d(gfs, i, j, k)] = 42;

    if (gfs->domain.upper_x_rank != INVALID_RANK)
        for (int64_t k = 0; k < nz; k++)
            for (int64_t j = 0; j < ny; j++)
                for (int64_t i = nx - gs; i < nx; i++)
                    gfs->vars[var]->val[gf_indx_3d(gfs, i, j, k)] = 45;

    if (gfs->domain.lower_y_rank != INVALID_RANK)
        for (int64_t k = 0; k < nz; k++)
            for (int j = 0; j < gs; j++)
                for (int64_t i = 0; i < nx; i++)
                    gfs->vars[var]->val[gf_indx_3d(gfs, i, j, k)] = 43;

    if (gfs->domain.upper_y_rank != INVALID_RANK)
        for (int64_t k = 0; k < nz; k++)
            for (int64_t j = ny - gs; j < ny; j++)
                for (int64_t i = 0; i < nx; i++)
                    gfs->vars[var]->val[gf_indx_3d(gfs, i, j, k)] = 46;

    if (gfs->domain.lower_z_rank != INVALID_RANK)
        for (int k = 0; k < gs; k++)
            for (int64_t j = 0; j < ny; j++)
                for (int64_t i = 0; i < nx; i++)
                    gfs->vars[var]->val[gf_indx_3d(gfs, i, j, k)] = 44;

    if (gfs->domain.upper_z_rank != INVALID_RANK)
        for (int64_t k = nz - gs; k < nz; k++)
            for (int64_t j = 0; j < ny; j++)
                for (int64_t i = 0; i < nx; i++)
                    gfs->vars[var]->val[gf_indx_3d(gfs, i, j, k)] = 47;
}

/* verify_var1: inline check that vars[1] matches f2 = 3x + y + 2z after sync */
static void verify_var1(struct ngfs_3d *gfs)
{
    const double tol = 1e-14;
    for (int64_t k = 0; k < gfs->nz; k++)
    {
        const double z = gfs->z0 + k * gfs->dz;
        for (int64_t j = 0; j < gfs->ny; j++)
        {
            const double y = gfs->y0 + j * gfs->dy;
            for (int64_t i = 0; i < gfs->nx; i++)
            {
                const double x = gfs->x0 + i * gfs->dx;
                const double expected = 3.0 * x + y + 2.0 * z;
                const double got = gfs->vars[1]->val[gf_indx_3d(gfs, i, j, k)];
                const double err = got - expected;
                if (err > tol || err < -tol)
                {
                    fprintf(stderr,
                            "Rank %d: vars[1] mismatch at (%lld,%lld,%lld): got %.16e expected %.16e\n",
                            gfs->domain.rank, (long long)i, (long long)j, (long long)k, got, expected);
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
            }
        }
    }
}
static void corrupt_box(struct ngfs_3d *gfs, IndexBox box, double val)
{
    for (int64_t k = box.ks; k < box.ke; k++)
    {
        for (int64_t j = box.js; j < box.je; j++)
        {
            for (int64_t i = box.is; i < box.ie; i++)
            {
                const int64_t ijk = gf_indx_3d(gfs, i, j, k);
                gfs->vars[0]->val[ijk] = val;
            }
        }
    }
}

static void corrupt_gf(struct ngfs_3d *gfs)
{
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t nz = gfs->nz;
    const int gs = gfs->gs;

    IndexBox x_lo = { .is = 0, .ie = gs, .js = 0, .je = ny, .ks = 0, .ke = nz };
    IndexBox x_hi = {
        .is = nx - gs, .ie = nx, .js = 0, .je = ny, .ks = 0, .ke = nz
    };

    IndexBox y_lo = { .is = 0, .ie = nx, .js = 0, .je = gs, .ks = 0, .ke = nz };
    IndexBox y_hi = {
        .is = 0, .ie = nx, .js = ny - gs, .je = ny, .ks = 0, .ke = nz
    };

    IndexBox z_lo = { .is = 0, .ie = nx, .js = 0, .je = ny, .ks = 0, .ke = gs };
    IndexBox z_hi = {
        .is = 0, .ie = nx, .js = 0, .je = ny, .ks = nz - gs, .ke = nz
    };

    // TODO FIX
    if (gfs->domain.lower_x_rank > -1)
    {
        corrupt_box(gfs, x_lo, 42);
    }

    if (gfs->domain.lower_y_rank > -1)
    {
        corrupt_box(gfs, y_lo, 43);
    }

    if (gfs->domain.lower_z_rank > -1)
    {
        corrupt_box(gfs, z_lo, 44);
    }

    if (gfs->domain.upper_x_rank > -1)
    {
        corrupt_box(gfs, x_hi, 45);
    }

    if (gfs->domain.upper_y_rank > -1)
    {
        corrupt_box(gfs, y_hi, 46);
    }

    if (gfs->domain.upper_z_rank > -1)
    {
        corrupt_box(gfs, z_hi, 47);
    }
}

