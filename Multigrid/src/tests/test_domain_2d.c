#include "comm.h"
#include "domain.h"
#include "gf.h"
#include "io.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void fill_gf_test_function(struct ngfs_2d *gfs);
static void fill_gf_test_function2(struct ngfs_2d *gfs);
static void corrupt_gf(struct ngfs_2d *gfs);
static void corrupt_gf_var(struct ngfs_2d *gfs, int var);
static void verify_var1(struct ngfs_2d *gfs);

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

    if (mpi_rank == 0)
    {
        for (int i = 0; i < 2; i++)
        {
            printf("Dimension(%d) total points %lu, processes %lu, (app.) points per process %lu\n", i, dims[i], topology[i],
                   dims[i] / topology[i]);
        }
    }

    const int px = topology[0];
    const int py = topology[1];

    if (global_nx_cells <= 0 || global_ny_cells <= 0)
    {
        // TODO: FIX
        fprintf(stderr, "NX, NY  > 0 required (%d, %d)\n", global_nx_cells,
                global_ny_cells);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (px * py != mpi_size)
    {
        // TODO: FIX
        fprintf(stderr, "PX * PY != MPI_SIZE (%d, %d, %d)\n", px, py, mpi_size);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const double dx = 1.0 / global_nx_cells;
    const double dy = 1.0 / global_ny_cells;
    const double global_x0 = 0.0;
    const double global_y0 = 0.0;

    const int gs = 2;

    struct ngfs_2d gfs;
    gfs.vars = NULL;

    const int nvars = 2;

    setup_2d_domain(px, py, mpi_rank, global_nx_cells, global_ny_cells, gs, global_x0,
                    global_y0, dx, dy, &gfs.domain);

    ngfs_2d_allocate(nvars, &gfs);

    fill_gf_test_function(&gfs);
    corrupt_gf(&gfs);
    sync_var_2d(&gfs, 0);
    output_2d_gf(&gfs, 0, NULL);

    /* Also exercise vars[1] to catch variable-offset bugs in sync_var_2d */
    fill_gf_test_function2(&gfs);
    corrupt_gf_var(&gfs, 1);
    sync_var_2d(&gfs, 1);
    verify_var1(&gfs);

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
            const int64_t ijk = gf_indx_2d(gfs, i, j);
            const double x = gfs->x0 + i * gfs->dx;
            gfs->vars[0]->val[ijk] = 1.5 * x - 2 * y;
        }
    }
}

/* Second test function: f2 = 3x + y  (distinct from vars[0]) */
static void fill_gf_test_function2(struct ngfs_2d *gfs)
{
    for (int64_t j = 0; j < gfs->ny; j++)
    {
        const double y = gfs->y0 + j * gfs->dy;
        for (int64_t i = 0; i < gfs->nx; i++)
        {
            const int64_t ijk = gf_indx_2d(gfs, i, j);
            const double x = gfs->x0 + i * gfs->dx;
            gfs->vars[1]->val[ijk] = 3.0 * x + y;
        }
    }
}
static void corrupt_box(struct ngfs_2d *gfs, IndexBox box, double val)
{
    for (int64_t j = box.js; j < box.je; j++)
    {
        for (int64_t i = box.is; i < box.ie; i++)
        {
            const int64_t ijk = gf_indx_2d(gfs, i, j);
            gfs->vars[0]->val[ijk] = val;
        }
    }
}

static void corrupt_gf(struct ngfs_2d *gfs)
{
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int gs = gfs->gs;

    IndexBox x_lo = { .is = 0, .ie = gs, .js = 0, .je = ny, .ks = 0, .ke = 0 };
    IndexBox x_hi = {
        .is = nx - gs, .ie = nx, .js = 0, .je = ny, .ks = 0, .ke = 0
    };

    IndexBox y_lo = { .is = 0, .ie = nx, .js = 0, .je = gs, .ks = 0, .ke = 0 };
    IndexBox y_hi = {
        .is = 0, .ie = nx, .js = ny - gs, .je = ny, .ks = 0, .ke = 0
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

    if (gfs->domain.upper_x_rank > -1)
    {
        corrupt_box(gfs, x_hi, 45);
    }

    if (gfs->domain.upper_y_rank > -1)
    {
        corrupt_box(gfs, y_hi, 46);
    }
}

/* corrupt_gf_var: same sentinel pattern as corrupt_gf but for an arbitrary var */
static void corrupt_gf_var(struct ngfs_2d *gfs, int var)
{
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int gs = gfs->gs;

    if (gfs->domain.lower_x_rank != INVALID_RANK)
        for (int64_t j = 0; j < ny; j++)
            for (int i = 0; i < gs; i++)
                gfs->vars[var]->val[gf_indx_2d(gfs, i, j)] = 42;

    if (gfs->domain.upper_x_rank != INVALID_RANK)
        for (int64_t j = 0; j < ny; j++)
            for (int64_t i = nx - gs; i < nx; i++)
                gfs->vars[var]->val[gf_indx_2d(gfs, i, j)] = 45;

    if (gfs->domain.lower_y_rank != INVALID_RANK)
        for (int j = 0; j < gs; j++)
            for (int64_t i = 0; i < nx; i++)
                gfs->vars[var]->val[gf_indx_2d(gfs, i, j)] = 43;

    if (gfs->domain.upper_y_rank != INVALID_RANK)
        for (int64_t j = ny - gs; j < ny; j++)
            for (int64_t i = 0; i < nx; i++)
                gfs->vars[var]->val[gf_indx_2d(gfs, i, j)] = 46;
}

/* verify_var1: inline check that vars[1] matches f2 = 3x + y after sync */
static void verify_var1(struct ngfs_2d *gfs)
{
    const double tol = 1e-14;
    for (int64_t j = 0; j < gfs->ny; j++)
    {
        const double y = gfs->y0 + j * gfs->dy;
        for (int64_t i = 0; i < gfs->nx; i++)
        {
            const double x = gfs->x0 + i * gfs->dx;
            const double expected = 3.0 * x + y;
            const double got = gfs->vars[1]->val[gf_indx_2d(gfs, i, j)];
            const double err = got - expected;
            if (err > tol || err < -tol)
            {
                fprintf(stderr,
                        "Rank %d: vars[1] mismatch at (%lld,%lld): got %.16e expected %.16e\n",
                        gfs->domain.rank, (long long)i, (long long)j, got, expected);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
    }
}

