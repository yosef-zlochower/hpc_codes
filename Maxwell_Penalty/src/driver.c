#include "maxwell_eqs.h"
#include "gf.h"
#include "rk4.h"
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> // needed for timing
#include <unistd.h>

#include "io.h"
#include "comm.h"
#include "domain.h"
#include "parameter.h"
#include "analytic_solutions.h"
#include "derivatives.h"
#include "timer.h"

/* Global parameter structs, populated from TOML file in main() */
struct maxwell_param_st maxwell_params;

struct analytic_params_st analytic_params;

int main(int argc, char **argv)
{
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Init failed\n");
        return 1;
    }

    int size;
    int rank;
    MPI_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &size));
    MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    /* ── Parse parameter file ─────────────────────────────────────── */
    if (argc != 2)
    {
        if (rank == 0)
        {
            fprintf(stderr,
                    "Usage: %s <parameter_file.toml>\n",
                    argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    const char *param_file = argv[1];
    if (parse_maxwell_parameters(&maxwell_params, param_file) != 0)
    {
        if (rank == 0)
            fprintf(stderr, "Error reading parameter file '%s'\n", param_file);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* ── MPI topology (automatic) ─────────────────────────────────── */
    const size_t grid_dims[3] = { maxwell_params.nx, maxwell_params.ny,
                                  maxwell_params.nz };
    size_t topo[3];
    if (automatic_topology(3, grid_dims, (size_t)size, topo) != 0)
    {
        if (rank == 0)
            fprintf(stderr, "automatic_topology() failed\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    const int cpux = (int)topo[0];
    const int cpuy = (int)topo[1];
    const int cpuz = (int)topo[2];

    if (rank == 0)
    {
        fprintf(stderr, "MPI topology: %d x %d x %d  (%d processes)\n",
                cpux, cpuy, cpuz, size);
    }

    const int gs = maxwell_params.ghost_size;
    const int global_nx = maxwell_params.nx;
    const int global_ny = maxwell_params.ny;
    const int global_nz = maxwell_params.nz;

    if (global_nx < 2 * gs || global_ny < 2 * gs || global_nz < 2 * gs)
    {
        if (rank == 0)
            fprintf(stderr, "Grid dimensions must be >= 2 * ghost_size (%d)\n",
                    2 * gs);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Populate analytic solution parameters from TOML. The struct-of-structs
     * layout mirrors the [source.<name>] TOML sub-tables, so each source
     * block copies in a single assignment. */
    analytic_params.t0                = 0.0;
    analytic_params.plane_wave        = maxwell_params.plane_wave;
    analytic_params.gaussian_beam     = maxwell_params.gaussian_beam;
    analytic_params.te_waveguide_mode = maxwell_params.te_waveguide_mode;

    /* Variable counts come from the slot enums in maxwell_eqs.h, so they
     * cannot drift from the DECLARE_*_VARS macros or the name tables. */
    const int n_evol_vars = N_EVOL;
    const int n_aux_vars  = N_AUX;

    int timer = register_timer("/main");

    {
        char name_buffer[1024];
        const int bufflen = sizeof(name_buffer);
        gethostname(name_buffer, bufflen);
        fprintf(stderr, "rank %d of %d running on %s\n", rank, size,
                name_buffer);
    }

    struct ngfs gfs = {0};

    setup_3d_domain(cpux, cpuy, cpuz, rank, global_nx, global_ny, global_nz, gs,
                    maxwell_params.x0, maxwell_params.y0, maxwell_params.z0,
                    maxwell_params.xn, maxwell_params.yn, maxwell_params.zn,
                    &(gfs.domain),
                    maxwell_params.periodic_x ? PERIODIC : NON_PERIODIC,
                    maxwell_params.periodic_y ? PERIODIC : NON_PERIODIC,
                    maxwell_params.periodic_z ? PERIODIC : NON_PERIODIC);

    /* Small-grid check: stencil_at (maxwell_eqs.c) picks the LEFT SBP
     * closure when the left and right SBP42_CLOSURE_ROWS-wide windows
     * overlap.  That happens on any axis where a single rank carries BOTH
     * physical boundaries and the local point count is less than
     * 2 * SBP42_CLOSURE_ROWS.  Fail fast with a clear message rather than
     * silently running a wrong scheme.  Multi-rank axes are fine because
     * no single rank then sees both physical boundaries. */
    {
        const int64_t min_n = 2 * SBP42_CLOSURE_ROWS;
        const int small_x = gfs.domain.bbox.x.lower && gfs.domain.bbox.x.upper
                            && gfs.domain.nx < min_n;
        const int small_y = gfs.domain.bbox.y.lower && gfs.domain.bbox.y.upper
                            && gfs.domain.ny < min_n;
        const int small_z = gfs.domain.bbox.z.lower && gfs.domain.bbox.z.upper
                            && gfs.domain.nz < min_n;
        if (small_x || small_y || small_z)
        {
            fprintf(stderr,
                    "rank %d: local grid too small for SBP-4-2 boundary "
                    "closures on a doubly-physical axis.  Need at least "
                    "2 * SBP42_CLOSURE_ROWS = %" PRId64 " local points; "
                    "have nx=%" PRId64 " ny=%" PRId64 " nz=%" PRId64 " "
                    "(bbox x={%d,%d} y={%d,%d} z={%d,%d}).  "
                    "Increase the grid size or split the axis across more "
                    "MPI ranks so no one rank owns both ends.\n",
                    rank, min_n,
                    gfs.domain.nx, gfs.domain.ny, gfs.domain.nz,
                    gfs.domain.bbox.x.lower, gfs.domain.bbox.x.upper,
                    gfs.domain.bbox.y.lower, gfs.domain.bbox.y.upper,
                    gfs.domain.bbox.z.lower, gfs.domain.bbox.z.upper);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    ngfs_allocate(n_evol_vars, n_aux_vars, &gfs);

    /* Tag each grid function with the HDF5 dataset name stored in the
     * slot-indexed name tables (maxwell_eqs.c).  Iterating by slot means
     * the driver can't silently misalign names vs. storage — the tables
     * and the DECLARE_*_VARS macros are both driven by the same enum. */
    for (int i = 0; i < N_EVOL; i++)
        gf_rename(gfs.vars[i],    evolved_field_names[i]);
    for (int i = 0; i < N_AUX; i++)
        gf_rename(gfs.auxvars[i], aux_field_names[i]);

    const double min_dspace = fmin(gfs.dx, fmin(gfs.dy, gfs.dz));
    const double dt = maxwell_params.cfl_factor * min_dspace;
    const int output_every = maxwell_params.output_every;
    const int checkpoint_every = maxwell_params.checkpoint_every;
    const int out2d_z = maxwell_params.output_2d_z_plane; /* < 0 = disabled */

    double t = 0;
    int it_start = 1;

    if (maxwell_params.recover)
    {
        /* ── Recovery path ────────────────────────────────────────── */
        read_checkpoint(&gfs, &t, &it_start);
        it_start += 1; /* resume from the next iteration */

        /* Set the output counters so HDF5 group / file numbering
         * continues correctly across the recovery boundary. */
        set_output_counter_3D(it_start / output_every);
        set_output_counter_2D_xy_h5(it_start / output_every);
        set_output_counter_2D_xy(it_start / output_every);

        if (!rank)
            fprintf(stderr, "Resuming from iteration %d, t = %g\n",
                    it_start, t);
    }
    else
    {
        /* ── Fresh start ──────────────────────────────────────────── */
        set_initial_data(&gfs, t);
        /* The RHS differentiates the material fields (dx_ieps, dx_imu,
         * dx_sigma), so their ghost zones must agree across ranks
         * before the first step. */
        sync_vars(&gfs, AUX);

        if (!rank)
        {
            fprintf(stderr, "Testing Sync ... ");
        }
        for (int v = 0; v < gfs.n_evol_vars; v++)
        {
            gfs.vars[v]->dot = gfs.vars[v]->new;
        }
        sync_vars(&gfs, EVOLVED);
        const double terr = l2_error_analytic(&gfs, t);
        /* The sync test just checks that sync_vars agrees with the
         * analytic source at the ghost zones. Because the initial data
         * is drawn from the same source() call, any mismatch is
         * communication error (not modelling error), so a tight
         * tolerance is appropriate for every source type. */
        const double sync_tol = 1.0e-8;
        if (!rank && terr > sync_tol)
        {
            fprintf(stderr, "FAIL (sync err = %g)\n", terr);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        if (!rank)
        {
            fprintf(stderr, "ok (sync err = %g)\n", terr);
        }
    }

    FILE *error_file = NULL;
    if (!maxwell_params.recover)
    {
        /* l2_error_analytic is collective (MPI_Allreduce) — all ranks call it */
        const double terr = l2_error_analytic(&gfs, t);
        if (!rank)
        {
            error_file = fopen("l2_norm.dat", "w");
            if (!error_file)
            {
                fprintf(stderr, "rank 0: cannot open l2_norm.dat\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            printf("%20.16e %20.16e\n", t, terr);
            fprintf(error_file, "%20.16e %20.16e\n", t, terr);
            fflush(error_file);
        }
        maxwell_constraints(&gfs);
#ifndef TIMING_ONLY
        output_gfs_3D_h5(&gfs);
        if (out2d_z >= 0)
            output_gfs_2D_xy_h5(&gfs, out2d_z);
#endif
    }
    else
    {
        if (!rank)
        {
            error_file = fopen("l2_norm.dat", "a");
            if (!error_file)
            {
                fprintf(stderr, "rank 0: cannot open l2_norm.dat\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    BEGIN_TIMER(timer)
    {
        for (int it = it_start; it < maxwell_params.max_iterations; it++)
        {
            RK4_Step(&gfs, t, dt, maxwell_eq_time_deriv);
            t += dt;
#ifndef TIMING_ONLY
            if (it % output_every == 0)
            {
                if (rank == 0)
                {
                    printf("it %d, t %g\n", it, t);
                }
                maxwell_constraints(&gfs);
                output_gfs_3D_h5(&gfs);
                if (out2d_z >= 0)
                    output_gfs_2D_xy_h5(&gfs, out2d_z);
                const double terr = l2_error_analytic(&gfs, t);
                if (!rank)
                {
                    printf("%20.16e %20.16e\n", t, terr);
                    fprintf(error_file, "%20.16e %20.16e\n", t, terr);
                    fflush(error_file);
                }
            }
            if (checkpoint_every > 0 && it % checkpoint_every == 0)
            {
                write_checkpoint(&gfs, t, it,
                                 maxwell_params.max_checkpoints);
            }
#endif
        }
    }
    END_TIMER(timer)
    if (rank == 0)
    {
        print_timers();
        fclose(error_file);
    }

    /* We're done. Deallocate the gridfunctions and exit */
    ngfs_deallocate(&gfs);
    cleanup_3d_domain(&gfs.domain);

    MPI_ERROR(MPI_Finalize());
    return 0;
}
