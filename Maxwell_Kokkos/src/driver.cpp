#include "comm.hpp"
#include "derivatives.hpp"        /* SBP42_CLOSURE_ROWS */
#include "gf.hpp"
#include "io.hpp"
#include "maxwell_eqs.hpp"
#include "rk4.hpp"
#include "timer.h"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

extern "C" {
#include "domain.h"
}

/* Globals consumed by maxwell_eqs.cpp; populated below from TOML. */
struct maxwell_param_st   maxwell_params;
struct analytic_params_st analytic_params;

int main(int argc, char **argv)
{
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    {
        std::fprintf(stderr, "MPI_Init failed\n");
        return 1;
    }
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        int size, rank;
        MPI_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &size));
        MPI_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

        if (argc != 2)
        {
            if (rank == 0)
                std::fprintf(stderr,
                             "Usage: %s <parameter_file.toml>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        const char *param_file = argv[1];
        if (parse_maxwell_parameters(&maxwell_params, param_file) != 0)
        {
            if (rank == 0)
                std::fprintf(stderr,
                             "Error reading parameter file '%s'\n", param_file);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        const size_t grid_dims[3] = { static_cast<size_t>(maxwell_params.nx),
                                      static_cast<size_t>(maxwell_params.ny),
                                      static_cast<size_t>(maxwell_params.nz) };
        size_t topo[3];
        if (automatic_topology(3, grid_dims, (size_t)size, topo) != 0)
        {
            if (rank == 0)
                std::fprintf(stderr, "automatic_topology() failed\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        const int cpux = (int)topo[0];
        const int cpuy = (int)topo[1];
        const int cpuz = (int)topo[2];

        if (rank == 0)
        {
            std::fprintf(stderr,
                "MPI topology: %d x %d x %d  (%d processes)\n"
                "Kokkos exec space: %s\n",
                cpux, cpuy, cpuz, size,
                Kokkos::DefaultExecutionSpace::name());
        }

        const int gs = maxwell_params.ghost_size;
        const int64_t global_nx = maxwell_params.nx;
        const int64_t global_ny = maxwell_params.ny;
        const int64_t global_nz = maxwell_params.nz;
        if (global_nx < 2 * gs || global_ny < 2 * gs || global_nz < 2 * gs)
        {
            if (rank == 0)
                std::fprintf(stderr,
                    "Grid dimensions must be >= 2 * ghost_size (%d)\n",
                    2 * gs);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        analytic_params.t0                = 0.0;
        analytic_params.plane_wave        = maxwell_params.plane_wave;
        analytic_params.gaussian_beam     = maxwell_params.gaussian_beam;
        analytic_params.te_waveguide_mode = maxwell_params.te_waveguide_mode;

        const int n_evol_vars = N_EVOL;
        const int n_aux_vars  = N_AUX;

        int timer = register_timer("/main");
        {
            char hostbuf[1024];
            gethostname(hostbuf, sizeof hostbuf);
            std::fprintf(stderr, "rank %d of %d running on %s\n",
                         rank, size, hostbuf);
        }

        NGFS gfs{};
        setup_3d_domain(cpux, cpuy, cpuz, rank,
                        global_nx, global_ny, global_nz, gs,
                        maxwell_params.x0, maxwell_params.y0,
                        maxwell_params.z0,
                        maxwell_params.xn, maxwell_params.yn,
                        maxwell_params.zn,
                        &gfs.domain,
                        maxwell_params.periodic_x ? PERIODIC : NON_PERIODIC,
                        maxwell_params.periodic_y ? PERIODIC : NON_PERIODIC,
                        maxwell_params.periodic_z ? PERIODIC : NON_PERIODIC);

        {
            const int64_t min_n = 2 * SBP42_CLOSURE_ROWS;
            const int small_x = gfs.domain.bbox.x.lower &&
                                gfs.domain.bbox.x.upper && gfs.domain.nx < min_n;
            const int small_y = gfs.domain.bbox.y.lower &&
                                gfs.domain.bbox.y.upper && gfs.domain.ny < min_n;
            const int small_z = gfs.domain.bbox.z.lower &&
                                gfs.domain.bbox.z.upper && gfs.domain.nz < min_n;
            if (small_x || small_y || small_z)
            {
                std::fprintf(stderr,
                    "rank %d: local grid too small for SBP-4-2 closures.\n",
                    rank);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }

        ngfs_allocate(n_evol_vars, n_aux_vars, &gfs);
        for (int i = 0; i < N_EVOL; i++)
            gf_rename_evol(&gfs, i, evolved_field_names[i]);
        for (int i = 0; i < N_AUX; i++)
            gf_rename_aux (&gfs, i, aux_field_names    [i]);

        const double min_dspace = std::fmin(gfs.dx, std::fmin(gfs.dy, gfs.dz));
        const double dt = maxwell_params.cfl_factor * min_dspace;
        const int output_every = maxwell_params.output_every;
        const int checkpoint_every = maxwell_params.checkpoint_every;
        const int out2d_z = maxwell_params.output_2d_z_plane; /* <0 = off */

        double t = 0.0;
        int it_start = 1;

        if (maxwell_params.recover)
        {
            read_checkpoint(&gfs, &t, &it_start);
            it_start += 1;
            set_output_counter_3D(it_start / output_every);
            set_output_counter_2D_xy_h5(it_start / output_every);
            if (!rank)
                std::fprintf(stderr,
                             "Resuming from iteration %d, t = %g\n",
                             it_start, t);
        }
        else
        {
            set_initial_data(&gfs, t);

            if (!rank) std::fprintf(stderr, "Testing Sync ... ");
            /* kidx = -1 syncs the `state` buffer directly — the C
             * version did this by aliasing dot to new before calling
             * sync_vars; we expose the choice as an explicit argument. */
            sync_vars(&gfs, EVOLVED, /*kidx=*/-1);
            const double terr = l2_error_analytic(&gfs, t);
            const double sync_tol = 1.0e-8;
            if (!rank && terr > sync_tol)
            {
                std::fprintf(stderr, "FAIL (sync err = %g)\n", terr);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
            if (!rank) std::fprintf(stderr, "ok (sync err = %g)\n", terr);
        }

        FILE *error_file = nullptr;
        if (!maxwell_params.recover)
        {
            const double terr = l2_error_analytic(&gfs, t);
            if (!rank)
            {
                error_file = std::fopen("l2_norm.dat", "w");
                assert(error_file);
                std::printf("%20.16e %20.16e\n", t, terr);
                std::fprintf(error_file, "%20.16e %20.16e\n", t, terr);
                std::fflush(error_file);
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
                error_file = std::fopen("l2_norm.dat", "a");
                assert(error_file);
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
                    if (rank == 0) std::printf("it %d, t %g\n", it, t);
                    maxwell_constraints(&gfs);
                    output_gfs_3D_h5(&gfs);
                    if (out2d_z >= 0)
                        output_gfs_2D_xy_h5(&gfs, out2d_z);
                    const double terr = l2_error_analytic(&gfs, t);
                    if (!rank)
                    {
                        std::printf("%20.16e %20.16e\n", t, terr);
                        std::fprintf(error_file, "%20.16e %20.16e\n",
                                     t, terr);
                        std::fflush(error_file);
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
            if (error_file) std::fclose(error_file);
        }

        ngfs_deallocate(&gfs);
        cleanup_3d_domain(&gfs.domain);
    }
    Kokkos::finalize();
    MPI_ERROR(MPI_Finalize());
    return rc;
}
