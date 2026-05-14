#include "comm.h"
#include "domain.h"
#include "gauss_seidel.h"
#include "gf.h"
#include "io.h"
#include "multigrid.h"
#include "multigrid_parameters.h"
#include "problem.h"
#include "timer.h"
#include <errno.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Recursive mkdir, like `mkdir -p`.  Returns 0 if the directory
 * exists (or was created) on return, -1 on a hard failure.  Empty or
 * NULL `path` is treated as success (cwd already exists).  Used by
 * the driver to ensure the output directory exists before per-rank
 * writes -- only rank 0 calls this; the others MPI_Barrier afterward. */
static int driver_mkdir_p(const char *path)
{
    if (!path || !path[0]) return 0;

    char tmp[PARAM_OUTPUT_DIR_MAX];
    snprintf(tmp, sizeof(tmp), "%s", path);
    /* Strip any trailing slashes so we don't try to mkdir "" at the end. */
    size_t len = strlen(tmp);
    while (len > 1 && tmp[len-1] == '/') tmp[--len] = '\0';

    /* Walk the path component by component, creating each as we go. */
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
    return 0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (argc < 2)
    {
        if (mpi_rank == 0)
            fprintf(stderr, "Usage: %s <params.toml>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    struct param_st param;
    if (parse_parameter_file(&param, argv[1]) != 0)
    {
        if (mpi_rank == 0)
            fprintf(stderr, "Failed to parse parameter file '%s'\n", argv[1]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (param.global_nx_cells < 1 || param.global_ny_cells < 1 || param.global_nz_cells < 1)
    {
        if (mpi_rank == 0)
            fprintf(stderr,
                    "Error: nx_cells=%lld ny_cells=%lld nz_cells=%lld — "
                    "each dimension needs at least 1 cell.\n",
                    (long long)param.global_nx_cells,
                    (long long)param.global_ny_cells,
                    (long long)param.global_nz_cells);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* ---- Resolve the problem preset ---- */
    const struct problem_t *problem = problem_lookup(param.problem_name);
    if (!problem)
    {
        if (mpi_rank == 0)
            fprintf(stderr, "Unknown problem preset '%s'\n", param.problem_name);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Validate per-axis bounds. */
    if (!(param.xN > param.x0) || !(param.yN > param.y0) || !(param.zN > param.z0))
    {
        if (mpi_rank == 0)
            fprintf(stderr,
                    "Error: domain bounds must satisfy aN > a0 on every axis "
                    "(got x: [%g, %g], y: [%g, %g], z: [%g, %g])\n",
                    param.x0, param.xN, param.y0, param.yN, param.z0, param.zN);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* ---- Domain decomposition ---- */
    size_t dims[3]     = { (size_t)param.global_nx_cells,
                           (size_t)param.global_ny_cells,
                           (size_t)param.global_nz_cells };
    size_t topology[3] = { 0, 0, 0 };
    automatic_topology(3, dims, (size_t)mpi_size, topology);

    const int px = (int)topology[0];
    const int py = (int)topology[1];
    const int pz = (int)topology[2];

    /* Per-axis spacing on the user-supplied box (defaults to [0,1]^3
     * when [grid] x0/xN/... are absent in the TOML). */
    const double dx = (param.xN - param.x0) / param.global_nx_cells;
    const double dy = (param.yN - param.y0) / param.global_ny_cells;
    const double dz = (param.zN - param.z0) / param.global_nz_cells;

    /* ---- Allocate root grid (gs=1 for 7-point stencil) ---- */
    struct ngfs_3d gfs;
    gfs.vars = NULL;

    /* Derive per-face Neumann flags from the preset's bc_spec for the
     * domain layout.  An axis is cell-centred when both ends are
     * Neumann; otherwise vertex-centred (DD or hybrid).  Order matches
     * face_id_t: LOWER_X, UPPER_X, LOWER_Y, UPPER_Y, LOWER_Z, UPPER_Z. */
    const bool neumann_face[6] = {
        problem->bc.face[FACE_LOWER_X].kind == BC_NEUMANN,
        problem->bc.face[FACE_UPPER_X].kind == BC_NEUMANN,
        problem->bc.face[FACE_LOWER_Y].kind == BC_NEUMANN,
        problem->bc.face[FACE_UPPER_Y].kind == BC_NEUMANN,
        problem->bc.face[FACE_LOWER_Z].kind == BC_NEUMANN,
        problem->bc.face[FACE_UPPER_Z].kind == BC_NEUMANN,
    };

    setup_3d_domain(px, py, pz, mpi_rank,
                    param.global_nx_cells,
                    param.global_ny_cells,
                    param.global_nz_cells,
                    neumann_face,
                    /*gs=*/1,
                    param.x0, param.y0, param.z0,
                    dx, dy, dz,
                    &gfs.domain);

    ngfs_3d_allocate(/*nvars=*/3, &gfs);

    /* Stamp the preset's BC spec onto the root level.  The hierarchy
     * constructor (called next) homogenises this for every coarse
     * level via bc_spec_homogenize -- the unknown on a coarse grid is
     * the correction e_H, whose BC is always homogeneous. */
    gfs.bc = malloc(sizeof(struct bc_spec_t));
    if (!gfs.bc)
    {
        if (mpi_rank == 0)
            fprintf(stderr, "Out of memory allocating bc_spec_t\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    *gfs.bc = problem->bc;

    /* ---- Build coarse-grid hierarchy (multigrid only) ---- */
    if (param.use_multigrid)
    {
        if (ngfs_3d_create_hierarchy(&gfs, param.min_cells) != 0)
        {
            if (mpi_rank == 0)
                fprintf(stderr, "Hierarchy creation failed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    if (mpi_rank == 0)
    {
        printf("Grid:     %lld x %lld x %lld cells\n",
               (long long)param.global_nx_cells,
               (long long)param.global_ny_cells,
               (long long)param.global_nz_cells);
        printf("Topology: %zu x %zu x %zu  (%d ranks)\n",
               topology[0], topology[1], topology[2], mpi_size);
        printf("Problem:  %s\n", problem->name);
        printf("Solver:   %s\n",
               param.use_multigrid ? "multigrid V-cycle" : "Gauss-Seidel");
        printf("omega = %.4f,  n_smooth = %d,  n_iters = %d,  tol = %g\n",
               param.omega, param.n_smooth, param.n_iters, param.tol);

        if (param.use_multigrid)
        {
            int nlevels = 1;
            struct ngfs_3d *p = gfs.child;
            while (p) { nlevels++; p = p->child; }
            printf("subcycles = %d,  min_cells = %d,  levels = %d\n",
                   param.subcycles, param.min_cells, nlevels);
        }
    }

    /* ---- Initialise from the preset ---- */
    problem_initialise_rhs   (&gfs, problem);
    problem_apply_initial_bc (&gfs, problem);
    apply_bc_3d(&gfs, VAR_SOL);

    /* ---- Solve ---- */
    if (param.use_multigrid)
        vcycle_3d_register_timers();

    double norm = calc_defect_3d(&gfs);
    if (mpi_rank == 0)
        printf("\niter %4d  |defect|_inf = %12.6e\n", 0, norm);

    for (int it = 1; it <= param.n_iters; it++)
    {
        if (param.use_multigrid)
        {
            norm = vcycle_3d(&gfs, param.n_smooth, param.omega,
                             param.tol, param.subcycles, param.verbose);
        }
        else
        {
            gauss_seidel_3d(&gfs, param.n_smooth, param.omega);
            norm = calc_defect_3d(&gfs);
        }

        /* Singular problems (all-Neumann): project the solution onto
         * the mean-zero subspace.  The discrete operator has the
         * constant function in its null space, so the V-cycle leaves
         * the mean of u undetermined; periodic projection prevents
         * round-off drift along that null direction. */
        if (problem->singular)
            problem_project_mean_zero(&gfs, VAR_SOL);

        /* The per-outer-iteration progress line stays unconditional --
         * it's a single line per V-cycle (60 lines total for n_iters=60)
         * and is the primary signal of progress; the verbose flag only
         * gates the much chattier per-level V-cycle trace inside
         * vcycle_3d. */
        if (mpi_rank == 0)
            printf("iter %4d  |defect|_inf = %12.6e\n", it, norm);

        if (norm < param.tol)
            break;
    }

    /* ---- Error vs exact solution (when the preset supplies one) ---- */
    const double err = problem_compute_max_error(&gfs, problem);
    if (mpi_rank == 0 && err >= 0.0)
        printf("\n|u - u_exact|_inf = %12.6e\n", err);

    /* ---- Per-phase wall-clock breakdown (rank 0 only) ----
     * Per-rank wall-clock totals; rank 0's numbers are representative
     * on the uniform-decomposition case but ranks at the domain edge
     * can differ slightly because their face-buffer work is
     * asymmetric.  For a full picture across ranks, run with
     * `mpirun -np <N> ... 2>&1 | grep vcycle` (every rank prints). */
    if (param.use_multigrid && mpi_rank == 0)
    {
        printf("\nWall-clock breakdown (rank 0):\n");
        print_timers();
    }

    /* ---- Per-rank JSON output ----
     * If [output] dir is set in the TOML, rank 0 mkdirs the directory
     * and the rest wait at the barrier; all ranks then write their
     * tile under that directory.  Empty dir = cwd (back-compat). */
    if (param.output_dir[0] != '\0' && mpi_rank == 0)
    {
        if (driver_mkdir_p(param.output_dir) != 0)
        {
            fprintf(stderr,
                    "rank 0: mkdir -p '%s' failed: %s\n",
                    param.output_dir, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    MPI_Barrier(gfs.domain.cart_comm);

    output_3d_gf(&gfs, VAR_SOL, param.output_dir);
    /* Optional defect and RHS dumps -- enabled by [output] write_defect /
     * write_rhs in the TOML.  VAR_DEF is fresh because every solve path
     * (vcycle_3d's final post-smooth defect compute, or the explicit
     * calc_defect_3d after gauss_seidel_3d in the non-multigrid branch)
     * writes it on the last outer iteration; VAR_RHS is set once by
     * problem_initialise_rhs and never modified. */
    if (param.write_defect)
        output_3d_gf(&gfs, VAR_DEF, param.output_dir);
    if (param.write_rhs)
        output_3d_gf(&gfs, VAR_RHS, param.output_dir);

    /* ---- Cleanup ---- */
    ngfs_3d_deallocate(&gfs);
    cleanup_3d_domain(&gfs.domain);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
