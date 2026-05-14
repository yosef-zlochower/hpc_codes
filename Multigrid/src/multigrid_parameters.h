#ifndef MULTIGRID_PARAMETERS_H
#define MULTIGRID_PARAMETERS_H

/* Public ABI for the multigrid-solver parameter file.
 *
 * Carries the POD parameter struct that the driver consumes, plus the
 * C-callable parse_parameter_file() entry point.  Free of toml++:
 * anything that needs the parser helpers (the parameters:: namespace)
 * includes parameter.hpp instead.
 *
 * The split keeps toml.hpp out of every translation unit that doesn't
 * implement the parser, which (a) lets the C driver include this
 * header directly without a __cplusplus guard around toml.hpp,
 * (b) avoids GCC 12's if-constexpr-in-generic-lambda bug
 * (toml.hpp:8239 under C++20), and (c) keeps the dependency structure
 * honest: parameter.hpp is an internal header of the parser, not a
 * public interface. */

#include <stdint.h>

/* Maximum length of a problem-preset name (including NUL).
 * 64 is generous: the longest registered preset is currently 32 chars. */
#define PARAM_PROBLEM_NAME_MAX 64

/* Maximum length of an output directory path (including NUL).
 * 256 covers reasonable absolute paths on the cluster filesystems
 * we run on. */
#define PARAM_OUTPUT_DIR_MAX 256

struct param_st
{
    int64_t global_nx_cells; /* number of physical cells in x */
    int64_t global_ny_cells; /* number of physical cells in y */
    int64_t global_nz_cells; /* number of physical cells in z */
    /* Per-axis Cartesian box bounds.  When [grid] x0/xN/y0/yN/z0/zN are
     * absent from the TOML they default to 0.0/1.0 (the back-compat
     * unit cube).  h_a = (b_a - a_a) / N_a is the per-axis spacing. */
    double  x0, xN;
    double  y0, yN;
    double  z0, zN;
    double  omega;
    int     n_smooth;
    int     n_iters;
    int     subcycles;     /* multigrid only */
    int     min_cells;     /* multigrid only */
    double  tol;
    int     use_multigrid; /* 1 = V-cycle multigrid, 0 = Gauss-Seidel */
    int     verbose;       /* 1 = print "Starting Vcycle" + per-level defect
                            * trace from vcycle_3d; 0 = silent.  Default 0. */
    char    problem_name[PARAM_PROBLEM_NAME_MAX]; /* problem preset key,
                                                   * defaults to
                                                   * "manufactured_dirichlet_homog"
                                                   * when [problem] is absent */
    char    output_dir[PARAM_OUTPUT_DIR_MAX];     /* per-rank JSON output
                                                   * directory; defaults to
                                                   * "" (cwd) when [output]
                                                   * is absent.  Driver
                                                   * mkdir-p's it from rank 0
                                                   * before writing. */
    int     write_defect;  /* [output] write_defect: 1 -> also dump VAR_DEF
                            * to the per-rank HDF5 file after the solve.
                            * Default 0 (back-compat: only VAR_SOL is
                            * written).  Useful for debugging a V-cycle
                            * stall by inspecting the per-cell residual. */
    int     write_rhs;     /* [output] write_rhs: 1 -> also dump VAR_RHS
                            * (the manufactured source term).  Default 0. */
};

/* parse_parameter_file is callable from C and C++.  In C++ the
 * extern "C" wrapper suppresses name mangling so the C-compiled driver
 * can link against the C++ implementation in multigrid_parameters.cc. */
#ifdef __cplusplus
extern "C" {
#endif

int parse_parameter_file(struct param_st *param, const char *fname);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif /* MULTIGRID_PARAMETERS_H */
