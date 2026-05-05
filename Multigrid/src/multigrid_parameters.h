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

struct param_st
{
    int64_t global_nx_cells; /* number of cells in x; grid points = cells + 1 */
    int64_t global_ny_cells; /* number of cells in y; grid points = cells + 1 */
    int64_t global_nz_cells; /* number of cells in z; grid points = cells + 1 */
    double  omega;
    int     n_smooth;
    int     n_iters;
    int     subcycles;     /* multigrid only */
    int     min_cells;     /* multigrid only */
    double  tol;
    int     use_multigrid; /* 1 = V-cycle multigrid, 0 = Gauss-Seidel */
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
