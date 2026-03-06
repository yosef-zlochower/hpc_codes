#ifndef PARAMETER_H
#define PARAMETER_H

#include <stdint.h>

/* ── Visible to both C and C++ ────────────────────────────────────────────── */

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

/* parse_parameter_file is callable from C and C++.
 * In C++ the extern "C" wrapper suppresses name mangling so the C-compiled
 * driver can link against the C++ implementation in multigrid_parameters.cc. */
#ifdef __cplusplus
extern "C" {
#endif

int parse_parameter_file(struct param_st *param, const char *fname);

#ifdef __cplusplus
} /* end extern "C" */

/* ── C++ only: helper function prototypes in the parameters:: namespace ────── */

#include "toml.hpp"

namespace parameters {

void    reset_error();
int     error_count();

int64_t get_integer_value            (const char *section, const char *element,
                                      toml::table &tbl);
double  get_real_value               (const char *section, const char *element,
                                      toml::table &tbl);
int64_t get_positive_integer_value   (const char *section, const char *element,
                                      toml::table &tbl);
int64_t get_nonnegative_integer_value(const char *section, const char *element,
                                      toml::table &tbl);
double  get_positive_real_value      (const char *section, const char *element,
                                      toml::table &tbl);
double  get_nonnegative_real_value   (const char *section, const char *element,
                                      toml::table &tbl);
bool    get_boolean_value            (const char *section, const char *element,
                                      toml::table &tbl, bool required = true);

} /* namespace parameters */

#endif /* __cplusplus */

#endif /* PARAMETER_H */
