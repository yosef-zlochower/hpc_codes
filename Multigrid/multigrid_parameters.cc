#include "parameter.h"
#include <iostream>

extern "C" int parse_parameter_file(struct param_st *param, const char *fname)
{
    parameters::reset_error();

    toml::table tbl;
    try
    {
        tbl = toml::parse_file(fname);
    }
    catch (const toml::parse_error &err)
    {
        std::cerr << "TOML parse error in '" << fname << "':\n"
                  << err.description()
                  << "\n  (line " << err.source().begin.line << ")\n";
        return -1;
    }

    param->global_nx_cells = parameters::get_positive_integer_value("grid", "nx_cells", tbl);
    param->global_ny_cells = parameters::get_positive_integer_value("grid", "ny_cells", tbl);
    param->global_nz_cells = parameters::get_positive_integer_value("grid", "nz_cells", tbl);

    param->omega    = parameters::get_positive_real_value    ("solver", "omega",    tbl);
    param->n_smooth = (int)parameters::get_positive_integer_value("solver", "n_smooth", tbl);
    param->n_iters  = (int)parameters::get_positive_integer_value("solver", "n_iters",  tbl);
    param->tol      = parameters::get_nonnegative_real_value ("solver", "tol",      tbl);

    param->use_multigrid = parameters::get_boolean_value("solver", "multigrid", tbl) ? 1 : 0;

    if (param->use_multigrid)
    {
        param->subcycles = (int)parameters::get_positive_integer_value("solver", "subcycles", tbl);
        param->min_cells = (int)parameters::get_positive_integer_value("solver", "min_cells", tbl);
    }
    else
    {
        param->subcycles = 1;
        param->min_cells = 4;
    }

    return parameters::error_count();
}
