#include "multigrid_parameters.h"
#include "parameter.hpp"
#include <cstring>
#include <iostream>

/* Default problem preset used when the TOML omits the [problem] section.
 * The string itself must match a name in g_problems[] (see
 * problem_registry.c). */
static constexpr const char *DEFAULT_PROBLEM_NAME =
    "manufactured_dirichlet_homog";

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

    /* Grid extents flow through the solver as int64_t (domain3d_st,
     * ngfs_3d, stride arithmetic), so use the 64-bit helper directly --
     * no narrowing cast needed. */
    param->global_nx_cells = parameters::get_positive_integer64_value("grid", "nx_cells", tbl);
    param->global_ny_cells = parameters::get_positive_integer64_value("grid", "ny_cells", tbl);
    param->global_nz_cells = parameters::get_positive_integer64_value("grid", "nz_cells", tbl);

    param->omega = parameters::get_positive_real_value("solver", "omega", tbl);

    /* The iteration-count fields are `int` because the downstream APIs
     * (gauss_seidel_3d, vcycle_3d, ngfs_3d_create_hierarchy) take `int`,
     * and physically reasonable values are well below INT_MAX.  The
     * 32-bit helper bounds-checks the TOML value against
     * [INT_MIN, INT_MAX] before narrowing, so any out-of-range input is
     * reported as a parser error instead of silently sign-flipping. */
    param->n_smooth = parameters::get_positive_integer32_value("solver", "n_smooth", tbl);
    param->n_iters  = parameters::get_positive_integer32_value("solver", "n_iters",  tbl);

    param->tol = parameters::get_nonnegative_real_value("solver", "tol", tbl);

    param->use_multigrid = parameters::get_boolean_value("solver", "multigrid", tbl) ? 1 : 0;

    if (param->use_multigrid)
    {
        param->subcycles = parameters::get_positive_integer32_value("solver", "subcycles", tbl);
        param->min_cells = parameters::get_positive_integer32_value("solver", "min_cells", tbl);
    }
    else
    {
        param->subcycles = 1;
        param->min_cells = 4;
    }

    /* [problem] is optional.  When absent, fall back to the default
     * preset (which reproduces the original hard-coded behaviour
     * bit-for-bit -- the back-compat invariant that lets every
     * pre-existing TOML keep working).  When present, validate its
     * keys against the allowlist below alongside [grid] and
     * [solver]. */
    {
        const auto problem_section = tbl["problem"].as_table();
        std::string name;
        if (problem_section)
        {
            name = parameters::get_string_value("problem", "name", tbl);
        }
        else
        {
            name = DEFAULT_PROBLEM_NAME;
        }

        if (name.size() >= sizeof(param->problem_name))
        {
            std::cerr << "invalid problem::name value -- '" << name
                      << "' exceeds " << (sizeof(param->problem_name) - 1)
                      << "-character limit\n";
            parameters::increment_error();
            name.clear();
        }
        std::strncpy(param->problem_name, name.c_str(),
                     sizeof(param->problem_name) - 1);
        param->problem_name[sizeof(param->problem_name) - 1] = '\0';
    }

    /* Reject any TOML entry that isn't in the documented schema.  This
     * catches typos like `mulitgrid = true`, which would otherwise be
     * silently ignored: the get_*_value helpers above would not find
     * the misspelled key, but `multigrid` *would* be reported as
     * "missing" -- and a user who only checks the boolean's effect
     * might never notice the mismatch.  Making unknown keys fatal is
     * therefore strictly more informative.
     *
     * `subcycles` and `min_cells` are listed unconditionally even
     * though they are only consumed when multigrid = true, because we
     * want to allow users to leave them in the file when toggling the
     * solver mode. */
    parameters::check_known_sections(tbl, { "grid", "solver", "problem" });
    parameters::check_known_keys(tbl, "grid",
        { "nx_cells", "ny_cells", "nz_cells" });
    parameters::check_known_keys(tbl, "solver",
        { "multigrid", "omega", "n_smooth", "n_iters", "tol",
          "subcycles", "min_cells" });
    parameters::check_known_keys(tbl, "problem",
        { "name" });

    return parameters::error_count();
}
