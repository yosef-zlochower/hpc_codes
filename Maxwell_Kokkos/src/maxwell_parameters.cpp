#include "maxwell_parameters.h"
#include "parameter.hpp"
#include <iostream>

extern "C" int parse_maxwell_parameters(struct maxwell_param_st *param,
                                        const char *fname)
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

    /* [grid] */
    /* Grid sizes are int64_t throughout the solver (domain3d_st, ngfs,
     * stride arithmetic), so use the 64-bit parser helper directly --
     * no narrowing cast is needed. */
    param->nx = parameters::get_positive_integer64_value("grid", "nx", tbl);
    param->ny = parameters::get_positive_integer64_value("grid", "ny", tbl);
    param->nz = parameters::get_positive_integer64_value("grid", "nz", tbl);

    param->x0 = parameters::get_real_value("grid", "x0", tbl);
    param->y0 = parameters::get_real_value("grid", "y0", tbl);
    param->z0 = parameters::get_real_value("grid", "z0", tbl);
    param->xn = parameters::get_real_value("grid", "xn", tbl);
    param->yn = parameters::get_real_value("grid", "yn", tbl);
    param->zn = parameters::get_real_value("grid", "zn", tbl);

    param->periodic_x = parameters::get_boolean_value("grid", "periodic_x", tbl) ? 1 : 0;
    param->periodic_y = parameters::get_boolean_value("grid", "periodic_y", tbl) ? 1 : 0;
    param->periodic_z = parameters::get_boolean_value("grid", "periodic_z", tbl) ? 1 : 0;

    /* [solver] -- the destination fields below are `int`, so use the
     * 32-bit parser helpers; they bounds-check the toml++ value
     * against [INT_MIN, INT_MAX] before narrowing and report a parser
     * error on overflow rather than silently truncating. */
    param->ghost_size       = parameters::get_positive_integer32_value   ("solver", "ghost_size",       tbl);
    param->cfl_factor       = parameters::get_positive_real_value        ("solver", "cfl_factor",       tbl);
    param->max_iterations   = parameters::get_positive_integer32_value   ("solver", "max_iterations",   tbl);
    param->output_every     = parameters::get_positive_integer32_value   ("solver", "output_every",     tbl);
    param->checkpoint_every = parameters::get_nonnegative_integer32_value("solver", "checkpoint_every", tbl);
    param->max_checkpoints  = parameters::get_positive_integer32_value   ("solver", "max_checkpoints",  tbl);
    param->output_2d_z_plane = parameters::get_integer32_value_or_default ("solver", "output_2d_z_plane", tbl, -1);
    param->recover         = parameters::get_boolean_value("solver", "recover", tbl, false) ? 1 : 0;
    param->use_dissipation = parameters::get_boolean_value("solver", "use_dissipation", tbl) ? 1 : 0;
    param->diss_coeff      = parameters::get_nonnegative_real_value("solver", "diss_coeff", tbl);
    param->tau             = parameters::get_positive_real_value("solver", "tau", tbl);

    /* [physics] */
    param->kappa_D = parameters::get_nonnegative_real_value("physics", "kappa_D", tbl);
    param->kappa_B = parameters::get_nonnegative_real_value("physics", "kappa_B", tbl);

    /* [source]
     *
     * The "source" group holds everything that drives the fields — both the
     * boundary injection on each physical face and the t = 0 initial data —
     * because the two are the same analytic function evaluated at different
     * (x,y,z,t). Which sub-table is consumed is picked by source.type; the
     * driver still populates all three sub-structs, so switching sources is
     * a one-line TOML edit with no per-sub-table reshuffling. */
    std::string src_type = parameters::get_string_value("source", "type", tbl);
    if (src_type == "plane_wave")
    {
        param->source_type = 0;
    }
    else if (src_type == "gaussian_beam")
    {
        param->source_type = 1;
    }
    else if (src_type == "te_waveguide_mode")
    {
        /* Convergence-test source: the analytic te_waveguide_mode is used as
         * boundary data on every physical face (not just lower-z). This
         * makes the SBP-SAT scheme imposition exact-to-analytic on each
         * face, so the numerical solution should converge to the analytic
         * at the scheme's global rate. */
        param->source_type = 2;
    }
    else
    {
        std::cerr << "invalid source::type — expected "
                  << "\"plane_wave\", \"gaussian_beam\", or "
                  << "\"te_waveguide_mode\"\n";
        parameters::increment_error();
    }

    /* [source.plane_wave] */
    param->plane_wave.ax     = parameters::get_real_value("source.plane_wave", "ax", tbl);
    param->plane_wave.ay     = parameters::get_real_value("source.plane_wave", "ay", tbl);
    param->plane_wave.k      = parameters::get_positive_real_value("source.plane_wave", "k", tbl);
    param->plane_wave.bump_a = parameters::get_nonnegative_real_value("source.plane_wave", "bump_a", tbl);
    param->plane_wave.bump_b = parameters::get_positive_real_value("source.plane_wave", "bump_b", tbl);

    /* [source.gaussian_beam] */
    param->gaussian_beam.w0        = parameters::get_positive_real_value("source.gaussian_beam", "w0", tbl);
    param->gaussian_beam.z_waist   = parameters::get_real_value("source.gaussian_beam", "z_waist", tbl);
    param->gaussian_beam.k         = parameters::get_positive_real_value("source.gaussian_beam", "k", tbl);
    param->gaussian_beam.amplitude = parameters::get_real_value("source.gaussian_beam", "amplitude", tbl);
    param->gaussian_beam.ramp_a    = parameters::get_nonnegative_real_value("source.gaussian_beam", "ramp_a", tbl);
    param->gaussian_beam.ramp_b    = parameters::get_positive_real_value("source.gaussian_beam", "ramp_b", tbl);

    /* [source.te_waveguide_mode] -- mode numbers are small ints (used
     * in `int` arithmetic in the analytic), so 32-bit helper applies. */
    param->te_waveguide_mode.l = parameters::get_positive_integer32_value("source.te_waveguide_mode", "l", tbl);
    param->te_waveguide_mode.m = parameters::get_positive_integer32_value("source.te_waveguide_mode", "m", tbl);
    param->te_waveguide_mode.n = parameters::get_positive_integer32_value("source.te_waveguide_mode", "n", tbl);

    /* [material]
     *
     * The selector epsilon_type chooses how permittivity varies in space.
     * [material.background] carries the always-used uniform scalars
     * (epsilon, mu, sigma); epsilon doubles as the uniform value (when
     * epsilon_type = "uniform") or the baseline of the elliptical profile
     * (when "elliptical").  When "elliptical", [material.elliptical] is
     * additionally read — it describes a lens-shaped bump layered on top
     * of the background. */
    param->epsilon = parameters::get_positive_real_value("material.background", "epsilon", tbl);
    param->mu      = parameters::get_positive_real_value("material.background", "mu",      tbl);
    param->sigma   = parameters::get_nonnegative_real_value("material.background", "sigma", tbl);

    std::string eps_type = parameters::get_string_value("material", "epsilon_type", tbl);
    if (eps_type == "uniform")
    {
        param->epsilon_type = 0;
    }
    else if (eps_type == "elliptical")
    {
        param->epsilon_type = 1;
        param->elliptical.max = parameters::get_positive_real_value("material.elliptical", "max", tbl);
        param->elliptical.x0  = parameters::get_real_value("material.elliptical", "x0", tbl);
        param->elliptical.z0  = parameters::get_real_value("material.elliptical", "z0", tbl);
        param->elliptical.s   = parameters::get_positive_real_value("material.elliptical", "s", tbl);
        param->elliptical.a   = parameters::get_positive_real_value("material.elliptical", "a", tbl);
        param->elliptical.b   = parameters::get_positive_real_value("material.elliptical", "b", tbl);
    }
    else
    {
        std::cerr << "invalid material::epsilon_type value — "
                  << "expected \"uniform\" or \"elliptical\"\n";
        parameters::increment_error();
    }

    return parameters::error_count();
}
