#ifndef MAXWELL_PARAMETERS_H
#define MAXWELL_PARAMETERS_H

/* The maxwell-solver parameter ABI.
 *
 * This header carries the POD parameter structs the rest of the solver
 * consumes (driver.cpp, maxwell_eqs.cpp, set_initial_data, etc.) plus
 * the C-callable parse_maxwell_parameters() entry point.  It is
 * deliberately free of toml++: anything that needs the parser helpers
 * (the parameters:: namespace, used internally by the parser
 * implementation) includes parameter.hpp instead.
 *
 * The split keeps toml.hpp out of every translation unit that doesn't
 * implement the parser, which (a) avoids GCC 12's
 * if-constexpr-in-generic-lambda bug, (b) shortens compile time for
 * the kernel-heavy translation units, and (c) makes the dependency
 * structure honest: parameter.hpp is an internal header of the parser,
 * not a public interface. */

#include <stdint.h>
#include "analytic_parameters.h" /* plane_wave_params, gaussian_beam_params,
                                  * te_waveguide_mode_params */

/* Parameters of the elliptical-lens permittivity profile, layered on
 * top of the background baseline eps_bg taken from [material.background]:
 *   eps(x,z) = (max - eps_bg) * exp(-s^2 * ((x-x0)^2/a^2 + (z-z0)^2/b^2))^2 + eps_bg
 * Consumed when [material] epsilon_type = "elliptical".  Mirrors the
 * [material.elliptical] TOML sub-table; max is the peak permittivity
 * at the lens centre, and the profile decays smoothly to eps_bg away
 * from it.  The "eps_" prefix is dropped from these keys since the
 * sub-table name already establishes the context. */
struct material_elliptical_params
{
    double max;    /* peak permittivity at the lens centre */
    double x0;     /* centre x */
    double z0;     /* centre z */
    double s;      /* steepness */
    double a;      /* semi-axis x */
    double b;      /* semi-axis z */
};

struct maxwell_param_st
{
    /* grid */
    int64_t nx;             /* global grid points in x; matches the int64_t
                             * widths in domain3d_st and ngfs so values flow
                             * through the pipeline without narrowing. */
    int64_t ny;             /* global grid points in y */
    int64_t nz;             /* global grid points in z */
    double  x0, y0, z0;    /* domain origin */
    double  xn, yn, zn;    /* domain upper corner */
    int     periodic_x;     /* 1 = periodic, 0 = non-periodic */
    int     periodic_y;
    int     periodic_z;

    /* solver */
    int     ghost_size;      /* ghost zone width */
    double  cfl_factor;      /* dt = cfl_factor * min(dx,dy,dz) */
    int     max_iterations;  /* maximum number of time steps */
    int     output_every;    /* output frequency (in iterations) */
    int     checkpoint_every; /* checkpoint frequency (0 = disabled) */
    int     max_checkpoints;  /* number of checkpoint sets to keep per run */
    int     output_2d_z_plane; /* global k index of an xy-slice to dump as
                                * HDF5 every output step; < 0 disables it
                                * (optional, default -1) */
    int     recover;         /* 1 = recover from checkpoint, 0 = fresh start */
    int     use_dissipation; /* 1 = enable Kreiss-Oliger dissipation */
    double  diss_coeff;      /* dissipation coefficient */
    double  tau;             /* SBP-SAT penalty strength (>= 0.5 for stability) */

    /* physics */
    double  kappa_D;        /* constraint-damping coefficient for div D */
    double  kappa_B;        /* constraint-damping coefficient for div B */

    /* source selection + per-source parameter sub-tables.  Each
     * sub-struct corresponds to a [source.<name>] TOML sub-table; only
     * the one matching source_type is consumed by the analytic source
     * functions, but the driver still populates all three so switching
     * sources between runs is just a TOML edit. */
    int     source_type;    /* 0 = plane_wave, 1 = gaussian_beam,
                             * 2 = te_waveguide_mode */
    struct plane_wave_params         plane_wave;
    struct gaussian_beam_params      gaussian_beam;
    struct te_waveguide_mode_params  te_waveguide_mode;

    /* material: [material.background] holds the always-used uniform
     * scalars (epsilon, mu, sigma), and the selector epsilon_type picks
     * whether the elliptical profile is layered on top.  When
     * epsilon_type = "uniform" the background value is used everywhere;
     * when "elliptical" the [material.elliptical] profile modulates it. */
    int     epsilon_type;   /* 0 = uniform, 1 = elliptical */
    double  epsilon;        /* background permittivity (from [material.background]) */
    double  mu;             /* permeability */
    double  sigma;          /* conductivity */
    struct material_elliptical_params elliptical;  /* [material.elliptical] */
};

#ifdef __cplusplus
extern "C" {
#endif

int parse_maxwell_parameters(struct maxwell_param_st *param, const char *fname);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif /* MAXWELL_PARAMETERS_H */
