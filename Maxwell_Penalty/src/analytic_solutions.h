#ifndef ANALYTIC_SOLUTIONS_H
#define ANALYTIC_SOLUTIONS_H
struct eb_st
{
    double Dx;
    double Dy;
    double Dz;
    double Bx;
    double By;
    double Bz;
    double PsiD;
    double PsiB;
    double rho;
};

/* Per-source parameter sub-structs.  Each mirrors a TOML [source.<name>]
 * sub-table read by maxwell_parameters.cc, and is the block of
 * parameters the corresponding analytic source function consumes. */
struct plane_wave_params
{
    double ax;       /* x-polarisation amplitude */
    double ay;       /* y-polarisation amplitude */
    double k;        /* wave number */
    double bump_a;   /* bump-envelope turn-on start */
    double bump_b;   /* bump-envelope turn-on end   */
};

struct gaussian_beam_params
{
    double w0;        /* beam waist radius */
    double z_waist;   /* z position of the waist */
    double k;         /* wave number */
    double amplitude; /* peak field amplitude */
    double ramp_a;    /* smooth temporal ramp: zero for t <= ramp_a */
    double ramp_b;    /* smooth temporal ramp: full for t >= ramp_b */
};

struct te_waveguide_mode_params
{
    int l;   /* transverse mode number in x (>= 1) */
    int m;   /* transverse mode number in y (>= 1) */
    int n;   /* longitudinal mode number in z (>= 1; propagation direction) */
};

/* Aggregate passed to every analytic source function.  Only the
 * sub-struct matching the active source_type is read by a given
 * function; the others are still populated by the driver, so a
 * sanity-check run doesn't require every TOML sub-table to match
 * the selected source. */
struct analytic_params_st
{
    double t0;                                    /* global time offset */
    struct plane_wave_params         plane_wave;
    struct gaussian_beam_params      gaussian_beam;
    struct te_waveguide_mode_params  te_waveguide_mode;
};

/* TE mode (l, m, n) of a rectangular unit-cube waveguide propagating in
 * +z.  Exact vacuum Maxwell solution; satisfies PEC boundary conditions
 * at x = 0, 1 and y = 0, 1, and is also periodic on the unit cube when
 * l, m, n are even.  Used by the convergence test as an exact reference
 * for both periodic and fully-physical boundary configurations.  See
 * the corresponding comment in analytic_solutions.c for details. */
void te_waveguide_mode(const struct analytic_params_st params, const double x,
                       const double y, const double z, const double t,
                       struct eb_st *A);

void incoming_plane_wave(const struct analytic_params_st params, const double x,
                         const double y, const double z, const double t,
                         struct eb_st *A);

/* Paraxial Gaussian beam: linearly polarised along x, travelling in +z.
 * Constructed from an electric Hertz potential Pi = x_hat * Phi_paraxial,
 * giving D = curl curl Pi and B = d_t(curl Pi). Both divergences are
 * zero by the identity div(curl v) = 0, independent of whether the
 * scalar Phi solves the wave equation. The paraxial Phi has wave-
 * equation residual O((k*w0)^-2), but div D = div B = 0 holds exactly,
 * so the extended-Maxwell damping fields PsiD, PsiB stay at numerical-
 * noise level throughout the evolution. See the comment block in
 * analytic_solutions.c for the full derivation. */
void incoming_gaussian_beam(const struct analytic_params_st params,
                            const double x, const double y,
                            const double z, const double t,
                            struct eb_st *A);
#endif
