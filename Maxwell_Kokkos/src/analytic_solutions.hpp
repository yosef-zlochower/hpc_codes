#ifndef ANALYTIC_SOLUTIONS_HPP
#define ANALYTIC_SOLUTIONS_HPP
/* Header-only port of analytic_solutions.{c,h} from Maxwell_Penalty.
 * Every function is KOKKOS_INLINE_FUNCTION so it can be called from
 * device kernels (the SAT shell sweep evaluates source_state per
 * boundary point, and set_initial_data evaluates it per cell). */

#include <Kokkos_Core.hpp>
#include <math.h>

#include "analytic_parameters.h"  /* POD structs shared with maxwell_parameters.h */

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

/* Smooth-bump turn-on f(t, a, b): zero for t <= a, one for t >= b,
 * C^4 transition in between. */
KOKKOS_INLINE_FUNCTION
double analytic_bump_f(double x, double a, double b)
{
    if (x <= a) return 0.0;
    if (x >= b) return 1.0;
    const double c   = b - a;
    const double xa  = (x - a) / c;
    const double xb  = (x - b) / c;
    const double xa2 = xa * xa;
    const double xa4 = xa2 * xa2;
    const double xb2 = xb * xb;
    const double xb3 = xb2 * xb;
    return xa4 * (-4.0 * xb + 10.0 * xb2 - 20.0 * xb3 + 1.0);
}

KOKKOS_INLINE_FUNCTION
double analytic_bump_df(double x, double a, double b)
{
    if (x <= a) return 0.0;
    if (x >= b) return 0.0;
    const double c   = b - a;
    const double ic  = 1.0 / c;
    const double xa  = (x - a) / c;
    const double xb  = (x - b) / c;
    const double xa2 = xa * xa;
    const double xa3 = xa2 * xa;
    const double xb2 = xb * xb;
    const double xb3 = xb2 * xb;
    return 4.0 * xa3 *
           (xa * (5.0 * xb - 15.0 * xb2 - 1.0) - 4.0 * xb + 10.0 * xb2 -
            20.0 * xb3 + 1.0) *
           ic;
}

KOKKOS_INLINE_FUNCTION
void te_waveguide_mode(const struct analytic_params_st params, const double x,
                       const double y, const double z, const double t,
                       struct eb_st *A)
{
    const int l = params.te_waveguide_mode.l;
    const int m = params.te_waveguide_mode.m;
    const int n = params.te_waveguide_mode.n;

    const double omega  = M_PI * sqrt((double)(l*l + m*m + n*n));
    const double tt     = t - params.t0;
    const double phase  = n * M_PI * z - omega * tt;
    const double cphase = cos(phase);
    const double sphase = sin(phase);

    A->Dx =  omega * sin(m * M_PI * y) * cos(l * M_PI * x) * cphase / (l * M_PI);
    A->Dy = -omega * sin(l * M_PI * x) * cos(m * M_PI * y) * cphase / (m * M_PI);
    A->Dz = 0.0;
    A->Bx = n * sin(l * M_PI * x) * cos(m * M_PI * y) * cphase / m;
    A->By = n * sin(m * M_PI * y) * cos(l * M_PI * x) * cphase / l;
    A->Bz = -(l*l + m*m) * sphase
            * cos(l * M_PI * x) * cos(m * M_PI * y) / (l * m);
    A->PsiD = 0.0;
    A->PsiB = 0.0;
    A->rho  = 0.0;
}

KOKKOS_INLINE_FUNCTION
void incoming_plane_wave(const struct analytic_params_st params, const double x,
                         const double y, const double z, const double t,
                         struct eb_st *A)
{
    (void)x; (void)y;
    const double ax = params.plane_wave.ax;
    const double ay = params.plane_wave.ay;
    const double k  = params.plane_wave.k;
    const double a  = params.plane_wave.bump_a;
    const double b  = params.plane_wave.bump_b;
    const double F  = analytic_bump_f (t - z, a, b);
    const double dF = analytic_bump_df(t - z, a, b);

    A->Dx = ax * ( k * F * cos(k * (-t + z)) - dF * sin(k * (-t + z)));
    A->Dy = ay * (-k * F * sin(k * (-t + z)) - dF * cos(k * (-t + z)));
    A->Dz = 0.0;
    A->Bx = ay * (-k * F * sin(k * ( t - z)) + dF * cos(k * ( t - z)));
    A->By = ax * ( k * F * cos(k * ( t - z)) + dF * sin(k * ( t - z)));
    A->Bz = 0.0;
    A->PsiD = 0.0;
    A->PsiB = 0.0;
    A->rho  = 0.0;
}

KOKKOS_INLINE_FUNCTION
void incoming_gaussian_beam(const struct analytic_params_st params,
                            const double x, const double y,
                            const double z, const double t,
                            struct eb_st *A)
{
    const double w0     = params.gaussian_beam.w0;
    const double zw     = params.gaussian_beam.z_waist;
    const double k      = params.gaussian_beam.k;
    const double E0     = params.gaussian_beam.amplitude;
    const double ramp_a = params.gaussian_beam.ramp_a;
    const double ramp_b = params.gaussian_beam.ramp_b;

    const double zR    = 0.5 * k * w0 * w0;
    const double dz    = z - zw;
    const double denom = dz * dz + zR * zR;
    const double inv_R = dz / denom;
    const double phi_g = atan2(dz, zR);
    const double w2    = w0 * w0 * (1.0 + (dz / zR) * (dz / zR));
    const double w_z   = sqrt(w2);

    const double u_R             = (zR * zR - dz * dz) / (denom * denom);
    const double du_R_dz         = -2.0 * dz * (3.0 * zR * zR - dz * dz)
                                   / (denom * denom * denom);
    const double d_zR_over_denom = -2.0 * dz * zR / (denom * denom);

    const double r2        = x * x + y * y;
    const double amp       = (w0 / w_z) * exp(-r2 / w2);
    const double Phi_tilde = E0 * amp / (k * k);
    const double gamma     = k * z - k * t + 0.5 * k * r2 * inv_R - phi_g;
    const double cg        = cos(gamma);
    const double sg        = sin(gamma);

    const double fx = -2.0 * x / w2;
    const double fy = -2.0 * y / w2;
    const double fz = inv_R * (2.0 * r2 / w2 - 1.0);

    const double gx = k * x * inv_R;
    const double gy = k * y * inv_R;
    const double gz = k + 0.5 * k * r2 * u_R - zR / denom;

    const double fy_y = -2.0 / w2;
    const double gy_y = k * inv_R;
    const double fz_x = 4.0 * x * inv_R / w2;
    const double gz_x = k * x * u_R;
    const double fz_z = u_R * (2.0 * r2 / w2 - 1.0)
                        - 4.0 * r2 * inv_R * inv_R / w2;
    const double gz_z = 0.5 * k * r2 * du_R_dz - d_zR_over_denom;

#define RE_DD(fN, gN, fM, gM, dfM_dN, dgM_dN)                                  \
    (  ((fN) * (fM) - (gN) * (gM) + (dfM_dN)) * cg                             \
     - ((fN) * (gM) + (fM) * (gN) + (dgM_dN)) * sg)

    const double Re_yy = Phi_tilde * RE_DD(fy, gy, fy, gy, fy_y, gy_y);
    const double Re_zz = Phi_tilde * RE_DD(fz, gz, fz, gz, fz_z, gz_z);
    const double Re_xy = Phi_tilde * RE_DD(fx, gx, fy, gy, 0.0,  0.0 );
    const double Re_xz = Phi_tilde * RE_DD(fx, gx, fz, gz, fz_x, gz_x);
#undef RE_DD

    const double Re_ty = k * Phi_tilde * (gy * cg + fy * sg);
    const double Re_tz = k * Phi_tilde * (gz * cg + fz * sg);

    const double turn_on = analytic_bump_f(t, ramp_a, ramp_b);

    A->Dx = turn_on * (-(Re_yy + Re_zz));
    A->Dy = turn_on *   Re_xy;
    A->Dz = turn_on *   Re_xz;

    A->Bx = 0.0;
    A->By = turn_on *   Re_tz;
    A->Bz = turn_on * (-Re_ty);

    A->PsiD = 0.0;
    A->PsiB = 0.0;
    A->rho  = 0.0;
}

#endif /* ANALYTIC_SOLUTIONS_HPP */
