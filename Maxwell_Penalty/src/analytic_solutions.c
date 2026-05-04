#include "analytic_solutions.h"
#include <math.h>
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif
static double f(double x, double a, double b)
{
    if (x <= a)
    {
        return 0;
    }
    if (x >= b)
    {
        return 1.0;
    }

    const double c = b - a;
    const double xa = (x - a) / c;
    const double xb = (x - b) / c;

    const double xa2 = xa * xa;
    const double xa4 = xa2 * xa2;

    const double xb2 = xb * xb;
    const double xb3 = xb2 * xb;

    return xa4 * (-4 * xb + 10 * xb2 - 20 * xb3 + 1);
}

static double df(double x, double a, double b)
{
    if (x <= a)
    {
        return 0;
    }
    if (x >= b)
    {
        return 0;
    }
    const double c = b - a;
    const double ic = 1.0 / c;
    const double xa = (x - a) / c;
    const double xb = (x - b) / c;

    const double xa2 = xa * xa;
    const double xa3 = xa2 * xa;

    const double xb2 = xb * xb;
    const double xb3 = xb2 * xb;

    return 4 * xa3 *
           (xa * (5 * xb - 15 * xb2 - 1) - 4 * xb + 10 * xb2 - 20 * xb3 + 1) *
           ic;
}

/* TE mode (l, m, n) of a rectangular unit-cube waveguide, propagating in
 * +z with transverse standing-wave structure in (x, y).
 *   - Tangential E (D_y, D_z) vanishes at x = 0, 1 and tangential E
 *     (D_x, D_z) vanishes at y = 0, 1 — PEC waveguide walls.
 *   - D_z == 0 identically, B_z != 0: this is the TE (transverse-
 *     electric) family.
 *   - Dispersion relation:  omega^2 = pi^2 (l^2 + m^2 + n^2),
 *     longitudinal wave number k_z = n*pi, phase velocity
 *     omega/k_z = sqrt(l^2 + m^2 + n^2) / n.
 *   - When l, m, n are even integers the analytic is also periodic on
 *     the unit cube (every sin/cos has an integer number of full
 *     periods across [0, 1]), so the same formula serves the
 *     convergence test as both a periodic-box reference and a PEC-
 *     walled waveguide reference.
 * The mode numbers must satisfy l, m, n >= 1 to avoid the 1/(l*m) etc.
 * divisions below; parameter parsing already enforces this via
 * get_positive_integer_value. */
void te_waveguide_mode(const struct analytic_params_st params, const double x,
                       const double y, const double z, const double t,
                       struct eb_st *A)
{
    const int l = params.te_waveguide_mode.l;
    const int m = params.te_waveguide_mode.m;
    const int n = params.te_waveguide_mode.n;

    const double omega = M_PI * sqrt((double)(l*l + m*m + n*n));
    const double tt    = t - params.t0;
    const double phase = n * M_PI * z - omega * tt;  /* n*pi*z - omega*t */
    const double cphase = cos(phase);
    const double sphase = sin(phase);

    A->Dx =  omega * sin(m * M_PI * y) * cos(l * M_PI * x) * cphase / (l * M_PI);
    A->Dy = -omega * sin(l * M_PI * x) * cos(m * M_PI * y) * cphase / (m * M_PI);
    A->Dz = 0;
    A->Bx = n * sin(l * M_PI * x) * cos(m * M_PI * y) * cphase / m;
    A->By = n * sin(m * M_PI * y) * cos(l * M_PI * x) * cphase / l;
    A->Bz = -(l*l + m*m) * sphase
            * cos(l * M_PI * x) * cos(m * M_PI * y) / (l * m);
    A->PsiD = 0;
    A->PsiB = 0;
    A->rho  = 0;
}

void incoming_plane_wave(const struct analytic_params_st params, const double x,
                         const double y, const double z, const double t,
                         struct eb_st *A)
{
    const double ax = params.plane_wave.ax;
    const double ay = params.plane_wave.ay;
    const double k  = params.plane_wave.k;
    const double a  = params.plane_wave.bump_a;
    const double b  = params.plane_wave.bump_b;

    A->Dx = ax * (k * f(t - z, a, b) * cos(k * (-t + z)) -
                  df(t - z, a, b) * sin(k * (-t + z)));
    A->Dy = ay * (-k * f(t - z, a, b) * sin(k * (-t + z)) -
                  df(t - z, a, b) * cos(k * (-t + z)));
    A->Dz = 0;
    A->Bx = ay * (-k * f(t - z, a, b) * sin(k * (t - z)) +
                  df(t - z, a, b) * cos(k * (t - z)));
    A->By = ax * (k * f(t - z, a, b) * cos(k * (t - z)) +
                  df(t - z, a, b) * sin(k * (t - z)));
    A->Bz = 0;
    A->PsiD = 0;
    A->PsiB = 0;
    A->rho = 0;
}

/* Paraxial Gaussian beam, linearly polarised along x, travelling in +z,
 * constructed from an electric Hertz potential so that div D = div B = 0
 * identically.
 *
 * Scalar building block (the usual paraxial Gaussian beam):
 *   Phi_c(x,y,z,t) = (E0 / k^2) * A(r,z) * exp( i * gamma(x,y,z,t) )
 *     A(r,z)    = (w0 / w(z)) * exp(-r^2 / w(z)^2)
 *     gamma     = k z - k t + (k r^2) / (2 R(z)) - phi_g(z)
 *     w(z)^2    = w0^2 * (1 + ((z-zw)/zR)^2)        (beam-width squared)
 *     1/R(z)    = (z-zw) / ((z-zw)^2 + zR^2)        (wavefront curvature)
 *     phi_g(z)  = atan2(z-zw, zR)                    (Gouy phase)
 *     zR        = k w0^2 / 2                         (Rayleigh range)
 *
 * The fields are defined from the electric Hertz vector  Pi = x_hat*Re[Phi_c]:
 *   D = curl curl Pi        =>  D_x = -(d_yy + d_zz) Phi,
 *                                D_y =  d_x d_y Phi,
 *                                D_z =  d_x d_z Phi.
 *   B = d_t (curl Pi)       =>  B_x = 0,
 *                                B_y =  d_t d_z Phi,
 *                                B_z = -d_t d_y Phi.
 * The divergences are zero by the identity  div(curl v) = 0 , independent of
 * whether Phi solves the wave equation. Since our Phi is only the paraxial
 * Gaussian (wave-equation residual O((k w0)^-2)), the resulting D, B are
 * not an exact Maxwell state, but  div D = div B = 0  holds *exactly*. As
 * a result the extended-Maxwell damping fields Psi_D, Psi_B stay at
 * numerical-noise level throughout the evolution instead of absorbing an
 * O((k w0)^-2) (or, without any longitudinal correction, O(1)) divergence.
 *
 * To evaluate the second derivatives in closed form, introduce the
 * log-derivatives  f_mu = d_mu(ln A),  g_mu = d_mu gamma. Using
 * Phi_c = Phi_tilde * exp(i gamma)  with Phi_tilde = E0 A / k^2,
 *   Re[d_nu d_mu Phi_c] / Phi_tilde
 *     =  (f_nu f_mu - g_nu g_mu + d_nu f_mu) * cos(gamma)
 *       -(f_nu g_mu + f_mu g_nu + d_nu g_mu) * sin(gamma).
 * For the time-mixed derivatives  d_t gamma = -k  and  d_t(f_mu) = d_t(g_mu) = 0,
 * so  Re[d_t d_mu Phi_c] / Phi_tilde = k*(g_mu cos gamma + f_mu sin gamma).
 *
 * A smooth temporal turn-on f(t, ramp_a, ramp_b) multiplies every field
 * equally. Because it depends only on t, spatial derivatives don't see
 * it, so the divergence identities are preserved at every instant during
 * the ramp as well as after. */
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

    /* ----- z-only geometric quantities ----- */
    const double zR    = 0.5 * k * w0 * w0;
    const double dz    = z - zw;
    const double denom = dz * dz + zR * zR;
    const double inv_R = dz / denom;                        /* 1/R(z) */
    const double phi_g = atan2(dz, zR);
    const double w2    = w0 * w0 * (1.0 + (dz / zR) * (dz / zR));
    const double w_z   = sqrt(w2);

    /* z-derivatives that enter the phase-gradient second derivatives:
     *   u_R             = d_z(1/R)     (appears in g_z)
     *   du_R_dz         = d_z(u_R)     (appears in d_z g_z)
     *   d_zR_over_denom = d_z(zR/denom) (appears in d_z g_z) */
    const double u_R             = (zR * zR - dz * dz) / (denom * denom);
    const double du_R_dz         = -2.0 * dz * (3.0 * zR * zR - dz * dz)
                                   / (denom * denom * denom);
    const double d_zR_over_denom = -2.0 * dz * zR / (denom * denom);

    /* ----- (r, z, t) dependent quantities ----- */
    const double r2        = x * x + y * y;
    const double amp       = (w0 / w_z) * exp(-r2 / w2);
    const double Phi_tilde = E0 * amp / (k * k);            /* Hertz amplitude */
    const double gamma     = k * z - k * t + 0.5 * k * r2 * inv_R - phi_g;
    const double cg        = cos(gamma);
    const double sg        = sin(gamma);

    /* Log-derivatives  f_mu = d_mu(ln A) ,  g_mu = d_mu gamma. */
    const double fx = -2.0 * x / w2;
    const double fy = -2.0 * y / w2;
    const double fz = inv_R * (2.0 * r2 / w2 - 1.0);

    const double gx = k * x * inv_R;
    const double gy = k * y * inv_R;
    const double gz = k + 0.5 * k * r2 * u_R - zR / denom;

    /* Non-vanishing second-derivative auxiliaries. The symmetries
     *   d_x f_y = d_x g_y = 0 ,   d_y f_x = d_y g_x = 0
     * (each f,g has no x-dependence in its y-variant) let us drop many
     * cross terms from the formulas below. */
    const double fy_y = -2.0 / w2;                          /* d_y f_y */
    const double gy_y = k * inv_R;                          /* d_y g_y */
    const double fz_x = 4.0 * x * inv_R / w2;               /* d_x f_z */
    const double gz_x = k * x * u_R;                        /* d_x g_z */
    const double fz_z = u_R * (2.0 * r2 / w2 - 1.0)         /* d_z f_z */
                        - 4.0 * r2 * inv_R * inv_R / w2;
    const double gz_z = 0.5 * k * r2 * du_R_dz              /* d_z g_z */
                        - d_zR_over_denom;

    /* Re[d_nu d_mu Phi_c] / Phi_tilde -- shared closed form. */
#define RE_DD(fN, gN, fM, gM, dfM_dN, dgM_dN)                                  \
    (  ((fN) * (fM) - (gN) * (gM) + (dfM_dN)) * cg                             \
     - ((fN) * (gM) + (fM) * (gN) + (dgM_dN)) * sg)

    const double Re_yy = Phi_tilde * RE_DD(fy, gy, fy, gy, fy_y, gy_y);
    const double Re_zz = Phi_tilde * RE_DD(fz, gz, fz, gz, fz_z, gz_z);
    const double Re_xy = Phi_tilde * RE_DD(fx, gx, fy, gy, 0.0,  0.0 );
    const double Re_xz = Phi_tilde * RE_DD(fx, gx, fz, gz, fz_x, gz_x);
#undef RE_DD

    /* Re[d_t d_mu Phi_c] / Phi_tilde = k (g_mu cos gamma + f_mu sin gamma). */
    const double Re_ty = k * Phi_tilde * (gy * cg + fy * sg);
    const double Re_tz = k * Phi_tilde * (gz * cg + fz * sg);

    /* Smooth temporal turn-on: multiplies every field equally. */
    const double turn_on = f(t, ramp_a, ramp_b);

    /* D = curl curl (x_hat * Phi). */
    A->Dx = turn_on * (-(Re_yy + Re_zz));
    A->Dy = turn_on *   Re_xy;
    A->Dz = turn_on *   Re_xz;

    /* B = d_t curl (x_hat * Phi). */
    A->Bx = 0.0;
    A->By = turn_on *   Re_tz;
    A->Bz = turn_on * (-Re_ty);

    A->PsiD = 0.0;
    A->PsiB = 0.0;
    A->rho  = 0.0;
}
