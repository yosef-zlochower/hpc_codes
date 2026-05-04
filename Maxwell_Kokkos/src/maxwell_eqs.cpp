#include "maxwell_eqs.hpp"
#include "comm.hpp"
#include "derivatives.hpp"
#include "numerical.hpp"
#include "simple_maxwell.h"
#include "timer.h"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cmath>
#include <cassert>

/* ──────────────────────────────────────────────────────────────────────────
 * SBP-SAT BOUNDARY TREATMENT (see comments in the C-version maxwell_eqs.c
 * for the derivation; the kernel structure is mirrored verbatim here.)
 *
 * Three shell regions:
 *   (A) z-shell × full(j,i)
 *   (B) z-interior × y-shell × full i
 *   (C) z-interior × y-interior × x-shell
 * partition the boundary shell so each point is visited exactly once.
 * SAT contributions are additive across faces; up to three faces fire at
 * a corner.
 *
 * The KOKKOS_LAMBDA bodies use SIMPLE_MAXWELL_INTERIOR_DOT and the six
 * APPLY_SAT_*_{X,Y,Z} macros from simple_maxwell.h verbatim — they work
 * unchanged because DECLARE_EVOLVED_VARS / DECLARE_AUX_VARS produce
 * raw double* pointers (extracted via Kokkos::View::data()) which the
 * macros' [ijk] accesses then index.
 * ──────────────────────────────────────────────────────────────────────── */

/* Order must match the evolved_slot enum in maxwell_eqs.hpp.
 * (C99 designated initialisers are a non-portable GNU extension in C++17,
 * so we use positional initialisers here.) */
const char *const evolved_field_names[N_EVOL] = {
    "Dx", "Dy", "Dz", "Bx", "By", "Bz", "PsiD", "PsiB", "rho",
};
/* Order must match the auxilliary_slot enum. */
const char *const aux_field_names[N_AUX] = {
    "ieps", "imu", "sigma", "cD", "cB",
};

typedef enum {
    ST_D4CEN = 0,
    ST_SBP_L0, ST_SBP_L1, ST_SBP_L2, ST_SBP_L3,
    ST_SBP_R0, ST_SBP_R1, ST_SBP_R2, ST_SBP_R3
} stencil_t;

KOKKOS_INLINE_FUNCTION
stencil_t stencil_at(int64_t i, int64_t n, int phys_lo, int phys_hi)
{
    if (phys_lo && i < SBP42_CLOSURE_ROWS)
        return (stencil_t)(ST_SBP_L0 + i);
    if (phys_hi && i >= n - SBP42_CLOSURE_ROWS)
        return (stencil_t)(ST_SBP_R0 + (n - 1 - i));
    return ST_D4CEN;
}

KOKKOS_INLINE_FUNCTION
double apply_stencil(const double *f, int64_t ijk, stencil_t s,
                     int64_t stride, double h)
{
    switch (s)
    {
        case ST_D4CEN:  return D4CEN    (f, ijk, stride, h);
        case ST_SBP_L0: return SBP42_L0 (f, ijk, stride, h);
        case ST_SBP_L1: return SBP42_L1 (f, ijk, stride, h);
        case ST_SBP_L2: return SBP42_L2 (f, ijk, stride, h);
        case ST_SBP_L3: return SBP42_L3 (f, ijk, stride, h);
        case ST_SBP_R0: return SBP42_RN  (f, ijk, stride, h);
        case ST_SBP_R1: return SBP42_RNm1(f, ijk, stride, h);
        case ST_SBP_R2: return SBP42_RNm2(f, ijk, stride, h);
        case ST_SBP_R3: return SBP42_RNm3(f, ijk, stride, h);
    }
    return 0.0;
}

/* Source dispatch — must be device-callable. The active source is selected
 * by maxwell_params.source_type (which we pass through `src_type`). */
KOKKOS_INLINE_FUNCTION
void source_state(int src_type, const analytic_params_st &p,
                  double x, double y, double z, double t, eb_st *A)
{
    switch (src_type)
    {
        case 1: incoming_gaussian_beam(p, x, y, z, t, A); break;
        case 2: te_waveguide_mode    (p, x, y, z, t, A); break;
        default: incoming_plane_wave (p, x, y, z, t, A); break;
    }
}

typedef enum {
    FACE_LOWER_X, FACE_UPPER_X,
    FACE_LOWER_Y, FACE_UPPER_Y,
    FACE_LOWER_Z, FACE_UPPER_Z,
} sat_face_t;

KOKKOS_INLINE_FUNCTION
void sat_boundary_data(int src_type, const analytic_params_st &p,
                       sat_face_t face, double x, double y, double z,
                       double t, eb_st *A)
{
    A->Dx = A->Dy = A->Dz = 0.0;
    A->Bx = A->By = A->Bz = 0.0;
    A->PsiD = A->PsiB = 0.0;
    A->rho = 0.0;
    if (src_type == 2)
    {
        te_waveguide_mode(p, x, y, z, t, A);
        return;
    }
    if (face == FACE_LOWER_Z)
        source_state(src_type, p, x, y, z, t, A);
}

/* ===========================================================================
 * TIME-DERIVATIVE FUNCTION
 * Kokkos parallel_for split into a deep-interior kernel and three shell
 * kernels (A: z-shell, B: y-shell within z-interior, C: x-shell within
 * y/z-interior) that mirror the C version's three-region partition.
 * =========================================================================== */
void maxwell_eq_time_deriv(NGFS *gfs, const double t, int kidx)
{
    DECLARE_EVOLVED_VARS(gfs, kidx);
    DECLARE_AUX_VARS(gfs);

    const double four_pi = 12.566370614359172954;
    const double kappa_B = maxwell_params.kappa_B;
    const double kappa_D = maxwell_params.kappa_D;
    const double tau     = maxwell_params.tau;
    const int    src_type = maxwell_params.source_type;
    const analytic_params_st p_local = analytic_params;

    static int timer_dot = -1;
    if (timer_dot < 0) timer_dot = register_timer("/Evol/dot");

    const int phys_xl = gfs->domain.bbox.x.lower;
    const int phys_xu = gfs->domain.bbox.x.upper;
    const int phys_yl = gfs->domain.bbox.y.lower;
    const int phys_yu = gfs->domain.bbox.y.upper;
    const int phys_zl = gfs->domain.bbox.z.lower;
    const int phys_zu = gfs->domain.bbox.z.upper;

    const int64_t i_min = phys_xl ? 0 : D4CEN_HALF;
    const int64_t i_max = phys_xu ? nx : nx - D4CEN_HALF;
    const int64_t j_min = phys_yl ? 0 : D4CEN_HALF;
    const int64_t j_max = phys_yu ? ny : ny - D4CEN_HALF;
    const int64_t k_min = phys_zl ? 0 : D4CEN_HALF;
    const int64_t k_max = phys_zu ? nz : nz - D4CEN_HALF;

    const int64_t i_int_min = phys_xl ? SBP42_CLOSURE_ROWS : D4CEN_HALF;
    const int64_t i_int_max = phys_xu ? nx - SBP42_CLOSURE_ROWS : nx - D4CEN_HALF;
    const int64_t j_int_min = phys_yl ? SBP42_CLOSURE_ROWS : D4CEN_HALF;
    const int64_t j_int_max = phys_yu ? ny - SBP42_CLOSURE_ROWS : ny - D4CEN_HALF;
    const int64_t k_int_min = phys_zl ? SBP42_CLOSURE_ROWS : D4CEN_HALF;
    const int64_t k_int_max = phys_zu ? nz - SBP42_CLOSURE_ROWS : nz - D4CEN_HALF;

    const double scale_x = tau * SBP42_HINV_0 / dx;
    const double scale_y = tau * SBP42_HINV_0 / dy;
    const double scale_z = tau * SBP42_HINV_0 / dz;

    const double x0 = gfs->x0, y0 = gfs->y0, z0 = gfs->z0;

    using Range3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    BEGIN_TIMER(timer_dot)
    {
        /* ---------------- DEEP INTERIOR (D4CEN everywhere) -------------- */
#undef DIFFX
#undef DIFFY
#undef DIFFZ
#define DIFFX(f_) D4CEN(f_, ijk, di, dx)
#define DIFFY(f_) D4CEN(f_, ijk, dj, dy)
#define DIFFZ(f_) D4CEN(f_, ijk, dk, dz)
        Kokkos::parallel_for("dot/interior",
            Range3D({i_int_min, j_int_min, k_int_min},
                    {i_int_max, j_int_max, k_int_max}),
            KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                const int64_t ijk = i + j * nx + k * nx * ny;
                SIMPLE_MAXWELL_INTERIOR_DOT;
            });

        /* ---------------- BOUNDARY SHELL (three non-overlapping regions) - */
#undef DIFFX
#undef DIFFY
#undef DIFFZ
#define DIFFX(f_) apply_stencil(f_, ijk, _sx, di, dx)
#define DIFFY(f_) apply_stencil(f_, ijk, _sy, dj, dy)
#define DIFFZ(f_) apply_stencil(f_, ijk, _sz, dk, dz)

#define SHELL_POINT_BODY                                                       \
    do {                                                                       \
        const int64_t ijk = i + j * nx + k * nx * ny;                          \
        SIMPLE_MAXWELL_INTERIOR_DOT;                                           \
        eb_st g_sat;                                                           \
        if (phys_zl && k == 0) {                                               \
            sat_boundary_data(src_type, p_local, FACE_LOWER_Z,                 \
                              x0 + i * dx, y0 + j * dy, z0, t, &g_sat);        \
            APPLY_SAT_LOWER_Z(scale_z, g_sat);                                 \
        }                                                                      \
        if (phys_zu && k == nz - 1) {                                          \
            sat_boundary_data(src_type, p_local, FACE_UPPER_Z,                 \
                              x0 + i * dx, y0 + j * dy, z0 + k * dz, t,        \
                              &g_sat);                                         \
            APPLY_SAT_UPPER_Z(scale_z, g_sat);                                 \
        }                                                                      \
        if (phys_xl && i == 0) {                                               \
            sat_boundary_data(src_type, p_local, FACE_LOWER_X,                 \
                              x0, y0 + j * dy, z0 + k * dz, t, &g_sat);        \
            APPLY_SAT_LOWER_X(scale_x, g_sat);                                 \
        }                                                                      \
        if (phys_xu && i == nx - 1) {                                          \
            sat_boundary_data(src_type, p_local, FACE_UPPER_X,                 \
                              x0 + i * dx, y0 + j * dy, z0 + k * dz, t,        \
                              &g_sat);                                         \
            APPLY_SAT_UPPER_X(scale_x, g_sat);                                 \
        }                                                                      \
        if (phys_yl && j == 0) {                                               \
            sat_boundary_data(src_type, p_local, FACE_LOWER_Y,                 \
                              x0 + i * dx, y0, z0 + k * dz, t, &g_sat);        \
            APPLY_SAT_LOWER_Y(scale_y, g_sat);                                 \
        }                                                                      \
        if (phys_yu && j == ny - 1) {                                          \
            sat_boundary_data(src_type, p_local, FACE_UPPER_Y,                 \
                              x0 + i * dx, y0 + j * dy, z0 + k * dz, t,        \
                              &g_sat);                                         \
            APPLY_SAT_UPPER_Y(scale_y, g_sat);                                 \
        }                                                                      \
    } while (0)

        /* Region (A): k in z-shell, full (j, i) range. */
        for (int kA = 0; kA < 2; kA++)
        {
            const int64_t k_lo = (kA == 0) ? k_min     : k_int_max;
            const int64_t k_hi = (kA == 0) ? k_int_min : k_max;
            if (k_lo >= k_hi) continue;
            Kokkos::parallel_for("dot/shell-A",
                Range3D({i_min, j_min, k_lo}, {i_max, j_max, k_hi}),
                KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                    const stencil_t _sx = stencil_at(i, nx, phys_xl, phys_xu);
                    const stencil_t _sy = stencil_at(j, ny, phys_yl, phys_yu);
                    const stencil_t _sz = stencil_at(k, nz, phys_zl, phys_zu);
                    SHELL_POINT_BODY;
                });
        }

        /* Region (B): k z-interior, j in y-shell, full i. */
        for (int jB = 0; jB < 2; jB++)
        {
            const int64_t j_lo = (jB == 0) ? j_min     : j_int_max;
            const int64_t j_hi = (jB == 0) ? j_int_min : j_max;
            if (j_lo >= j_hi) continue;
            Kokkos::parallel_for("dot/shell-B",
                Range3D({i_min, j_lo, k_int_min},
                        {i_max, j_hi, k_int_max}),
                KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                    const stencil_t _sx = stencil_at(i, nx, phys_xl, phys_xu);
                    const stencil_t _sy = stencil_at(j, ny, phys_yl, phys_yu);
                    const stencil_t _sz = ST_D4CEN;
                    SHELL_POINT_BODY;
                });
        }

        /* Region (C): k z-interior, j y-interior, i in x-shell. */
        for (int iC = 0; iC < 2; iC++)
        {
            const int64_t i_lo = (iC == 0) ? i_min     : i_int_max;
            const int64_t i_hi = (iC == 0) ? i_int_min : i_max;
            if (i_lo >= i_hi) continue;
            Kokkos::parallel_for("dot/shell-C",
                Range3D({i_lo, j_int_min, k_int_min},
                        {i_hi, j_int_max, k_int_max}),
                KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                    const stencil_t _sx = stencil_at(i, nx, phys_xl, phys_xu);
                    const stencil_t _sy = ST_D4CEN;
                    const stencil_t _sz = ST_D4CEN;
                    SHELL_POINT_BODY;
                });
        }
#undef SHELL_POINT_BODY
        Kokkos::fence();
    }
    END_TIMER(timer_dot)

    if (maxwell_params.use_dissipation)
    {
        apply_dissipation(gfs, maxwell_params.diss_coeff, kidx);
    }
    sync_vars(gfs, EVOLVED, kidx);
}

/* ===========================================================================
 * CONSTRAINT DIAGNOSTICS (cD = div D - 4 pi rho,  cB = div B)
 * Same deep-interior + boundary-shell decomposition as the RHS.
 * =========================================================================== */
void maxwell_constraints(NGFS *gfs)
{
    DECLARE_EVOLVED_VARS(gfs, /*kidx unused*/0);
    DECLARE_AUX_VARS(gfs);

    const double four_pi = 12.566370614359172954;

    static int timer_const = -1;
    if (timer_const < 0) timer_const = register_timer("/Evol/constraints");

    const int phys_xl = gfs->domain.bbox.x.lower;
    const int phys_xu = gfs->domain.bbox.x.upper;
    const int phys_yl = gfs->domain.bbox.y.lower;
    const int phys_yu = gfs->domain.bbox.y.upper;
    const int phys_zl = gfs->domain.bbox.z.lower;
    const int phys_zu = gfs->domain.bbox.z.upper;

    const int64_t i_min = phys_xl ? 0 : D4CEN_HALF;
    const int64_t i_max = phys_xu ? nx : nx - D4CEN_HALF;
    const int64_t j_min = phys_yl ? 0 : D4CEN_HALF;
    const int64_t j_max = phys_yu ? ny : ny - D4CEN_HALF;
    const int64_t k_min = phys_zl ? 0 : D4CEN_HALF;
    const int64_t k_max = phys_zu ? nz : nz - D4CEN_HALF;

    const int64_t i_int_min = phys_xl ? SBP42_CLOSURE_ROWS : D4CEN_HALF;
    const int64_t i_int_max = phys_xu ? nx - SBP42_CLOSURE_ROWS : nx - D4CEN_HALF;
    const int64_t j_int_min = phys_yl ? SBP42_CLOSURE_ROWS : D4CEN_HALF;
    const int64_t j_int_max = phys_yu ? ny - SBP42_CLOSURE_ROWS : ny - D4CEN_HALF;
    const int64_t k_int_min = phys_zl ? SBP42_CLOSURE_ROWS : D4CEN_HALF;
    const int64_t k_int_max = phys_zu ? nz - SBP42_CLOSURE_ROWS : nz - D4CEN_HALF;

    using Range3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    BEGIN_TIMER(timer_const)
    {
        /* zero entire arrays (incl. ghost zones) */
        for (int slot : {(int)CD_SLOT, (int)CB_SLOT})
            Kokkos::deep_copy(gfs->aux[slot].state, 0.0);

#define CONSTRAINT_BODY                                                        \
    const double dx_Bx = DIFFX(Bx);                                            \
    const double dy_By = DIFFY(By);                                            \
    const double dz_Bz = DIFFZ(Bz);                                            \
    const double dx_Dx = DIFFX(Dx);                                            \
    const double dy_Dy = DIFFY(Dy);                                            \
    const double dz_Dz = DIFFZ(Dz);                                            \
    cD[ijk] = dx_Dx + dy_Dy + dz_Dz - four_pi * rho[ijk];                      \
    cB[ijk] = dx_Bx + dy_By + dz_Bz

#undef DIFFX
#undef DIFFY
#undef DIFFZ
#define DIFFX(f_) D4CEN(f_, ijk, di, dx)
#define DIFFY(f_) D4CEN(f_, ijk, dj, dy)
#define DIFFZ(f_) D4CEN(f_, ijk, dk, dz)
        Kokkos::parallel_for("constraints/interior",
            Range3D({i_int_min, j_int_min, k_int_min},
                    {i_int_max, j_int_max, k_int_max}),
            KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                const int64_t ijk = i + j * nx + k * nx * ny;
                CONSTRAINT_BODY;
            });

#undef DIFFX
#undef DIFFY
#undef DIFFZ
#define DIFFX(f_) apply_stencil(f_, ijk, _sx, di, dx)
#define DIFFY(f_) apply_stencil(f_, ijk, _sy, dj, dy)
#define DIFFZ(f_) apply_stencil(f_, ijk, _sz, dk, dz)

#define CONSTRAINT_SHELL_POINT                                                 \
    do {                                                                       \
        const int64_t ijk = i + j * nx + k * nx * ny;                          \
        CONSTRAINT_BODY;                                                       \
    } while (0)

        for (int kA = 0; kA < 2; kA++)
        {
            const int64_t k_lo = (kA == 0) ? k_min     : k_int_max;
            const int64_t k_hi = (kA == 0) ? k_int_min : k_max;
            if (k_lo >= k_hi) continue;
            Kokkos::parallel_for("constraints/shell-A",
                Range3D({i_min, j_min, k_lo}, {i_max, j_max, k_hi}),
                KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                    const stencil_t _sx = stencil_at(i, nx, phys_xl, phys_xu);
                    const stencil_t _sy = stencil_at(j, ny, phys_yl, phys_yu);
                    const stencil_t _sz = stencil_at(k, nz, phys_zl, phys_zu);
                    CONSTRAINT_SHELL_POINT;
                });
        }
        for (int jB = 0; jB < 2; jB++)
        {
            const int64_t j_lo = (jB == 0) ? j_min     : j_int_max;
            const int64_t j_hi = (jB == 0) ? j_int_min : j_max;
            if (j_lo >= j_hi) continue;
            Kokkos::parallel_for("constraints/shell-B",
                Range3D({i_min, j_lo, k_int_min},
                        {i_max, j_hi, k_int_max}),
                KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                    const stencil_t _sx = stencil_at(i, nx, phys_xl, phys_xu);
                    const stencil_t _sy = stencil_at(j, ny, phys_yl, phys_yu);
                    const stencil_t _sz = ST_D4CEN;
                    CONSTRAINT_SHELL_POINT;
                });
        }
        for (int iC = 0; iC < 2; iC++)
        {
            const int64_t i_lo = (iC == 0) ? i_min     : i_int_max;
            const int64_t i_hi = (iC == 0) ? i_int_min : i_max;
            if (i_lo >= i_hi) continue;
            Kokkos::parallel_for("constraints/shell-C",
                Range3D({i_lo, j_int_min, k_int_min},
                        {i_hi, j_int_max, k_int_max}),
                KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                    const stencil_t _sx = stencil_at(i, nx, phys_xl, phys_xu);
                    const stencil_t _sy = ST_D4CEN;
                    const stencil_t _sz = ST_D4CEN;
                    CONSTRAINT_SHELL_POINT;
                });
        }
#undef CONSTRAINT_SHELL_POINT
#undef CONSTRAINT_BODY
        Kokkos::fence();
    }
    END_TIMER(timer_const)
    sync_vars(gfs, AUX, /*kidx unused*/0);
}

/* ===========================================================================
 * INITIAL DATA
 * =========================================================================== */
void set_initial_data(NGFS *gfs, const double t)
{
    DECLARE_EVOLVED_VARS(gfs, /*kidx unused*/0);
    DECLARE_AUX_VARS(gfs);

    const int    src_type   = maxwell_params.source_type;
    const int    epsilon_type = maxwell_params.epsilon_type;
    const double sigma_bg   = maxwell_params.sigma;
    const double mu_bg      = maxwell_params.mu;
    const double eps_bg     = maxwell_params.epsilon;
    const material_elliptical_params ell = maxwell_params.elliptical;
    const analytic_params_st p_local = analytic_params;

    const double x0 = gfs->x0, y0 = gfs->y0, z0 = gfs->z0;

    using Range3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
    Kokkos::parallel_for("set_initial_data",
        Range3D({0, 0, 0}, {nx, ny, nz}),
        KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
            const double x = x0 + i * dx;
            const double y = y0 + j * dy;
            const double z = z0 + k * dz;
            const int64_t ijk = i + j * nx + k * nx * ny;
            eb_st A;
            source_state(src_type, p_local, x, y, z, t, &A);

            PsiB[ijk] = A.PsiB;
            PsiD[ijk] = A.PsiD;
            rho [ijk] = A.rho;
            Dx  [ijk] = A.Dx;
            Dy  [ijk] = A.Dy;
            Dz  [ijk] = A.Dz;
            Bx  [ijk] = A.Bx;
            By  [ijk] = A.By;
            Bz  [ijk] = A.Bz;

            sigma[ijk] = sigma_bg;
            imu  [ijk] = 1.0 / mu_bg;
            if (epsilon_type == 1)
            {
                const double rx = (x - ell.x0) / ell.a;
                const double rz = (z - ell.z0) / ell.b;
                const double s2 = ell.s * ell.s;
                const double r2 = (rx * rx + rz * rz) * s2;
                const double r4 = r2 * r2;
                const double eps =
                    (ell.max - eps_bg) * Kokkos::exp(-r4) + eps_bg;
                ieps[ijk] = 1.0 / eps;
            }
            else
            {
                ieps[ijk] = 1.0 / eps_bg;
            }
        });
    Kokkos::fence();
}

/* ===========================================================================
 * L2 ERROR DIAGNOSTIC (collective)
 * =========================================================================== */
double l2_error_analytic(NGFS *gfs, const double t)
{
    DECLARE_EVOLVED_VARS(gfs, /*kidx unused*/0);
    DECLARE_AUX_VARS(gfs);

    const int     src_type = maxwell_params.source_type;
    const analytic_params_st p_local = analytic_params;
    const double x0 = gfs->x0, y0 = gfs->y0, z0 = gfs->z0;

    /* Bounds match the C version: skip ghost rows on MPI/periodic axes,
     * include the boundary point on physical sides. */
    const int64_t i_start = gfs->domain.lower_x_rank >= 0 ? gs : 0;
    const int64_t i_end   = gfs->domain.upper_x_rank >= 0 ? nx - gs : nx;
    const int64_t j_start = gfs->domain.lower_y_rank >= 0 ? gs : 0;
    const int64_t j_end   = gfs->domain.upper_y_rank >= 0 ? ny - gs : ny;
    const int64_t k_start = gfs->domain.lower_z_rank >= 0 ? gs : 0;
    const int64_t k_end   = gfs->domain.upper_z_rank >= 0 ? nz - gs : nz;

    using Range3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
    double error = 0.0;
    Kokkos::parallel_reduce("l2_error_analytic",
        Range3D({i_start, j_start, k_start}, {i_end, j_end, k_end}),
        KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k, double &le) {
            const double x = x0 + i * dx;
            const double y = y0 + j * dy;
            const double z = z0 + k * dz;
            const int64_t ijk = i + j * nx + k * nx * ny;
            eb_st A;
            source_state(src_type, p_local, x, y, z, t, &A);
            const double e0 = PsiB[ijk] - A.PsiB;
            const double e1 = PsiD[ijk] - A.PsiD;
            const double e2 = rho [ijk] - A.rho;
            const double e3 = Dx  [ijk] - A.Dx;
            const double e4 = Dy  [ijk] - A.Dy;
            const double e5 = Dz  [ijk] - A.Dz;
            const double e6 = Bx  [ijk] - A.Bx;
            const double e7 = By  [ijk] - A.By;
            const double e8 = Bz  [ijk] - A.Bz;
            le += e0*e0 + e1*e1 + e2*e2 + e3*e3 + e4*e4
                + e5*e5 + e6*e6 + e7*e7 + e8*e8;
        }, error);

    const double local_npts = (double)(i_end - i_start) *
                              (double)(j_end - j_start) *
                              (double)(k_end - k_start);
    double local_data[2]  = {error, local_npts};
    double global_data[2] = {0.0,   0.0};
    MPI_Allreduce(local_data, global_data, 2, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    return std::sqrt(global_data[0] / global_data[1]);
}
