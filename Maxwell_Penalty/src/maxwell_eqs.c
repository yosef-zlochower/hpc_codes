#include "maxwell_eqs.h"
#include "gf.h"
#include "rk4.h"
#include "derivatives.h"
#include "numerical.h"
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "io.h"
#include "comm.h"
#include "domain.h"
#include "parameter.h"
#include "analytic_solutions.h"
#include "simple_maxwell.h"
#include "timer.h"

/* ===========================================================================
 * SBP-SAT BOUNDARY TREATMENT
 *
 * Every physical face is handled by:
 *   1) replacing the normal-direction derivative with the SBP-4-2 one-sided
 *      closure in the first/last 4 rows, and
 *   2) adding a SAT penalty at the exact boundary row,
 *        dot <- dot - tau * H_inv / h * A_pm * (u - g)
 *      where A_pm is the positive- (lower face) or negative- (upper face)
 *      eigenvalue projection of the axis flux matrix, and g is the prescribed
 *      incoming data. Only the lower-z face carries non-zero g (the incoming
 *      beam); the other five faces use g = 0 (absorbing).
 *
 * Flux decomposition (block structure; see doc/documentation.tex §4.4 and
 * the Summary appendix for the full derivation and sign conventions):
 *   z-axis blocks: (Bx, Dy) +/-c,  (By, Dx) +/-c,  (Bz, PsiB) +/-1,
 *                  (Dz, PsiD) +/-1,  rho 0
 *   x-axis blocks: (By, Dz),       (Bz, Dy),       (Bx, PsiB),
 *                  (Dx, PsiD)
 *   y-axis blocks: (Bz, Dx),       (Bx, Dz),       (By, PsiB),
 *                  (Dy, PsiD)
 *
 * SAT contributions from different faces are additive: at an edge two
 * penalties fire, at a corner three. No edge/corner-specific term is needed —
 * this additivity is the key feature of SBP-SAT in multiple dimensions.
 * =========================================================================== */

/* Slot-name tables.  Indexed by the enum constants in maxwell_eqs.h,
 * so a reordering of the enum flows through to these without any manual
 * re-alignment. */
const char *const evolved_field_names[N_EVOL] = {
    [DX_SLOT]   = "Dx",
    [DY_SLOT]   = "Dy",
    [DZ_SLOT]   = "Dz",
    [BX_SLOT]   = "Bx",
    [BY_SLOT]   = "By",
    [BZ_SLOT]   = "Bz",
    [PSID_SLOT] = "PsiD",
    [PSIB_SLOT] = "PsiB",
    [RHO_SLOT]  = "rho",
};
const char *const aux_field_names[N_AUX] = {
    [IEPS_SLOT]  = "ieps",
    [IMU_SLOT]   = "imu",
    [SIGMA_SLOT] = "sigma",
    [CD_SLOT]    = "cD",
    [CB_SLOT]    = "cB",
};

/* Runtime stencil tag used by the boundary-shell sweep. The deep-interior
 * loop uses the D4CEN macro directly (no dispatch) for speed. */
typedef enum
{
    ST_D4CEN = 0,
    ST_SBP_L0,
    ST_SBP_L1,
    ST_SBP_L2,
    ST_SBP_L3,
    ST_SBP_R0,
    ST_SBP_R1,
    ST_SBP_R2,
    ST_SBP_R3
} stencil_t;

static inline stencil_t stencil_at(int64_t i, int64_t n,
                                   int phys_lo, int phys_hi)
{
    if (phys_lo && i < SBP42_CLOSURE_ROWS)
    {
        return (stencil_t)(ST_SBP_L0 + i);
    }
    if (phys_hi && i >= n - SBP42_CLOSURE_ROWS)
    {
        return (stencil_t)(ST_SBP_R0 + (n - 1 - i));
    }
    return ST_D4CEN;
}

static inline double apply_stencil(const double *f, int64_t ijk, stencil_t s,
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

/* Source dispatch: which analytic state to use for initial data and the
 * L2 error diagnostic. For source_type == 2 (te_waveguide_mode) this is also
 * the boundary data on every physical face; for the other source types
 * only the lower-z face uses nonzero data (see sat_boundary_data below). */
static inline void source_state(const double x, const double y,
                                const double z, const double t,
                                struct eb_st *A)
{
    switch (maxwell_params.source_type)
    {
        case 1:
            incoming_gaussian_beam(analytic_params, x, y, z, t, A);
            break;
        case 2:
            te_waveguide_mode(analytic_params, x, y, z, t, A);
            break;
        default:
            incoming_plane_wave(analytic_params, x, y, z, t, A);
            break;
    }
}

/* Face tag for boundary-data dispatch. */
typedef enum
{
    FACE_LOWER_X, FACE_UPPER_X,
    FACE_LOWER_Y, FACE_UPPER_Y,
    FACE_LOWER_Z, FACE_UPPER_Z,
} sat_face_t;

/* Boundary data g(x, y, z, t) prescribed by the SAT at a given face.
 *   - plane_wave / gaussian_beam: g = source at lower-z only, zero elsewhere
 *     (physical beam enters through the bottom; other five faces absorb).
 *   - te_waveguide_mode: g = analytic waveguide mode on every physical face
 *     (convergence-test mode; SBP-SAT now imposes the analytic exactly). */
static inline void sat_boundary_data(const sat_face_t face,
                                     const double x, const double y,
                                     const double z, const double t,
                                     struct eb_st *A)
{
    /* Start from zero (all unspecified components default to 0). */
    A->Dx = A->Dy = A->Dz = 0.0;
    A->Bx = A->By = A->Bz = 0.0;
    A->PsiD = A->PsiB = 0.0;
    A->rho = 0.0;

    if (maxwell_params.source_type == 2)
    {
        /* te_waveguide_mode: analytic on every face */
        te_waveguide_mode(analytic_params, x, y, z, t, A);
        return;
    }

    /* plane_wave or gaussian_beam: nonzero only on lower-z */
    if (face == FACE_LOWER_Z)
    {
        source_state(x, y, z, t, A);
    }
}

/* ----------------------------------------------------------------------------
 * SAT penalty blocks
 *
 * Six macros APPLY_SAT_{LOWER,UPPER}_{X,Y,Z}(scale_, g_) subtract
 *   scale * A_face * (u - g)
 * from the dot arrays at ijk, where A_face is the characteristic projection
 * onto the incoming modes at the named face. They are machine-generated from
 * the PDE by generate_ccode.py (via SymPy diagonalisation of the principal-
 * part flux matrix) and live in simple_maxwell.h, which we have already
 * included. See doc/documentation.tex §4.4 and the SAT-formulas appendix
 * for the characteristic analysis.
 *
 * The macros assume DECLARE_EVOLVED_VARS / DECLARE_AUX_VARS are in scope so
 * that Bx[], dotBx[], ieps[], etc. resolve. Only lower-z has non-zero g in
 * the current sources; the other five faces pass g = 0, which is the
 * absorbing condition for waves leaving the domain.
 *
 * Sanity check (x-polarised +z traveling wave  Dx = By = psi,  Bx = Dy = 0):
 *   (By, Dx) block:  w+ = By + PP*Dx = (1+PP)*psi   — incoming, nonzero ✓
 *   (Bx, Dy) block:  w+ = Bx - PP*Dy = 0            — this mode not excited ✓
 * A sign flip on any coupling term would make w+ = 0 for a real incoming
 * wave and the SAT would silently fail to inject it.
 * -------------------------------------------------------------------------- */

/* ===========================================================================
 * TIME-DERIVATIVE FUNCTION
 * Purpose: Calculate time derivative of the state vector. Store results in
 * var[*]->dot.  var[*]->dot is a pointer to k1, k2, k3, or k4. The actual
 * array pointed to is set by RK4_Step.
 * =========================================================================== */

/* Compute the PDE right-hand side and write it into the dot arrays.
 *
 * NOTE on dot-pointer semantics: `dotBx, dotDy, ..., dotrho` come from
 * DECLARE_EVOLVED_VARS and alias `gfs->vars[slot]->dot`.  That pointer is
 * rewired by RK4_Step between stages to point at K1, K2, K3, or K4 in
 * turn (see rk4.c).  So "writing into dotBx" writes into whichever RK
 * stage buffer the caller has selected — there is no separate K_i
 * parameter here, by design.  This function is therefore only safe to
 * call from RK4_Step (or another integrator that sets up the same
 * pointer convention). */
void maxwell_eq_time_deriv(struct ngfs *gfs, const double t)
{
    DECLARE_EVOLVED_VARS(gfs);
    DECLARE_AUX_VARS(gfs);

    const double four_pi = 12.566370614359172954;
    const double kappa_B = maxwell_params.kappa_B;
    const double kappa_D = maxwell_params.kappa_D;
    const double tau     = maxwell_params.tau;

    static int timer_dot = -1;
    if (timer_dot < 0)
    {
        timer_dot = register_timer("/Evol/dot");
    }

    const int phys_xl = gfs->domain.bbox.x.lower;
    const int phys_xu = gfs->domain.bbox.x.upper;
    const int phys_yl = gfs->domain.bbox.y.lower;
    const int phys_yu = gfs->domain.bbox.y.upper;
    const int phys_zl = gfs->domain.bbox.z.lower;
    const int phys_zu = gfs->domain.bbox.z.upper;

    /* Computed-point range: excludes ghost zones on MPI/periodic sides, but
     * includes the boundary point on physical sides.  The D4CEN_HALF offset
     * on an MPI side leaves room for D4CEN to reach into the ghost layer. */
    const int64_t i_min = phys_xl ? 0 : D4CEN_HALF;
    const int64_t i_max = phys_xu ? nx : nx - D4CEN_HALF;
    const int64_t j_min = phys_yl ? 0 : D4CEN_HALF;
    const int64_t j_max = phys_yu ? ny : ny - D4CEN_HALF;
    const int64_t k_min = phys_zl ? 0 : D4CEN_HALF;
    const int64_t k_max = phys_zu ? nz : nz - D4CEN_HALF;

    /* Deep-interior range: all three axes safely away from physical SBP rows.
     * On MPI/periodic sides D4CEN works from index D4CEN_HALF (ghost zones
     * supply the +/-half stencil points).  On physical sides D4CEN works
     * from index SBP42_CLOSURE_ROWS, skipping the one-sided closure rows. */
    const int64_t i_int_min = phys_xl ? SBP42_CLOSURE_ROWS : D4CEN_HALF;
    const int64_t i_int_max = phys_xu ? nx - SBP42_CLOSURE_ROWS : nx - D4CEN_HALF;
    const int64_t j_int_min = phys_yl ? SBP42_CLOSURE_ROWS : D4CEN_HALF;
    const int64_t j_int_max = phys_yu ? ny - SBP42_CLOSURE_ROWS : ny - D4CEN_HALF;
    const int64_t k_int_min = phys_zl ? SBP42_CLOSURE_ROWS : D4CEN_HALF;
    const int64_t k_int_max = phys_zu ? nz - SBP42_CLOSURE_ROWS : nz - D4CEN_HALF;

    const double scale_x = tau * SBP42_HINV_0 / dx;
    const double scale_y = tau * SBP42_HINV_0 / dy;
    const double scale_z = tau * SBP42_HINV_0 / dz;

    BEGIN_TIMER(timer_dot)
    {
        /* ---------------- DEEP INTERIOR (fast path, D4CEN everywhere) ----- */
#undef DIFFX
#undef DIFFY
#undef DIFFZ
#define DIFFX(f_) D4CEN(f_, ijk, di, dx)
#define DIFFY(f_) D4CEN(f_, ijk, dj, dy)
#define DIFFZ(f_) D4CEN(f_, ijk, dk, dz)
        for (int64_t k = k_int_min; k < k_int_max; k++)
        {
            for (int64_t j = j_int_min; j < j_int_max; j++)
            {
                for (int64_t i = i_int_min; i < i_int_max; i++)
                {
                    const int64_t ijk = ijk_indx(i, j, k, gfs);
                    SIMPLE_MAXWELL_INTERIOR_DOT;
                }
            }
        }

        /* ---------------- BOUNDARY SHELL (three non-overlapping regions) --
         * The shell = "at least one of _sx, _sy, _sz is not ST_D4CEN".  We
         * sweep it as three nested regions that partition the shell
         * exactly once:
         *   (A)  z-shell rows  × full (j, i) range.
         *   (B)  z-interior    × y-shell rows × full i range.
         *   (C)  z-interior    × y-interior   × x-shell rows.
         * In (B) the z-stencil is known to be ST_D4CEN; in (C) both z- and
         * y-stencils are ST_D4CEN; we record that so apply_stencil takes
         * its D4CEN case directly.
         *
         * SAT penalties are additive across faces; up to three fire at a
         * corner point.  Since each point is visited exactly once, the
         * per-face checks  if (phys_* && i/j/k == face_row)  each fire at
         * most once per RK stage, as required.
         *
         * Each of the k_shell / k_int sub-ranges degenerates to empty when
         * the corresponding physical-boundary flag is false, so no extra
         * guards are needed. */
#undef DIFFX
#undef DIFFY
#undef DIFFZ
#define DIFFX(f_) apply_stencil(f_, ijk, _sx, di, dx)
#define DIFFY(f_) apply_stencil(f_, ijk, _sy, dj, dy)
#define DIFFZ(f_) apply_stencil(f_, ijk, _sz, dk, dz)

/* Per-point body of the shell sweep: assumes _sx, _sy, _sz are defined
 * in the enclosing scope.  Computes the PDE RHS, then adds each
 * applicable SAT penalty (at most three at a corner). */
#define SHELL_POINT_BODY                                                       \
    do {                                                                       \
        const int64_t ijk = ijk_indx(i, j, k, gfs);                            \
        SIMPLE_MAXWELL_INTERIOR_DOT;                                           \
                                                                               \
        struct eb_st g_sat;                                                    \
        if (phys_zl && k == 0) {                                               \
            sat_boundary_data(FACE_LOWER_Z, gfs->x0 + i * dx,                  \
                              gfs->y0 + j * dy, gfs->z0, t, &g_sat);           \
            APPLY_SAT_LOWER_Z(scale_z, g_sat);                                 \
        }                                                                      \
        if (phys_zu && k == nz - 1) {                                          \
            sat_boundary_data(FACE_UPPER_Z, gfs->x0 + i * dx,                  \
                              gfs->y0 + j * dy, gfs->z0 + k * dz, t, &g_sat);  \
            APPLY_SAT_UPPER_Z(scale_z, g_sat);                                 \
        }                                                                      \
        if (phys_xl && i == 0) {                                               \
            sat_boundary_data(FACE_LOWER_X, gfs->x0,                           \
                              gfs->y0 + j * dy, gfs->z0 + k * dz, t, &g_sat);  \
            APPLY_SAT_LOWER_X(scale_x, g_sat);                                 \
        }                                                                      \
        if (phys_xu && i == nx - 1) {                                          \
            sat_boundary_data(FACE_UPPER_X, gfs->x0 + i * dx,                  \
                              gfs->y0 + j * dy, gfs->z0 + k * dz, t, &g_sat);  \
            APPLY_SAT_UPPER_X(scale_x, g_sat);                                 \
        }                                                                      \
        if (phys_yl && j == 0) {                                               \
            sat_boundary_data(FACE_LOWER_Y, gfs->x0 + i * dx,                  \
                              gfs->y0, gfs->z0 + k * dz, t, &g_sat);           \
            APPLY_SAT_LOWER_Y(scale_y, g_sat);                                 \
        }                                                                      \
        if (phys_yu && j == ny - 1) {                                          \
            sat_boundary_data(FACE_UPPER_Y, gfs->x0 + i * dx,                  \
                              gfs->y0 + j * dy, gfs->z0 + k * dz, t, &g_sat);  \
            APPLY_SAT_UPPER_Y(scale_y, g_sat);                                 \
        }                                                                      \
    } while (0)

        /* Region (A): k is a z-shell row (lower or upper physical face).
         * j and i run over the whole computed range; all three stencils
         * need runtime dispatch. */
        for (int64_t kA = 0; kA < 2; kA++) {
            const int64_t k_lo = (kA == 0) ? k_min     : k_int_max;
            const int64_t k_hi = (kA == 0) ? k_int_min : k_max;
            for (int64_t k = k_lo; k < k_hi; k++) {
                const stencil_t _sz = stencil_at(k, nz, phys_zl, phys_zu);
                for (int64_t j = j_min; j < j_max; j++) {
                    const stencil_t _sy = stencil_at(j, ny, phys_yl, phys_yu);
                    for (int64_t i = i_min; i < i_max; i++) {
                        const stencil_t _sx =
                            stencil_at(i, nx, phys_xl, phys_xu);
                        SHELL_POINT_BODY;
                    }
                }
            }
        }

        /* Region (B): k z-interior (_sz known ST_D4CEN), j in y-shell.
         * Region (C): k z-interior, j y-interior (_sy known ST_D4CEN),
         *             i in x-shell. */
        for (int64_t k = k_int_min; k < k_int_max; k++) {
            const stencil_t _sz = ST_D4CEN;

            /* (B) — lower and upper y-face slabs */
            for (int64_t jB = 0; jB < 2; jB++) {
                const int64_t j_lo = (jB == 0) ? j_min     : j_int_max;
                const int64_t j_hi = (jB == 0) ? j_int_min : j_max;
                for (int64_t j = j_lo; j < j_hi; j++) {
                    const stencil_t _sy =
                        stencil_at(j, ny, phys_yl, phys_yu);
                    for (int64_t i = i_min; i < i_max; i++) {
                        const stencil_t _sx =
                            stencil_at(i, nx, phys_xl, phys_xu);
                        SHELL_POINT_BODY;
                    }
                }
            }

            /* (C) — left and right x-face slabs, within y-interior */
            for (int64_t j = j_int_min; j < j_int_max; j++) {
                const stencil_t _sy = ST_D4CEN;
                for (int64_t iC = 0; iC < 2; iC++) {
                    const int64_t i_lo = (iC == 0) ? i_min     : i_int_max;
                    const int64_t i_hi = (iC == 0) ? i_int_min : i_max;
                    for (int64_t i = i_lo; i < i_hi; i++) {
                        const stencil_t _sx =
                            stencil_at(i, nx, phys_xl, phys_xu);
                        SHELL_POINT_BODY;
                    }
                }
            }
        }
#undef SHELL_POINT_BODY
    }
    END_TIMER(timer_dot)

    if (maxwell_params.use_dissipation)
    {
        /* Note that apply_dissipation modifies *->dot arrays. Must be called after *->dot
         * is filled with standard RHS values.
         */
        apply_dissipation(gfs, maxwell_params.diss_coeff);
    }
    sync_vars(gfs, EVOLVED);
}

/* ===========================================================================
 * CONSTRAINT DIAGNOSTICS (cD = div D - 4 pi rho,  cB = div B)
 * Same deep-interior + boundary-shell decomposition as the RHS.
 * =========================================================================== */

void maxwell_constraints(struct ngfs *gfs)
{
    DECLARE_EVOLVED_VARS(gfs);
    DECLARE_AUX_VARS(gfs);

    const double four_pi = 12.566370614359172954;

    static int timer_const = -1;
    if (timer_const < 0)
    {
        timer_const = register_timer("/Evol/constraints");
    }

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

#define CONSTRAINT_BODY                                                        \
    const double dx_Bx = DIFFX(Bx);                                            \
    const double dy_By = DIFFY(By);                                            \
    const double dz_Bz = DIFFZ(Bz);                                            \
    const double dx_Dx = DIFFX(Dx);                                            \
    const double dy_Dy = DIFFY(Dy);                                            \
    const double dz_Dz = DIFFZ(Dz);                                            \
    cD[ijk] = dx_Dx + dy_Dy + dz_Dz - four_pi * rho[ijk];                      \
    cB[ijk] = dx_Bx + dy_By + dz_Bz;

    BEGIN_TIMER(timer_const)
    {
        for (int64_t all = 0; all < gfs->n_tot; all++)
        {
            cD[all] = 0;
            cB[all] = 0;
        }

#undef DIFFX
#undef DIFFY
#undef DIFFZ
#define DIFFX(f_) D4CEN(f_, ijk, di, dx)
#define DIFFY(f_) D4CEN(f_, ijk, dj, dy)
#define DIFFZ(f_) D4CEN(f_, ijk, dk, dz)
        for (int64_t k = k_int_min; k < k_int_max; k++)
            for (int64_t j = j_int_min; j < j_int_max; j++)
                for (int64_t i = i_int_min; i < i_int_max; i++)
                {
                    const int64_t ijk = ijk_indx(i, j, k, gfs);
                    CONSTRAINT_BODY
                }

#undef DIFFX
#undef DIFFY
#undef DIFFZ
#define DIFFX(f_) apply_stencil(f_, ijk, _sx, di, dx)
#define DIFFY(f_) apply_stencil(f_, ijk, _sy, dj, dy)
#define DIFFZ(f_) apply_stencil(f_, ijk, _sz, dk, dz)

/* Per-point body of the constraint shell sweep: _sx, _sy, _sz in scope. */
#define CONSTRAINT_SHELL_POINT                     \
    do {                                           \
        const int64_t ijk = ijk_indx(i, j, k, gfs);\
        CONSTRAINT_BODY                            \
    } while (0)

        /* Same three-region partition as in maxwell_eq_time_deriv. */

        /* Region (A): k in z-shell; full j, i ranges. */
        for (int64_t kA = 0; kA < 2; kA++) {
            const int64_t k_lo = (kA == 0) ? k_min     : k_int_max;
            const int64_t k_hi = (kA == 0) ? k_int_min : k_max;
            for (int64_t k = k_lo; k < k_hi; k++) {
                const stencil_t _sz = stencil_at(k, nz, phys_zl, phys_zu);
                for (int64_t j = j_min; j < j_max; j++) {
                    const stencil_t _sy = stencil_at(j, ny, phys_yl, phys_yu);
                    for (int64_t i = i_min; i < i_max; i++) {
                        const stencil_t _sx =
                            stencil_at(i, nx, phys_xl, phys_xu);
                        CONSTRAINT_SHELL_POINT;
                    }
                }
            }
        }

        /* Regions (B) and (C): k z-interior. */
        for (int64_t k = k_int_min; k < k_int_max; k++) {
            const stencil_t _sz = ST_D4CEN;

            /* (B) — j in y-shell; full i range. */
            for (int64_t jB = 0; jB < 2; jB++) {
                const int64_t j_lo = (jB == 0) ? j_min     : j_int_max;
                const int64_t j_hi = (jB == 0) ? j_int_min : j_max;
                for (int64_t j = j_lo; j < j_hi; j++) {
                    const stencil_t _sy =
                        stencil_at(j, ny, phys_yl, phys_yu);
                    for (int64_t i = i_min; i < i_max; i++) {
                        const stencil_t _sx =
                            stencil_at(i, nx, phys_xl, phys_xu);
                        CONSTRAINT_SHELL_POINT;
                    }
                }
            }

            /* (C) — j y-interior, i in x-shell. */
            for (int64_t j = j_int_min; j < j_int_max; j++) {
                const stencil_t _sy = ST_D4CEN;
                for (int64_t iC = 0; iC < 2; iC++) {
                    const int64_t i_lo = (iC == 0) ? i_min     : i_int_max;
                    const int64_t i_hi = (iC == 0) ? i_int_min : i_max;
                    for (int64_t i = i_lo; i < i_hi; i++) {
                        const stencil_t _sx =
                            stencil_at(i, nx, phys_xl, phys_xu);
                        CONSTRAINT_SHELL_POINT;
                    }
                }
            }
        }
#undef CONSTRAINT_SHELL_POINT
    }
    END_TIMER(timer_const)
#undef CONSTRAINT_BODY
    sync_vars(gfs, AUX);
}

/* ===========================================================================
 * INITIAL DATA
 * Fills the full local grid from the selected analytic source. For the
 * gaussian_beam source this seeds the domain with the paraxial beam at t = 0;
 * the SAT boundary then keeps injecting it at z = 0 so the solution stays
 * steady-state from step 1 up to the paraxial residual.
 * =========================================================================== */

void set_initial_data(struct ngfs *gfs, const double t)
{
    DECLARE_EVOLVED_VARS(gfs);
    DECLARE_AUX_VARS(gfs);

    for (int64_t k = 0; k < nz; k++)
    {
        for (int64_t j = 0; j < ny; j++)
        {
            for (int64_t i = 0; i < nx; i++)
            {
                const double x = gfs->x0 + i * dx;
                const double y = gfs->y0 + j * dy;
                const double z = gfs->z0 + k * dz;
                const int64_t ijk = ijk_indx(i, j, k, gfs);
                struct eb_st A;

                source_state(x, y, z, t, &A);

                PsiB[ijk] = A.PsiB;
                PsiD[ijk] = A.PsiD;
                rho[ijk]  = A.rho;

                Dx[ijk] = A.Dx;
                Dy[ijk] = A.Dy;
                Dz[ijk] = A.Dz;

                Bx[ijk] = A.Bx;
                By[ijk] = A.By;
                Bz[ijk] = A.Bz;

                sigma[ijk] = maxwell_params.sigma;
                imu[ijk]   = 1.0 / maxwell_params.mu;

                if (maxwell_params.epsilon_type == 1)
                {
                    const struct material_elliptical_params ep =
                        maxwell_params.elliptical;
                    const double eps_bg = maxwell_params.epsilon;
                    const double rx = (x - ep.x0) / ep.a;
                    const double rz = (z - ep.z0) / ep.b;
                    const double s  = ep.s;
                    const double r2 = (rx * rx + rz * rz) * s * s;
                    const double r4 = r2 * r2;
                    /* Profile decays to the background eps_bg far from
                     * the lens and peaks at ep.max at the centre. */
                    const double eps =
                        (ep.max - eps_bg) * exp(-r4) + eps_bg;
                    ieps[ijk] = 1.0 / eps;
                }
                else
                {
                    ieps[ijk] = 1.0 / maxwell_params.epsilon;
                }
            }
        }
    }
}

/* ===========================================================================
 * L2 ERROR DIAGNOSTIC
 * Compares the numerical state against the analytic source evaluated at the
 * current time. For source_type == "plane_wave" with periodic x,y this is an
 * exact convergence measure. For "gaussian_beam" the paraxial source is only
 * an approximate solution (and the lens refracts it); this returns a
 * diagnostic that is meaningful only before the beam reaches the lens or
 * upper-z face.
 * =========================================================================== */

double l2_error_analytic(struct ngfs *gfs, const double t)
{
    DECLARE_EVOLVED_VARS(gfs);
    DECLARE_AUX_VARS(gfs);

    double error = 0;

#define L2_ADD_TO_ERROR(_exp)                                                  \
    {                                                                          \
        const double le = _exp;                                                \
        error += le * le;                                                      \
    }

    const int64_t i_start = gfs->domain.lower_x_rank >= 0 ? gs : 0;
    const int64_t i_end   = gfs->domain.upper_x_rank >= 0 ? nx - gs : nx;
    const int64_t j_start = gfs->domain.lower_y_rank >= 0 ? gs : 0;
    const int64_t j_end   = gfs->domain.upper_y_rank >= 0 ? ny - gs : ny;
    const int64_t k_start = gfs->domain.lower_z_rank >= 0 ? gs : 0;
    const int64_t k_end   = gfs->domain.upper_z_rank >= 0 ? nz - gs : nz;

    for (int64_t k = k_start; k < k_end; k++)
    {
        for (int64_t j = j_start; j < j_end; j++)
        {
            for (int64_t i = i_start; i < i_end; i++)
            {
                struct eb_st A;

                const double x = gfs->x0 + i * dx;
                const double y = gfs->y0 + j * dy;
                const double z = gfs->z0 + k * dz;

                source_state(x, y, z, t, &A);
                const int64_t ijk = ijk_indx(i, j, k, gfs);

                L2_ADD_TO_ERROR(PsiB[ijk] - A.PsiB);
                L2_ADD_TO_ERROR(PsiD[ijk] - A.PsiD);
                L2_ADD_TO_ERROR(rho[ijk]  - A.rho);

                L2_ADD_TO_ERROR(Dx[ijk] - A.Dx);
                L2_ADD_TO_ERROR(Dy[ijk] - A.Dy);
                L2_ADD_TO_ERROR(Dz[ijk] - A.Dz);

                L2_ADD_TO_ERROR(Bx[ijk] - A.Bx);
                L2_ADD_TO_ERROR(By[ijk] - A.By);
                L2_ADD_TO_ERROR(Bz[ijk] - A.Bz);
            }
        }
    }
    const double local_npts = (double)(i_end - i_start) *
                              (double)(j_end - j_start) *
                              (double)(k_end - k_start);
    double local_data[2] = {error, local_npts};
    double global_data[2];
    MPI_Allreduce(local_data, global_data, 2, MPI_DOUBLE, MPI_SUM,
               MPI_COMM_WORLD);
    return sqrt(global_data[0] / global_data[1]);
}
