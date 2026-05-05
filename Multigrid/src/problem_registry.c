/*
 * Built-in problem presets for the multigrid solver.
 *
 * Each preset bundles the BC spec, RHS, and (optionally) an exact
 * solution into a single problem_t record.  Adding a new problem is
 * a one-file edit: define the pure functions, append the record to
 * g_problems[].  The driver and the test harness consume these via
 * problem_lookup(name).
 *
 * Phase 1 ships only `manufactured_dirichlet_homog`, which reproduces
 * the original hard-coded behaviour of driver_multigrid.c bit-for-bit.
 * Phases 2-4 will register additional presets covering inhomogeneous
 * Dirichlet, homogeneous and inhomogeneous Neumann, and mixed BCs.
 */
#include "problem.h"
#include <math.h>
#include <stddef.h>
#include <string.h>

/* ============================================================
 * manufactured_dirichlet_homog
 *
 *   u*(x,y,z) = sin(pi x) sin(pi y) sin(pi z)
 *   f (x,y,z) = -3 pi^2 u*(x,y,z)
 *   homogeneous Dirichlet on all six faces (u* vanishes on every face).
 *
 * This is the original test problem hard-coded in driver_multigrid.c
 * before the §6.1 refactor.  Kept as the default preset so absent
 * `[problem]` sections in TOML files reproduce today's behaviour.
 * ============================================================ */

static double sin3_rhs(double x, double y, double z)
{
    const double pi  = 3.14159265358979323846;
    const double pi2 = pi * pi;
    return -3.0 * pi2 * sin(pi * x) * sin(pi * y) * sin(pi * z);
}

static double sin3_exact(double x, double y, double z)
{
    const double pi = 3.14159265358979323846;
    return sin(pi * x) * sin(pi * y) * sin(pi * z);
}

/* ============================================================
 * manufactured_dirichlet_inhomog
 *
 *   u*(x,y,z) = cos(pi x) cos(pi y) cos(pi z)
 *   f (x,y,z) = -3 pi^2 u*(x,y,z)
 *   inhomogeneous Dirichlet on every face: u* is non-zero on the
 *   x=0, x=1, y=0, y=1, z=0, z=1 planes generically.
 * ============================================================ */

static double cos3_rhs(double x, double y, double z)
{
    const double pi  = 3.14159265358979323846;
    const double pi2 = pi * pi;
    return -3.0 * pi2 * cos(pi * x) * cos(pi * y) * cos(pi * z);
}

static double cos3_exact(double x, double y, double z)
{
    const double pi = 3.14159265358979323846;
    return cos(pi * x) * cos(pi * y) * cos(pi * z);
}

/* Per-face Dirichlet value: u*(x,y,z) evaluated at the face point.
 * The face argument is unused here (the function is evaluated on the
 * face, so the appropriate coordinate is already pinned to 0 or 1),
 * but the bc_fn_t signature allows it for callers that need to
 * dispatch by face. */
static double cos3_dirichlet(double x, double y, double z, face_id_t face)
{
    (void)face;
    return cos3_exact(x, y, z);
}

/* ============================================================
 * manufactured_neumann_homog
 *
 *   u_star(x,y,z) = cos(pi x) cos(pi y) cos(pi z)
 *   f      (x,y,z) = -3 pi^2 u_star(x,y,z)
 *   homogeneous Neumann on all six faces:
 *     d u_star / d x = -pi sin(pi x) cos(pi y) cos(pi z)
 *     vanishes at x=0,1 (and symmetrically for y, z),
 *   so the outward normal derivative is 0 on every face.
 *
 * This is the singular case: with all Neumann faces the operator has
 * the constant function in its null space.  problem_initialise_rhs
 * subtracts the discrete mean of f at startup, and the driver applies
 * mean-zero projection to u after each V-cycle iteration.  Because
 * the integral of cos^3 over [0,1]^3 is zero in the continuous
 * problem, the manufactured solution itself is mean-zero; the
 * projection is essentially a round-off guard.
 * ============================================================ */

/* The cos^3 RHS and exact solution were already defined above for the
 * inhomogeneous-Dirichlet preset; we reuse them here. */

/* ============================================================
 * manufactured_neumann_inhomog
 *
 *   u_star(x,y,z) = sin(pi x/2) sin(pi y/2) sin(pi z/2)
 *   f      (x,y,z) = -3 (pi/2)^2 u_star(x,y,z)
 *
 * Three faces (x=0, y=0, z=0) carry inhomogeneous Neumann data;
 * the other three (x=1, y=1, z=1) carry naturally-zero normal
 * derivatives (cos(pi/2) = 0) so they reduce to homogeneous
 * Neumann.  The compatibility condition int f = int q evaluates
 * to -6/pi on both sides, so the singular system is solvable.
 * Singular case: mean-zero projection is required.
 * ============================================================ */

static double sin3_half_rhs(double x, double y, double z)
{
    const double pi   = 3.14159265358979323846;
    const double pi_h = pi / 2.0;
    return -3.0 * pi_h * pi_h
         * sin(pi_h * x) * sin(pi_h * y) * sin(pi_h * z);
}

static double sin3_half_exact(double x, double y, double z)
{
    const double pi_h = 3.14159265358979323846 / 2.0;
    return sin(pi_h * x) * sin(pi_h * y) * sin(pi_h * z);
}

/* Outward-normal derivative of u_star at the point (x,y,z), dispatched
 * by face.  Used uniformly on all six faces; returns 0 on the upper
 * faces where cos(pi/2) = 0, which is consistent with the homogeneous
 * Neumann condition there. */
static double sin3_half_neumann(double x, double y, double z, face_id_t face)
{
    const double pi_h = 3.14159265358979323846 / 2.0;
    const double sx = sin(pi_h * x), cx_ = cos(pi_h * x);
    const double sy = sin(pi_h * y), cy_ = cos(pi_h * y);
    const double sz = sin(pi_h * z), cz_ = cos(pi_h * z);
    const double dudx = pi_h * cx_ * sy  * sz;
    const double dudy = pi_h * sx  * cy_ * sz;
    const double dudz = pi_h * sx  * sy  * cz_;
    switch (face) {
        case FACE_LOWER_X: return -dudx;  /* outward normal -x */
        case FACE_UPPER_X: return  dudx;  /* outward normal +x */
        case FACE_LOWER_Y: return -dudy;
        case FACE_UPPER_Y: return  dudy;
        case FACE_LOWER_Z: return -dudz;
        case FACE_UPPER_Z: return  dudz;
        default:           return 0.0;
    }
}

/* ============================================================
 * manufactured_mixed
 *
 *   u_star(x,y,z) = x^2 cos(pi y) cos(pi z)
 *   f      (x,y,z) = (2 - 2 pi^2 x^2) cos(pi y) cos(pi z)
 *
 * Boundary conditions:
 *   FACE_LOWER_X (x=0): u_star = 0       -> homog Dirichlet
 *   FACE_UPPER_X (x=1): du/dx = 2 cos(pi y) cos(pi z) -> inhomog Neumann
 *   FACE_LOWER_Y (y=0): du/dy = 0        -> homog Neumann (sin 0 = 0)
 *   FACE_UPPER_Y (y=1): du/dy = 0        -> homog Neumann (sin pi = 0)
 *   FACE_LOWER_Z (z=0): du/dz = 0        -> homog Neumann
 *   FACE_UPPER_Z (z=1): du/dz = 0        -> homog Neumann
 *
 * Non-singular: the lone Dirichlet face fixes the constant mode.
 * The trig factors give a non-trivial truncation error so the rate
 * test measures real second-order convergence (unlike a low-degree
 * polynomial u*, which the 7-point stencil can represent exactly).
 * ============================================================ */

static double mixed_rhs(double x, double y, double z)
{
    const double pi  = 3.14159265358979323846;
    const double pi2 = pi * pi;
    return (2.0 - 2.0 * pi2 * x * x) * cos(pi * y) * cos(pi * z);
}

static double mixed_exact(double x, double y, double z)
{
    const double pi = 3.14159265358979323846;
    return x * x * cos(pi * y) * cos(pi * z);
}

/* Outward-normal derivative at each face.  Lower-x is Dirichlet so
 * never sees this callback; the Neumann faces evaluate du/dn at the
 * face point.  Returns the analytic value computed from u_star. */
static double mixed_neumann(double x, double y, double z, face_id_t face)
{
    const double pi = 3.14159265358979323846;
    /* Partials of u_star = x^2 cos(pi y) cos(pi z): */
    const double dudx =  2.0 * x * cos(pi * y) * cos(pi * z);
    const double dudy = -pi  * x * x * sin(pi * y) * cos(pi * z);
    const double dudz = -pi  * x * x * cos(pi * y) * sin(pi * z);
    switch (face) {
        case FACE_UPPER_X: return  dudx;  /* outward normal +x */
        case FACE_LOWER_Y: return -dudy;  /* outward normal -y */
        case FACE_UPPER_Y: return  dudy;
        case FACE_LOWER_Z: return -dudz;
        case FACE_UPPER_Z: return  dudz;
        default:           return 0.0;
    }
}

/* ============================================================
 * manufactured_mixed_inhomog
 *
 *   u_star(x,y,z) = exp(x + y + z)
 *   f      (x,y,z) = 3 exp(x + y + z)         (since Laplacian = 3 u_star)
 *
 * Boundary conditions: every face carries genuinely non-zero data,
 * with the three Dirichlet (lower) and three Neumann (upper) faces
 * each inhomogeneous.
 *
 *   FACE_LOWER_X (x=0): u_star = e^(y+z)         -> inhomog Dirichlet
 *   FACE_LOWER_Y (y=0): u_star = e^(x+z)         -> inhomog Dirichlet
 *   FACE_LOWER_Z (z=0): u_star = e^(x+y)         -> inhomog Dirichlet
 *   FACE_UPPER_X (x=1): du/dx = e * e^(y+z)      -> inhomog Neumann
 *   FACE_UPPER_Y (y=1): du/dy = e^(x+1+z)        -> inhomog Neumann
 *   FACE_UPPER_Z (z=1): du/dz = e^(x+y+1)        -> inhomog Neumann
 *
 * Non-singular (Dirichlet faces pin the constant).  All six faces
 * exercise a non-trivial code path: every Dirichlet write is to a
 * non-zero value, and every Neumann ghost mirror picks up a non-zero
 * + 2 h q correction.  This is the only preset where the entire BC
 * machinery sees inhomogeneous data simultaneously.
 * ============================================================ */

static double exp3_rhs(double x, double y, double z)
{
    return 3.0 * exp(x + y + z);
}

static double exp3_exact(double x, double y, double z)
{
    return exp(x + y + z);
}

/* Dirichlet boundary value: u_star evaluated at (x,y,z).  The face
 * argument is unused -- on lower-x the caller pins x=0, on lower-y
 * pins y=0, etc., so the formula evaluates to the right per-face
 * boundary value automatically. */
static double exp3_dirichlet(double x, double y, double z, face_id_t face)
{
    (void)face;
    return exp(x + y + z);
}

/* Outward-normal derivative.  For u_star = e^(x+y+z),
 *   du/dx = du/dy = du/dz = u_star.
 * On the upper faces the outward normal is +x_hat / +y_hat / +z_hat,
 * so q = +du/dx_n = u_star at the face point.  The lower faces are
 * Dirichlet here and never invoke this callback. */
static double exp3_neumann(double x, double y, double z, face_id_t face)
{
    (void)face;
    return exp(x + y + z);
}

/* ============================================================
 * Registry table.  Designated initialisers (C99) make the BC spec
 * legible without hand-counting array indices.
 * ============================================================ */

#define DIRICHLET_HOMOG_FACE          { BC_DIRICHLET, true,  NULL }
#define DIRICHLET_INHOMOG_FACE(cb)    { BC_DIRICHLET, false, (cb) }
#define NEUMANN_HOMOG_FACE            { BC_NEUMANN,   true,  NULL }
#define NEUMANN_INHOMOG_FACE(cb)      { BC_NEUMANN,   false, (cb) }

const struct problem_t g_problems[] = {
    {
        .name = "manufactured_dirichlet_homog",
        .bc   = {
            .face = {
                [FACE_LOWER_X] = DIRICHLET_HOMOG_FACE,
                [FACE_UPPER_X] = DIRICHLET_HOMOG_FACE,
                [FACE_LOWER_Y] = DIRICHLET_HOMOG_FACE,
                [FACE_UPPER_Y] = DIRICHLET_HOMOG_FACE,
                [FACE_LOWER_Z] = DIRICHLET_HOMOG_FACE,
                [FACE_UPPER_Z] = DIRICHLET_HOMOG_FACE,
            },
        },
        .rhs      = sin3_rhs,
        .u_exact  = sin3_exact,
        .singular = false,
    },

    {
        .name = "manufactured_dirichlet_inhomog",
        .bc   = {
            .face = {
                [FACE_LOWER_X] = DIRICHLET_INHOMOG_FACE(cos3_dirichlet),
                [FACE_UPPER_X] = DIRICHLET_INHOMOG_FACE(cos3_dirichlet),
                [FACE_LOWER_Y] = DIRICHLET_INHOMOG_FACE(cos3_dirichlet),
                [FACE_UPPER_Y] = DIRICHLET_INHOMOG_FACE(cos3_dirichlet),
                [FACE_LOWER_Z] = DIRICHLET_INHOMOG_FACE(cos3_dirichlet),
                [FACE_UPPER_Z] = DIRICHLET_INHOMOG_FACE(cos3_dirichlet),
            },
        },
        .rhs      = cos3_rhs,
        .u_exact  = cos3_exact,
        .singular = false,
    },

    {
        .name = "manufactured_neumann_homog",
        .bc   = {
            .face = {
                [FACE_LOWER_X] = NEUMANN_HOMOG_FACE,
                [FACE_UPPER_X] = NEUMANN_HOMOG_FACE,
                [FACE_LOWER_Y] = NEUMANN_HOMOG_FACE,
                [FACE_UPPER_Y] = NEUMANN_HOMOG_FACE,
                [FACE_LOWER_Z] = NEUMANN_HOMOG_FACE,
                [FACE_UPPER_Z] = NEUMANN_HOMOG_FACE,
            },
        },
        .rhs      = cos3_rhs,
        .u_exact  = cos3_exact,
        .singular = true,
    },

    {
        .name = "manufactured_neumann_inhomog",
        .bc   = {
            .face = {
                [FACE_LOWER_X] = NEUMANN_INHOMOG_FACE(sin3_half_neumann),
                [FACE_UPPER_X] = NEUMANN_INHOMOG_FACE(sin3_half_neumann),
                [FACE_LOWER_Y] = NEUMANN_INHOMOG_FACE(sin3_half_neumann),
                [FACE_UPPER_Y] = NEUMANN_INHOMOG_FACE(sin3_half_neumann),
                [FACE_LOWER_Z] = NEUMANN_INHOMOG_FACE(sin3_half_neumann),
                [FACE_UPPER_Z] = NEUMANN_INHOMOG_FACE(sin3_half_neumann),
            },
        },
        .rhs      = sin3_half_rhs,
        .u_exact  = sin3_half_exact,
        .singular = true,
    },

    {
        .name = "manufactured_mixed_inhomog",
        .bc   = {
            .face = {
                /* All six faces inhomogeneous: 3 Dirichlet (lower) +
                 * 3 Neumann (upper), each with non-zero data from
                 * u_star = e^(x+y+z). */
                [FACE_LOWER_X] = DIRICHLET_INHOMOG_FACE(exp3_dirichlet),
                [FACE_LOWER_Y] = DIRICHLET_INHOMOG_FACE(exp3_dirichlet),
                [FACE_LOWER_Z] = DIRICHLET_INHOMOG_FACE(exp3_dirichlet),
                [FACE_UPPER_X] = NEUMANN_INHOMOG_FACE(exp3_neumann),
                [FACE_UPPER_Y] = NEUMANN_INHOMOG_FACE(exp3_neumann),
                [FACE_UPPER_Z] = NEUMANN_INHOMOG_FACE(exp3_neumann),
            },
        },
        .rhs      = exp3_rhs,
        .u_exact  = exp3_exact,
        .singular = false,
    },

    {
        .name = "manufactured_mixed",
        .bc   = {
            .face = {
                /* Lower-x: u_star vanishes there (homogeneous Dirichlet).
                 * The other five faces all carry Neumann data via
                 * mixed_neumann; the y- and z-face callbacks evaluate
                 * to 0 because sin(0) = sin(pi) = 0, so those faces
                 * are effectively homogeneous Neumann.  Upper-x
                 * carries du/dx = 2 cos(pi y) cos(pi z), which is
                 * non-zero -- the inhomogeneous Neumann path under
                 * test. */
                [FACE_LOWER_X] = DIRICHLET_HOMOG_FACE,
                [FACE_UPPER_X] = NEUMANN_INHOMOG_FACE(mixed_neumann),
                [FACE_LOWER_Y] = NEUMANN_INHOMOG_FACE(mixed_neumann),
                [FACE_UPPER_Y] = NEUMANN_INHOMOG_FACE(mixed_neumann),
                [FACE_LOWER_Z] = NEUMANN_INHOMOG_FACE(mixed_neumann),
                [FACE_UPPER_Z] = NEUMANN_INHOMOG_FACE(mixed_neumann),
            },
        },
        .rhs      = mixed_rhs,
        .u_exact  = mixed_exact,
        .singular = false,
    },

    /* sentinel */
    { .name = NULL },
};

const struct problem_t *problem_lookup(const char *name)
{
    if (!name) return NULL;
    for (const struct problem_t *p = g_problems; p->name != NULL; p++)
        if (strcmp(p->name, name) == 0) return p;
    return NULL;
}
