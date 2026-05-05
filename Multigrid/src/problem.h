#ifndef PROBLEM_H
#define PROBLEM_H

#include "bc.h"
#include <stdbool.h>

/* Forward declaration to avoid pulling gf.h into every consumer of
 * this header.  The helpers below take a struct ngfs_3d * which is
 * defined in gf.h. */
struct ngfs_3d;

/* Scalar field on the unit cube.  Used both for the RHS f(x,y,z) and
 * for the exact solution u_exact(x,y,z) when one is available. */
typedef double (*scalar_field_fn)(double x, double y, double z);

/* A "problem preset" is a static record bundling every problem-
 * specific callback the driver needs: per-face BC kinds + value
 * callbacks, RHS, and (optionally) an exact solution for use in
 * manufactured-solution tests.  Adding a new problem is a one-file
 * edit: define a few pure functions in problem_registry.c and append
 * a problem_t record to g_problems[]. */
struct problem_t {
    const char        *name;
    struct bc_spec_t   bc;        /* per-face BC kinds + value callbacks */
    scalar_field_fn    rhs;       /* f(x,y,z); must be non-NULL */
    scalar_field_fn    u_exact;   /* optional; NULL if no closed form */
    bool               singular;  /* true => mean-zero projection on the
                                   * V-cycle, plus a compatibility check
                                   * at configure time.  Required for
                                   * all-Neumann presets. */
};

/* Registry of built-in presets; terminated by .name = NULL.  Defined
 * in problem_registry.c. */
extern const struct problem_t g_problems[];

/******************************************************************
* Purpose: Look up a preset by name.  Linear scan over g_problems[].
* Input Variables:
*     name: const char*, preset name; may be NULL (returns NULL).
* Output Variables:
*     (none)
* Return Values and indicators of success / failure
*     const struct problem_t*, the matching record, or NULL on miss.
*******************************************************************/
const struct problem_t *problem_lookup(const char *name);

/* ----- Helpers used by the driver and the test harness ----------- */

/******************************************************************
* Purpose: Initialise VAR_RHS at every grid point from
*     problem->rhs(x,y,z).  Initialises VAR_SOL = 0.  When the
*     problem is flagged singular, additionally subtracts the discrete
*     mean of VAR_RHS so the compatibility condition is satisfied to
*     within discretisation error.
* Input Variables:
*     gfs: struct ngfs_3d*, root grid; must have VAR_SOL and VAR_RHS
*         allocated
*     problem: const struct problem_t*, must be non-NULL with non-NULL
*         rhs callback
* Output Variables:
*     gfs->vars[VAR_RHS]->val, gfs->vars[VAR_SOL]->val written
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void problem_initialise_rhs(struct ngfs_3d *gfs,
                            const struct problem_t *problem);

/******************************************************************
* Purpose: Initialise the boundary nodes of VAR_SOL from the per-face
*     BC callbacks.  This is the only place inhomogeneous Dirichlet
*     data enters the fine-grid solution; coarse levels never call it.
*     For homogeneous Dirichlet faces the function is a no-op (the
*     calloc'd zeros are already correct).  Phase 3+ extends this to
*     also set Neumann ghost rows.
* Input Variables:
*     gfs: struct ngfs_3d*, root grid; gfs->bc must point to the same
*         spec the rest of the solver uses
*     problem: const struct problem_t*, must be non-NULL
* Output Variables:
*     gfs->vars[VAR_SOL]->val updated at the physical-boundary nodes
*         and ghost rows owned by this rank
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void problem_apply_initial_bc(struct ngfs_3d *gfs,
                              const struct problem_t *problem);

/******************************************************************
* Purpose: When problem->u_exact is non-NULL, compute |u_h - u_exact|
*     in the L^infty norm over the locally-owned grid, then reduce
*     across the Cartesian communicator with MPI_MAX.  For singular
*     problems the global mean of (u_h - u_exact) is subtracted first,
*     since the discrete solution is determined only up to an additive
*     constant.  Returns -1.0 if the preset has no exact solution.
* Input Variables:
*     gfs: struct ngfs_3d*, root grid
*     problem: const struct problem_t*, must be non-NULL
* Output Variables:
*     (none)
* Return Values and indicators of success / failure
*     double, global L-infinity error, or -1.0 if no u_exact.
*******************************************************************/
double problem_compute_max_error(struct ngfs_3d *gfs,
                                 const struct problem_t *problem);

/******************************************************************
* Purpose: Subtract the global mean of `var` from every locally-owned
*     non-ghost value.  Used for singular (all-Neumann) problems where
*     the discrete operator has the constant function in its null
*     space; periodic projection prevents the constant component from
*     drifting under finite-precision arithmetic during the V-cycle.
* Input Variables:
*     gfs: struct ngfs_3d*
*     var: int, index of the variable in gfs->vars[]
* Output Variables:
*     gfs->vars[var]->val: double*, mean-subtracted in place
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void problem_project_mean_zero(struct ngfs_3d *gfs, int var);

#endif /* PROBLEM_H */
