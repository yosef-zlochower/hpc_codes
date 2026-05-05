#ifndef BC_H
#define BC_H

#include <stdbool.h>

/* Boundary-condition kind on a single physical-boundary face. */
typedef enum {
    BC_DIRICHLET = 0,
    BC_NEUMANN   = 1,
    /* room for future kinds: BC_ROBIN, BC_PERIODIC */
} bc_kind_t;

/* Identifier for one of the six physical-boundary faces of the box.
 * The numeric values are stable; rely on them in switch statements
 * but prefer the named constants for readability. */
typedef enum {
    FACE_LOWER_X = 0,
    FACE_UPPER_X = 1,
    FACE_LOWER_Y = 2,
    FACE_UPPER_Y = 3,
    FACE_LOWER_Z = 4,
    FACE_UPPER_Z = 5,
    NUM_FACES    = 6
} face_id_t;

/* Boundary value/flux callback.  For Dirichlet BCs, returns u(x,y,z);
 * for Neumann BCs, returns du/dn (outward-normal derivative).  The
 * `face` argument lets a single callback dispatch per-face logic if
 * convenient.  Ignored when the corresponding bc_face_t has
 * homogeneous == true. */
typedef double (*bc_fn_t)(double x, double y, double z, face_id_t face);

struct bc_face_t {
    bc_kind_t kind;
    bool      homogeneous; /* true => value/flux is identically 0 */
    bc_fn_t   value;       /* may be NULL when homogeneous == true */
};

/* Per-face boundary-condition specification for a 3D box. */
struct bc_spec_t {
    struct bc_face_t face[NUM_FACES];
};

/******************************************************************
* Purpose: Build the homogeneous variant of a boundary spec.  Each
*     face keeps its kind from `src` but is forced to
*     `homogeneous = true` and `value = NULL` in `dst`.  Used by the
*     hierarchy constructor to seed coarse-level BCs: the unknown on
*     a coarse grid is the correction e_H, whose boundary condition
*     is always homogeneous regardless of the original problem.
* Input Variables:
*     src: const struct bc_spec_t*, fine-grid spec (may be NULL,
*         in which case dst is left untouched)
*     dst: struct bc_spec_t*, output spec; must be non-NULL when
*         src is non-NULL
* Output Variables:
*     dst: struct bc_spec_t*, homogenised copy of src
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void bc_spec_homogenize(const struct bc_spec_t *src,
                        struct bc_spec_t *dst);

#endif /* BC_H */
