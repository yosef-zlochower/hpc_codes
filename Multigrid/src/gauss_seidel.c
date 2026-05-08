#include "gauss_seidel.h"
#include "comm.h"
#include "domain.h"
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>

/* "Has any Neumann face" predicates: an axis with at least one
 * Neumann face uses the cell-centred-with-extra-point layout
 * (CellCentred Phase 3).  In that layout the standard formula
 * x = gfs->x0 + i*dx gives correct coordinates at every interior
 * cell centre and at every Neumann ghost; Dirichlet vertices on
 * hybrid axes are at +/- h/2 from the formula's prediction and
 * apply_bc_3d adjusts explicitly.  Pure D-D axes have no extra
 * point and no shift -- the formula is exact for both vertices.
 *
 * Phase 3 unification: every Neumann face -- pure N-N or hybrid --
 * has its ghost row written by apply_bc_3d via the cell-centred
 * mirror u_ghost = u_int + h*q.  The smoother and defect kernels
 * consequently never need an in-stencil mirror substitution; the
 * sweep range is always [gs, nx-gs) and the 7-point stencil reads
 * the ghost cell as a regular stored value.  The legacy
 * sweep_bounds_3d struct, the in-stencil NEUMANN_MIRROR macro, and
 * the q-callback lookup that fed it have all been removed
 * (Phase 4 cleanup). */
static inline bool axis_x_neumann(const struct ngfs_3d *gfs)
{ return gfs->domain.neumann_lower_x || gfs->domain.neumann_upper_x; }
static inline bool axis_y_neumann(const struct ngfs_3d *gfs)
{ return gfs->domain.neumann_lower_y || gfs->domain.neumann_upper_y; }
static inline bool axis_z_neumann(const struct ngfs_3d *gfs)
{ return gfs->domain.neumann_lower_z || gfs->domain.neumann_upper_z; }

/* ----- Per-face BC accessors ------------------------------------------------
 *
 * Centralise the "what does this face do?" decision so the six face blocks in
 * apply_bc_3d stay readable.  When gfs->bc is NULL the historical default --
 * homogeneous Dirichlet on every physical-boundary face -- is returned, which
 * keeps operator-level unit tests that don't set BCs explicitly working
 * unchanged.  The defect (any var != VAR_SOL) is always treated as
 * homogeneous regardless of the user's spec, because the residual equation
 * carries no inhomogeneous boundary data of its own.
 */
static inline bc_kind_t face_kind(const struct ngfs_3d *gfs, face_id_t f)
{
    return gfs->bc ? gfs->bc->face[f].kind : BC_DIRICHLET;
}

static inline bc_fn_t face_value(const struct ngfs_3d *gfs, face_id_t f, int var)
{
    if (var != VAR_SOL) return NULL;       /* defect: always homogeneous */
    if (!gfs->bc)       return NULL;       /* default: homogeneous */
    if (gfs->bc->face[f].homogeneous) return NULL;
    return gfs->bc->face[f].value;
}

/******************************************************************
* Purpose: Apply per-face boundary conditions to variable `var` at every
*     physical-boundary face owned by this rank.  A face is a physical
*     boundary when the rank has no MPI neighbour on that side
*     (lower/upper_*_rank == INVALID_RANK).  Ghost faces that have an
*     MPI neighbour are filled by sync_var_3d, not touched here.
*
*     Behaviour per face is determined by gfs->bc:
*       * Dirichlet: the boundary node (i=0 or i=nx-1 on a pure D-D
*         axis, or i=0 / i=nx-1 on the D end of a hybrid D-N / N-D
*         axis) is *written* with the prescribed value g(x,y,z).  On
*         hybrid axes the vertex's physical coordinate is shifted by
*         +/-h/2 from the standard x = gfs->x0 + i*dx formula
*         (because the axis is shifted by -h/2 to give correct cell
*         coordinates) -- the writes below add the +/-h/2 fix-up.
*         When the face is homogeneous (or var != VAR_SOL, since the
*         defect carries no inhomogeneous data of its own) the value
*         is 0; otherwise the per-face callback supplies u(x,y,z).
*       * Neumann: the ghost row at i=0 (lower) or i=nx-1 (upper) is
*         written via the cell-centred mirror u_ghost = u_int + h*q,
*         where q is the user's outward-normal-derivative callback
*         and h is the per-axis spacing.  The boundary plane lies at
*         the midpoint of the ghost and the first interior cell
*         centre -- coordinate gfs->x0 + h/2 (lower) or gfs->x0 +
*         (nx-1)*dx - h/2 (upper).
*
*     When gfs->bc == NULL, every face defaults to homogeneous
*     Dirichlet -- the historical behaviour that operator-level unit
*     tests rely on.
* Input Variables:
*     gfs: struct ngfs_3d*, 3D grid function container
*     var: int, index of the variable in gfs->vars[] to apply BCs to
* Output Variables:
*     gfs->vars[var]->val: double*, physical-boundary nodes (D vertex)
*         and ghost rows (N face) updated according to the per-face spec
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void apply_bc_3d(struct ngfs_3d *gfs, int var)
{
    double *v  = gfs->vars[var]->val;
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t nz = gfs->nz;

    /* Phase 3 unification: every Neumann face -- whether the axis is
     * pure N-N, hybrid D-N, or hybrid N-D -- is treated the same way.
     * apply_bc_3d writes the ghost row via u_ghost = u_int + h*q.
     *
     * The axis-shifted-or-not flag controls *coordinate* fix-ups:
     * - When the axis has at least one Neumann face, gfs->x0 is
     *   shifted by -h/2 so that the standard formula x = x0 + i*dx
     *   gives correct cell-centre coordinates.  Dirichlet vertices
     *   on hybrid axes are at i=0 (D-N) or i=nx-1 (N-D) but their
     *   *physical* position is at +/-h/2 from the formula's
     *   prediction; the writes below add the +/-h/2 fix-up.
     * - When the axis is pure D-D, gfs->x0 is at the box edge and
     *   the formula is exact for both vertices. */
    const bool x_neumann_axis = axis_x_neumann(gfs);
    const bool y_neumann_axis = axis_y_neumann(gfs);
    const bool z_neumann_axis = axis_z_neumann(gfs);

    /* Lower-x face: i = 0 */
    if (gfs->domain.lower_x_rank == INVALID_RANK)
    {
        const face_id_t f = FACE_LOWER_X;
        if (face_kind(gfs, f) == BC_DIRICHLET)
        {
            /* Vertex at i=0.  Pure DD: coord is gfs->x0 = a.  D-N:
             * gfs->x0 = a-h/2 (shifted), so vertex coord = a = x0 + h/2. */
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  x  = gfs->x0 + (x_neumann_axis ? 0.5 * gfs->dx : 0.0);
            if (cb)
            {
                for (int k = 0; k < nz; k++) {
                    const double z = gfs->z0 + k * gfs->dz;
                    for (int j = 0; j < ny; j++) {
                        const double y = gfs->y0 + j * gfs->dy;
                        v[gf_indx_3d(gfs, 0, j, k)] = cb(x, y, z, f);
                    }
                }
            }
            else
            {
                for (int k = 0; k < nz; k++)
                    for (int j = 0; j < ny; j++)
                        v[gf_indx_3d(gfs, 0, j, k)] = 0.0;
            }
        }
        else
        {
            /* Cell-centred Neumann ghost: write u[0] = u[1] + h*q.
             * Boundary plane at gfs->x0 + h/2 = a.  q is the outward
             * normal derivative; lower-x outward normal is -x_hat, so
             * (u[1]-u[0])/h = -q -> u[0] = u[1] + h*q. */
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  x_bdy = gfs->x0 + 0.5 * gfs->dx;
            for (int k = 0; k < nz; k++) {
                const double z = gfs->z0 + k * gfs->dz;
                for (int j = 0; j < ny; j++) {
                    const double y = gfs->y0 + j * gfs->dy;
                    const double q = cb ? cb(x_bdy, y, z, f) : 0.0;
                    const int64_t i0 = gf_indx_3d(gfs, 0, j, k);
                    const int64_t i1 = gf_indx_3d(gfs, 1, j, k);
                    v[i0] = v[i1] + gfs->dx * q;
                }
            }
        }
    }

    /* Upper-x face: i = nx-1 */
    if (gfs->domain.upper_x_rank == INVALID_RANK)
    {
        const face_id_t f = FACE_UPPER_X;
        if (face_kind(gfs, f) == BC_DIRICHLET)
        {
            /* Vertex at i=nx-1.  Pure DD: coord = x0 + (nx-1)*dx = b.
             * N-D: x0 + (nx-1)*dx = b+h/2, so vertex coord = b = ... - h/2. */
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  x  = gfs->x0 + (nx - 1) * gfs->dx
                             - (x_neumann_axis ? 0.5 * gfs->dx : 0.0);
            if (cb)
            {
                for (int k = 0; k < nz; k++) {
                    const double z = gfs->z0 + k * gfs->dz;
                    for (int j = 0; j < ny; j++) {
                        const double y = gfs->y0 + j * gfs->dy;
                        v[gf_indx_3d(gfs, nx - 1, j, k)] = cb(x, y, z, f);
                    }
                }
            }
            else
            {
                for (int k = 0; k < nz; k++)
                    for (int j = 0; j < ny; j++)
                        v[gf_indx_3d(gfs, nx - 1, j, k)] = 0.0;
            }
        }
        else
        {
            /* Cell-centred Neumann ghost: write u[nx-1] = u[nx-2] + h*q.
             * Upper-x outward normal is +x_hat, so (u[nx-1]-u[nx-2])/h
             * = q -> u[nx-1] = u[nx-2] + h*q. */
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  x_bdy = gfs->x0 + (nx - 1) * gfs->dx - 0.5 * gfs->dx;
            for (int k = 0; k < nz; k++) {
                const double z = gfs->z0 + k * gfs->dz;
                for (int j = 0; j < ny; j++) {
                    const double y = gfs->y0 + j * gfs->dy;
                    const double q = cb ? cb(x_bdy, y, z, f) : 0.0;
                    const int64_t in_1 = gf_indx_3d(gfs, nx - 1, j, k);
                    const int64_t in_2 = gf_indx_3d(gfs, nx - 2, j, k);
                    v[in_1] = v[in_2] + gfs->dx * q;
                }
            }
        }
    }

    /* Lower-y face: j = 0 */
    if (gfs->domain.lower_y_rank == INVALID_RANK)
    {
        const face_id_t f = FACE_LOWER_Y;
        if (face_kind(gfs, f) == BC_DIRICHLET)
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  y  = gfs->y0 + (y_neumann_axis ? 0.5 * gfs->dy : 0.0);
            if (cb)
            {
                for (int k = 0; k < nz; k++) {
                    const double z = gfs->z0 + k * gfs->dz;
                    for (int i = 0; i < nx; i++) {
                        const double x = gfs->x0 + i * gfs->dx;
                        v[gf_indx_3d(gfs, i, 0, k)] = cb(x, y, z, f);
                    }
                }
            }
            else
            {
                for (int k = 0; k < nz; k++)
                    for (int i = 0; i < nx; i++)
                        v[gf_indx_3d(gfs, i, 0, k)] = 0.0;
            }
        }
        else
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  y_bdy = gfs->y0 + 0.5 * gfs->dy;
            for (int k = 0; k < nz; k++) {
                const double z = gfs->z0 + k * gfs->dz;
                for (int i = 0; i < nx; i++) {
                    const double x = gfs->x0 + i * gfs->dx;
                    const double q = cb ? cb(x, y_bdy, z, f) : 0.0;
                    const int64_t j0 = gf_indx_3d(gfs, i, 0, k);
                    const int64_t j1 = gf_indx_3d(gfs, i, 1, k);
                    v[j0] = v[j1] + gfs->dy * q;
                }
            }
        }
    }

    /* Upper-y face: j = ny-1 */
    if (gfs->domain.upper_y_rank == INVALID_RANK)
    {
        const face_id_t f = FACE_UPPER_Y;
        if (face_kind(gfs, f) == BC_DIRICHLET)
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  y  = gfs->y0 + (ny - 1) * gfs->dy
                             - (y_neumann_axis ? 0.5 * gfs->dy : 0.0);
            if (cb)
            {
                for (int k = 0; k < nz; k++) {
                    const double z = gfs->z0 + k * gfs->dz;
                    for (int i = 0; i < nx; i++) {
                        const double x = gfs->x0 + i * gfs->dx;
                        v[gf_indx_3d(gfs, i, ny - 1, k)] = cb(x, y, z, f);
                    }
                }
            }
            else
            {
                for (int k = 0; k < nz; k++)
                    for (int i = 0; i < nx; i++)
                        v[gf_indx_3d(gfs, i, ny - 1, k)] = 0.0;
            }
        }
        else
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  y_bdy = gfs->y0 + (ny - 1) * gfs->dy - 0.5 * gfs->dy;
            for (int k = 0; k < nz; k++) {
                const double z = gfs->z0 + k * gfs->dz;
                for (int i = 0; i < nx; i++) {
                    const double x = gfs->x0 + i * gfs->dx;
                    const double q = cb ? cb(x, y_bdy, z, f) : 0.0;
                    const int64_t jn_1 = gf_indx_3d(gfs, i, ny - 1, k);
                    const int64_t jn_2 = gf_indx_3d(gfs, i, ny - 2, k);
                    v[jn_1] = v[jn_2] + gfs->dy * q;
                }
            }
        }
    }

    /* Lower-z face: k = 0 */
    if (gfs->domain.lower_z_rank == INVALID_RANK)
    {
        const face_id_t f = FACE_LOWER_Z;
        if (face_kind(gfs, f) == BC_DIRICHLET)
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  z  = gfs->z0 + (z_neumann_axis ? 0.5 * gfs->dz : 0.0);
            if (cb)
            {
                for (int j = 0; j < ny; j++) {
                    const double y = gfs->y0 + j * gfs->dy;
                    for (int i = 0; i < nx; i++) {
                        const double x = gfs->x0 + i * gfs->dx;
                        v[gf_indx_3d(gfs, i, j, 0)] = cb(x, y, z, f);
                    }
                }
            }
            else
            {
                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++)
                        v[gf_indx_3d(gfs, i, j, 0)] = 0.0;
            }
        }
        else
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  z_bdy = gfs->z0 + 0.5 * gfs->dz;
            for (int j = 0; j < ny; j++) {
                const double y = gfs->y0 + j * gfs->dy;
                for (int i = 0; i < nx; i++) {
                    const double x = gfs->x0 + i * gfs->dx;
                    const double q = cb ? cb(x, y, z_bdy, f) : 0.0;
                    const int64_t k0 = gf_indx_3d(gfs, i, j, 0);
                    const int64_t k1 = gf_indx_3d(gfs, i, j, 1);
                    v[k0] = v[k1] + gfs->dz * q;
                }
            }
        }
    }

    /* Upper-z face: k = nz-1 */
    if (gfs->domain.upper_z_rank == INVALID_RANK)
    {
        const face_id_t f = FACE_UPPER_Z;
        if (face_kind(gfs, f) == BC_DIRICHLET)
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  z  = gfs->z0 + (nz - 1) * gfs->dz
                             - (z_neumann_axis ? 0.5 * gfs->dz : 0.0);
            if (cb)
            {
                for (int j = 0; j < ny; j++) {
                    const double y = gfs->y0 + j * gfs->dy;
                    for (int i = 0; i < nx; i++) {
                        const double x = gfs->x0 + i * gfs->dx;
                        v[gf_indx_3d(gfs, i, j, nz - 1)] = cb(x, y, z, f);
                    }
                }
            }
            else
            {
                for (int j = 0; j < ny; j++)
                    for (int i = 0; i < nx; i++)
                        v[gf_indx_3d(gfs, i, j, nz - 1)] = 0.0;
            }
        }
        else
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  z_bdy = gfs->z0 + (nz - 1) * gfs->dz - 0.5 * gfs->dz;
            for (int j = 0; j < ny; j++) {
                const double y = gfs->y0 + j * gfs->dy;
                for (int i = 0; i < nx; i++) {
                    const double x = gfs->x0 + i * gfs->dx;
                    const double q = cb ? cb(x, y, z_bdy, f) : 0.0;
                    const int64_t kn_1 = gf_indx_3d(gfs, i, j, nz - 1);
                    const int64_t kn_2 = gf_indx_3d(gfs, i, j, nz - 2);
                    v[kn_1] = v[kn_2] + gfs->dz * q;
                }
            }
        }
    }
}

/******************************************************************
* Purpose: Perform n_smooth red-black Gauss-Seidel SOR iterations on
*     VAR_SOL, using VAR_RHS as the right-hand side. Red and black coloring
*     is determined by the parity of the global grid index
*     (global_i + global_j + global_k) so the pattern is consistent across
*     all MPI ranks. Each iteration consists of a red sweep, ghost
*     synchronisation + BC application, a black sweep, then ghost
*     synchronisation + BC application.
* Input Variables:
*     gfs: struct ngfs_3d*, grid function container; VAR_SOL and VAR_RHS
*         must be allocated and initialised
*     n_smooth: int, number of complete red-black iterations to perform
*     omega: double, SOR relaxation parameter; 1.0 = plain Gauss-Seidel,
*         1 < omega < 2 for SOR
* Output Variables:
*     gfs->vars[VAR_SOL]->val: double*, solution field updated in-place;
*         ghost zones are valid on exit
* Return Values and indicators of success / failure
*     void. Side effects: calls sync_var_3d and apply_bc_3d after each
*     half-sweep.
*******************************************************************/
void gauss_seidel_3d(struct ngfs_3d *gfs, int n_smooth, double omega)
{
    const int     gs = gfs->gs;
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t nz = gfs->nz;

    /* Stencil coefficients (precomputed once per level) */
    const double dx2    = gfs->dx * gfs->dx;
    const double dy2    = gfs->dy * gfs->dy;
    const double dz2    = gfs->dz * gfs->dz;
    const double dx2dy2 = dx2 * dy2;
    const double dx2dz2 = dx2 * dz2;
    const double dy2dz2 = dy2 * dz2;
    const double denom  = 2.0 * (dx2dy2 + dx2dz2 + dy2dz2);
    const double cx = dy2dz2 / denom;   /* weight for u[i±1, j, k] */
    const double cy = dx2dz2 / denom;   /* weight for u[i, j±1, k] */
    const double cz = dx2dy2 / denom;   /* weight for u[i, j, k±1] */
    const double cs = dx2dy2 * dz2 / denom; /* weight for source   */

    /* Index strides matching gf_indx_3d: idx = i + (j + k*ny)*nx */
    const int64_t stride_y = nx;
    const int64_t stride_z = nx * ny;

    double *u   = gfs->vars[VAR_SOL]->val;
    double *rhs = gfs->vars[VAR_RHS]->val;

    /* Global index offsets: global_i = local_i0 + i  (similarly j, k) */
    const int64_t local_i0 = gfs->domain.local_i0;
    const int64_t local_j0 = gfs->domain.local_j0;
    const int64_t local_k0 = gfs->domain.local_k0;

    /* Sweep range: under CellCentred Phase 3 every Neumann face has
     * its ghost row written by apply_bc_3d before each sweep, so the
     * smoother always stays inside [gs, nx-gs) on every axis. */

    /* Hybrid axis flags: a hybrid axis has D on one end and N on
     * the other.  At the cell adjacent to the D vertex the gap to
     * the vertex is h/2 rather than h, and the standard 7-point
     * stencil is replaced on that axis by the non-uniform formula
     *   u_xx ~= (4/(3 h^2)) * (2 u_v - 3 u_self + u_far).
     * (CellCentred_plan.md sec. 4.2.) */
    const bool x_lower_d_v = (gfs->domain.lower_x_rank == INVALID_RANK)
                          && !gfs->domain.neumann_lower_x
                          &&  gfs->domain.neumann_upper_x;
    const bool x_upper_d_v = (gfs->domain.upper_x_rank == INVALID_RANK)
                          &&  gfs->domain.neumann_lower_x
                          && !gfs->domain.neumann_upper_x;
    const bool y_lower_d_v = (gfs->domain.lower_y_rank == INVALID_RANK)
                          && !gfs->domain.neumann_lower_y
                          &&  gfs->domain.neumann_upper_y;
    const bool y_upper_d_v = (gfs->domain.upper_y_rank == INVALID_RANK)
                          &&  gfs->domain.neumann_lower_y
                          && !gfs->domain.neumann_upper_y;
    const bool z_lower_d_v = (gfs->domain.lower_z_rank == INVALID_RANK)
                          && !gfs->domain.neumann_lower_z
                          &&  gfs->domain.neumann_upper_z;
    const bool z_upper_d_v = (gfs->domain.upper_z_rank == INVALID_RANK)
                          &&  gfs->domain.neumann_lower_z
                          && !gfs->domain.neumann_upper_z;

    /* Inverse-h^2 constants for the non-uniform stencil's slow path. */
    const double inv_dx2 = 1.0 / dx2;
    const double inv_dy2 = 1.0 / dy2;
    const double inv_dz2 = 1.0 / dz2;

    /* Compute the SOR update at one cell, including the per-axis
     * non-uniform stencil at cells adjacent to a hybrid Dirichlet
     * vertex (the gap on the D side is h/2, not h).  Most cells take
     * the fast path (precomputed cx, cy, cz, cs).  The slow path is
     * triggered only at i in {1, nx-2} on a hybrid x axis, etc. --
     * O(N^2) cells out of O(N^3). */
    #define APPLY_SOR_3D(idx_, i_, j_, k_)                                                 \
        do {                                                                                \
            const double u_xm_ = u[(idx_) - 1];                                            \
            const double u_xp_ = u[(idx_) + 1];                                            \
            const double u_ym_ = u[(idx_) - stride_y];                                     \
            const double u_yp_ = u[(idx_) + stride_y];                                     \
            const double u_zm_ = u[(idx_) - stride_z];                                     \
            const double u_zp_ = u[(idx_) + stride_z];                                     \
            const bool x_sp_ = ((i_) == 1 && x_lower_d_v) || ((i_) == nx - 2 && x_upper_d_v); \
            const bool y_sp_ = ((j_) == 1 && y_lower_d_v) || ((j_) == ny - 2 && y_upper_d_v); \
            const bool z_sp_ = ((k_) == 1 && z_lower_d_v) || ((k_) == nz - 2 && z_upper_d_v); \
            double u_new_;                                                                  \
            if (!x_sp_ && !y_sp_ && !z_sp_) {                                              \
                u_new_ = cx * (u_xp_ + u_xm_) + cy * (u_yp_ + u_ym_)                       \
                       + cz * (u_zp_ + u_zm_) - cs * rhs[(idx_)];                          \
            } else {                                                                        \
                /* Finite-volume non-uniform stencil at the cell adjacent \
                 * to a hybrid Dirichlet vertex.  Cell-averaged Laplacian \
                 * (1/h) * (u'(right_face) - u'(left_face)) with: \
                 *   right face (interior): centred (u_far - u_self)/h \
                 *   left  face (vertex):  one-sided (u_self - u_v)/(h/2) = 2(u_self-u_v)/h \
                 * gives  (2 u_v - 3 u_self + u_far)/h^2.  Diagonal coeff \
                 * is -3 (vs. -2 for uniform); off-diagonal is 2*u_v + u_far. */ \
                double diag_x_ = 2.0, off_x_ = u_xm_ + u_xp_;                              \
                if (x_sp_) {                                                                \
                    diag_x_ = 3.0;                                                          \
                    if ((i_) == 1) off_x_ = 2.0 * u_xm_ + u_xp_;                            \
                    else           off_x_ = 2.0 * u_xp_ + u_xm_;                            \
                }                                                                           \
                double diag_y_ = 2.0, off_y_ = u_ym_ + u_yp_;                              \
                if (y_sp_) {                                                                \
                    diag_y_ = 3.0;                                                          \
                    if ((j_) == 1) off_y_ = 2.0 * u_ym_ + u_yp_;                            \
                    else           off_y_ = 2.0 * u_yp_ + u_ym_;                            \
                }                                                                           \
                double diag_z_ = 2.0, off_z_ = u_zm_ + u_zp_;                              \
                if (z_sp_) {                                                                \
                    diag_z_ = 3.0;                                                          \
                    if ((k_) == 1) off_z_ = 2.0 * u_zm_ + u_zp_;                            \
                    else           off_z_ = 2.0 * u_zp_ + u_zm_;                            \
                }                                                                           \
                const double diag_ = diag_x_ * inv_dx2 + diag_y_ * inv_dy2 + diag_z_ * inv_dz2; \
                const double off_  = off_x_  * inv_dx2 + off_y_  * inv_dy2 + off_z_  * inv_dz2; \
                u_new_ = (off_ - rhs[(idx_)]) / diag_;                                      \
            }                                                                               \
            u[(idx_)] = (1.0 - omega) * u[(idx_)] + omega * u_new_;                        \
        } while (0)

    for (int iter = 0; iter < n_smooth; iter++)
    {
        /* ---- Red sweep ---- */
        for (int64_t k = gs; k < nz - gs; k++)
        {
            for (int64_t j = gs; j < ny - gs; j++)
            {
                /* Red points have (global_i + global_j + global_k) even. */
                const int jk_parity      = (local_j0 + j + local_k0 + k) % 2;
                const int first_is_black = (local_i0 + gs + jk_parity) % 2;
                const int64_t red_start  = gs + first_is_black;

                for (int64_t i = red_start; i < nx - gs; i += 2)
                {
                    const int64_t idx = i + (j + k * ny) * nx;
                    APPLY_SOR_3D(idx, i, j, k);
                }
            }
        }
        sync_var_3d(gfs, VAR_SOL);
        apply_bc_3d(gfs, VAR_SOL);

        /* ---- Black sweep ---- */
        for (int64_t k = gs; k < nz - gs; k++)
        {
            for (int64_t j = gs; j < ny - gs; j++)
            {
                const int jk_parity       = (local_j0 + j + local_k0 + k) % 2;
                const int first_is_black  = (local_i0 + gs + jk_parity) % 2;
                const int64_t black_start = gs + 1 - first_is_black;

                for (int64_t i = black_start; i < nx - gs; i += 2)
                {
                    const int64_t idx = i + (j + k * ny) * nx;
                    APPLY_SOR_3D(idx, i, j, k);
                }
            }
        }
        sync_var_3d(gfs, VAR_SOL);
        apply_bc_3d(gfs, VAR_SOL);
    }

    #undef APPLY_SOR_3D
}

/******************************************************************
* Purpose: Compute the defect d = L(u) - f at every interior grid point and
*     store it in VAR_DEF, where L is the 7-point finite-difference
*     Laplacian. Returns the global L-infinity norm of the defect via
*     MPI_Allreduce. Also synchronises VAR_DEF ghost zones via sync_var_3d.
* Input Variables:
*     gfs: struct ngfs_3d*, grid function container; VAR_SOL and VAR_RHS
*         must be up-to-date; ghost zones of VAR_SOL must be valid
* Output Variables:
*     gfs->vars[VAR_DEF]->val: double*, defect field
*         d[i,j,k] = L(u)[i,j,k] - f[i,j,k] written at all interior points;
*         boundary points set to 0 via apply_bc_3d; ghost zones synchronised
*         via sync_var_3d
* Return Values and indicators of success / failure
*     double, global L-infinity norm of the defect (maximum over all ranks
*     and all interior points). Side effects: MPI_Allreduce, sync_var_3d.
*******************************************************************/
double calc_defect_3d(struct ngfs_3d *gfs)
{
    const int     gs = gfs->gs;
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t nz = gfs->nz;

    const double idx2 = 1.0 / (gfs->dx * gfs->dx);
    const double idy2 = 1.0 / (gfs->dy * gfs->dy);
    const double idz2 = 1.0 / (gfs->dz * gfs->dz);

    const int64_t stride_y = nx;
    const int64_t stride_z = nx * ny;

    double *u   = gfs->vars[VAR_SOL]->val;
    double *rhs = gfs->vars[VAR_RHS]->val;
    double *def = gfs->vars[VAR_DEF]->val;

    /* Hybrid axis flags (see gauss_seidel_3d for details); used to
     * apply the non-uniform 3-point stencil at the cell adjacent to
     * a Dirichlet vertex. */
    const bool x_lower_d_v = (gfs->domain.lower_x_rank == INVALID_RANK)
                          && !gfs->domain.neumann_lower_x
                          &&  gfs->domain.neumann_upper_x;
    const bool x_upper_d_v = (gfs->domain.upper_x_rank == INVALID_RANK)
                          &&  gfs->domain.neumann_lower_x
                          && !gfs->domain.neumann_upper_x;
    const bool y_lower_d_v = (gfs->domain.lower_y_rank == INVALID_RANK)
                          && !gfs->domain.neumann_lower_y
                          &&  gfs->domain.neumann_upper_y;
    const bool y_upper_d_v = (gfs->domain.upper_y_rank == INVALID_RANK)
                          &&  gfs->domain.neumann_lower_y
                          && !gfs->domain.neumann_upper_y;
    const bool z_lower_d_v = (gfs->domain.lower_z_rank == INVALID_RANK)
                          && !gfs->domain.neumann_lower_z
                          &&  gfs->domain.neumann_upper_z;
    const bool z_upper_d_v = (gfs->domain.upper_z_rank == INVALID_RANK)
                          &&  gfs->domain.neumann_lower_z
                          && !gfs->domain.neumann_upper_z;

    double local_max = 0.0;

    for (int64_t k = gs; k < nz - gs; k++)
    {
        for (int64_t j = gs; j < ny - gs; j++)
        {
            for (int64_t i = gs; i < nx - gs; i++)
            {
                const int64_t idx = i + (j + k * ny) * nx;
                const double  u_self = u[idx];
                const double  u_xm = u[idx - 1];
                const double  u_xp = u[idx + 1];
                const double  u_ym = u[idx - stride_y];
                const double  u_yp = u[idx + stride_y];
                const double  u_zm = u[idx - stride_z];
                const double  u_zp = u[idx + stride_z];

                const bool x_sp = (i == 1 && x_lower_d_v) || (i == nx - 2 && x_upper_d_v);
                const bool y_sp = (j == 1 && y_lower_d_v) || (j == ny - 2 && y_upper_d_v);
                const bool z_sp = (k == 1 && z_lower_d_v) || (k == nz - 2 && z_upper_d_v);

                /* Per-axis Laplacian contribution.  Standard
                 * (u_- + u_+ - 2 u_self) for uniform spacing; FV
                 * non-uniform (2 u_v - 3 u_self + u_far) at the cell
                 * adjacent to a hybrid Dirichlet vertex.  See
                 * gauss_seidel_3d for the derivation. */
                double Lx_raw, Ly_raw, Lz_raw;
                if (x_sp) {
                    Lx_raw = (i == 1)
                        ? 2.0 * u_xm - 3.0 * u_self + u_xp
                        : 2.0 * u_xp - 3.0 * u_self + u_xm;
                } else {
                    Lx_raw = u_xm + u_xp - 2.0 * u_self;
                }
                if (y_sp) {
                    Ly_raw = (j == 1)
                        ? 2.0 * u_ym - 3.0 * u_self + u_yp
                        : 2.0 * u_yp - 3.0 * u_self + u_ym;
                } else {
                    Ly_raw = u_ym + u_yp - 2.0 * u_self;
                }
                if (z_sp) {
                    Lz_raw = (k == 1)
                        ? 2.0 * u_zm - 3.0 * u_self + u_zp
                        : 2.0 * u_zp - 3.0 * u_self + u_zm;
                } else {
                    Lz_raw = u_zm + u_zp - 2.0 * u_self;
                }

                const double Lu = Lx_raw * idx2 + Ly_raw * idy2 + Lz_raw * idz2;
                def[idx] = Lu - rhs[idx];
                const double absval = fabs(def[idx]);
                if (absval > local_max)
                    local_max = absval;
            }
        }
    }

    apply_bc_3d(gfs, VAR_DEF);

    double global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                  gfs->domain.cart_comm);

    sync_var_3d(gfs, VAR_DEF);

    return global_max;
}
