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

/* Per-face geometric descriptor consumed by apply_face_3d.
 *
 *   id        : enum identifying the face (used by callbacks).
 *   axis      : 0=x, 1=y, 2=z -- which Cartesian axis is normal to the face.
 *   is_upper  : false for the lower (i=0) end of `axis`, true for upper (i=N-1).
 *
 * The three pieces of metadata are enough to recover everything else
 * apply_face_3d needs (boundary/interior indices, spacing, the two
 * looped axes, the +/- h/2 vertex fix-up on hybrid axes) from gfs. */
struct face_geom {
    face_id_t id;
    int       axis;
    bool      is_upper;
};

/* Apply the BC for one physical-boundary face.  The face is identified
 * by `g`; `gfs->bc` (queried via face_kind / face_value) decides
 * whether it is Dirichlet (write the boundary vertex) or Neumann
 * (write the ghost row via u_ghost = u_int + h*q).
 *
 * Dirichlet vertex coordinate on hybrid axes: an axis with at least
 * one Neumann face has gfs->x0 shifted by -h/2 so the cell-centre
 * formula x = x0 + i*dx is exact.  Dirichlet vertices sit at +/-h/2
 * from that formula; the bdy_coord computation below adds the fix-up
 * (sign = +1 at lower, -1 at upper).
 *
 * Neumann boundary-plane coordinate: the boundary plane lies at the
 * midpoint of the ghost slot (i_bdy) and the first interior cell
 * centre (i_int).  Phase 6.2B attempted a 4-point higher-order ghost
 * extrapolation here but it made the matrix non-M (negative
 * off-diagonal on u_3) and SOR with omega=1.5 diverged on it; the
 * cell-centred mirror is the supported approach. */
static void apply_face_3d(struct ngfs_3d *gfs, int var,
                          const struct face_geom *g)
{
    /* Pull axis-dependent geometry. */
    int64_t nax, n_outer, n_middle;
    double  dax, a0, out0, dout, mid0, dmid;
    bool    axis_has_neu;

    switch (g->axis) {
        case 0:
            nax  = gfs->nx; dax = gfs->dx; a0   = gfs->x0;
            n_outer  = gfs->nz; out0 = gfs->z0; dout = gfs->dz;
            n_middle = gfs->ny; mid0 = gfs->y0; dmid = gfs->dy;
            axis_has_neu = axis_x_neumann(gfs);
            break;
        case 1:
            nax  = gfs->ny; dax = gfs->dy; a0   = gfs->y0;
            n_outer  = gfs->nz; out0 = gfs->z0; dout = gfs->dz;
            n_middle = gfs->nx; mid0 = gfs->x0; dmid = gfs->dx;
            axis_has_neu = axis_y_neumann(gfs);
            break;
        default:  /* 2 */
            nax  = gfs->nz; dax = gfs->dz; a0   = gfs->z0;
            n_outer  = gfs->ny; out0 = gfs->y0; dout = gfs->dy;
            n_middle = gfs->nx; mid0 = gfs->x0; dmid = gfs->dx;
            axis_has_neu = axis_z_neumann(gfs);
            break;
    }

    const int64_t i_bdy = g->is_upper ? (nax - 1) : 0;
    const int64_t i_int = g->is_upper ? (nax - 2) : 1;
    const double  sign  = g->is_upper ? -1.0 : +1.0;  /* outward direction */
    const double  half  = 0.5 * dax;

    const bc_kind_t kind = face_kind(gfs, g->id);
    const bc_fn_t   cb   = face_value(gfs, g->id, var);
    const face_id_t f    = g->id;

    /* face_coord: x|y|z value passed to the callback.
     *   Dirichlet: physical position of the boundary vertex.
     *   Neumann:   physical position of the boundary plane (mirror midpoint). */
    const double face_coord =
        a0 + (double)i_bdy * dax
        + sign * ((kind == BC_DIRICHLET && !axis_has_neu) ? 0.0 : half);

    double *v = gfs->vars[var]->val;

    /* Sweep the two axes other than `g->axis`.
     *   axis=0 (x-face): outer=k, middle=j -> (x, y, z) = (face, middle, outer)
     *   axis=1 (y-face): outer=k, middle=i -> (x, y, z) = (middle, face, outer)
     *   axis=2 (z-face): outer=j, middle=i -> (x, y, z) = (middle, outer, face)
     * The choice puts the i axis in the inner loop where it gives a
     * unit-stride access through gf_indx_3d (i fastest-varying). */
    for (int64_t o = 0; o < n_outer; o++) {
        const double coord_outer  = out0 + (double)o * dout;
        for (int64_t m = 0; m < n_middle; m++) {
            const double coord_middle = mid0 + (double)m * dmid;

            int64_t idx_bdy, idx_int;
            double  x, y, z;
            switch (g->axis) {
                case 0:
                    idx_bdy = gf_indx_3d(gfs, i_bdy, m, o);
                    idx_int = gf_indx_3d(gfs, i_int, m, o);
                    x = face_coord;   y = coord_middle; z = coord_outer; break;
                case 1:
                    idx_bdy = gf_indx_3d(gfs, m, i_bdy, o);
                    idx_int = gf_indx_3d(gfs, m, i_int, o);
                    x = coord_middle; y = face_coord;   z = coord_outer; break;
                default:  /* 2 */
                    idx_bdy = gf_indx_3d(gfs, m, o, i_bdy);
                    idx_int = gf_indx_3d(gfs, m, o, i_int);
                    x = coord_middle; y = coord_outer; z = face_coord;   break;
            }

            if (kind == BC_DIRICHLET) {
                v[idx_bdy] = cb ? cb(x, y, z, f) : 0.0;
            } else {
                const double q = cb ? cb(x, y, z, f) : 0.0;
                v[idx_bdy] = v[idx_int] + dax * q;
            }
        }
    }
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
*         axis) is *written* with the prescribed value g(x,y,z).
*       * Neumann: the ghost row at i=0 (lower) or i=nx-1 (upper) is
*         written via the cell-centred mirror u_ghost = u_int + h*q.
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
    static const struct face_geom faces[6] = {
        { FACE_LOWER_X, 0, false },
        { FACE_UPPER_X, 0, true  },
        { FACE_LOWER_Y, 1, false },
        { FACE_UPPER_Y, 1, true  },
        { FACE_LOWER_Z, 2, false },
        { FACE_UPPER_Z, 2, true  },
    };
    const int neighbour_rank[6] = {
        gfs->domain.lower_x_rank, gfs->domain.upper_x_rank,
        gfs->domain.lower_y_rank, gfs->domain.upper_y_rank,
        gfs->domain.lower_z_rank, gfs->domain.upper_z_rank,
    };

    for (int i = 0; i < 6; i++) {
        if (neighbour_rank[i] == INVALID_RANK)
            apply_face_3d(gfs, var, &faces[i]);
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
/* All inputs to the per-cell SOR update, packed so the kernel can be
 * lifted out of gauss_seidel_3d into a static inline function (the
 * data pointers, the precomputed stencil weights, the per-axis hybrid
 * Dirichlet-vertex flags, and the relaxation parameter). */
struct sor_ctx_3d {
    double       *u;          /* solution buffer (read+write)                */
    const double *rhs;        /* source term (read only)                     */

    int64_t nx, ny, nz;       /* local extents incl. ghosts                  */
    int64_t stride_y;         /* gf_indx_3d j-stride: nx                     */
    int64_t stride_z;         /* gf_indx_3d k-stride: nx * ny                */

    double cx, cy, cz, cs;    /* fast-path stencil weights (per-axis + src)  */
    double inv_dx2, inv_dy2, inv_dz2;  /* slow-path inverse spacings         */

    double omega;             /* SOR relaxation parameter                    */

    /* "Hybrid-Dirichlet-vertex on this end of this axis?" flags.  Set
     * iff the rank owns the physical face, the face is Dirichlet, and
     * the opposite face on the same axis is Neumann.  The slow path
     * (4-point Lagrange stencil) triggers at i in {1, nx-2} on the
     * affected axis. */
    bool x_lower_d_v, x_upper_d_v;
    bool y_lower_d_v, y_upper_d_v;
    bool z_lower_d_v, z_upper_d_v;
};

/* Compute and apply the SOR update at one cell, including the per-axis
 * non-uniform stencil at cells adjacent to a hybrid Dirichlet vertex
 * (the gap on the D side is h/2, not h).  Most cells take the fast
 * path (precomputed cx, cy, cz, cs).  The slow path is triggered only
 * at i in {1, nx-2} on a hybrid x axis, etc. -- O(N^2) cells out of
 * O(N^3) -- so the branch is well predicted.
 *
 * Marked static inline so the compiler folds the function body into
 * both red and black sweep loops with no call overhead at -O3. */
static inline void sor_update_3d(int64_t idx, int64_t i, int64_t j, int64_t k,
                                 const struct sor_ctx_3d *c)
{
    const double u_xm = c->u[idx - 1];
    const double u_xp = c->u[idx + 1];
    const double u_ym = c->u[idx - c->stride_y];
    const double u_yp = c->u[idx + c->stride_y];
    const double u_zm = c->u[idx - c->stride_z];
    const double u_zp = c->u[idx + c->stride_z];

    const bool x_sp = (i == 1 && c->x_lower_d_v) || (i == c->nx - 2 && c->x_upper_d_v);
    const bool y_sp = (j == 1 && c->y_lower_d_v) || (j == c->ny - 2 && c->y_upper_d_v);
    const bool z_sp = (k == 1 && c->z_lower_d_v) || (k == c->nz - 2 && c->z_upper_d_v);

    double u_new;
    if (!x_sp && !y_sp && !z_sp)
    {
        u_new = c->cx * (u_xp + u_xm) + c->cy * (u_yp + u_ym)
              + c->cz * (u_zp + u_zm) - c->cs * c->rhs[idx];
    }
    else
    {
        /* Phase 6.2 4-point Lagrange stencil at the cell adjacent to a
         * hybrid Dirichlet vertex.  Cubic Lagrange-extrapolate a virtual
         * ghost u(-h/2) through the four points (u_v, u_self, u_far,
         * u_far_far), then plug into the standard centred Laplacian.
         * Result (lower D side, i=1):
         *   u_xx ~= (1/(5 h^2))(16 u_v - 25 u_self + 10 u_far - u_far_far).
         * O(h^2) Laplacian truncation -> rate 2 globally (vs. the O(h)
         * of the Phase 3 3-point form).  No q-coefficient, so the
         * discrete compatibility is unaffected (the Phase 5 concern
         * was specific to the Neumann ghost variant). */
        double diag_x = 2.0, off_x = u_xm + u_xp;
        if (x_sp) {
            diag_x = 5.0;
            if (i == 1) {
                const double u_ff = c->u[idx + 2];
                off_x = (16.0/5.0) * u_xm + 2.0 * u_xp - (1.0/5.0) * u_ff;
            } else {
                const double u_ff = c->u[idx - 2];
                off_x = (16.0/5.0) * u_xp + 2.0 * u_xm - (1.0/5.0) * u_ff;
            }
        }
        double diag_y = 2.0, off_y = u_ym + u_yp;
        if (y_sp) {
            diag_y = 5.0;
            if (j == 1) {
                const double u_ff = c->u[idx + 2 * c->stride_y];
                off_y = (16.0/5.0) * u_ym + 2.0 * u_yp - (1.0/5.0) * u_ff;
            } else {
                const double u_ff = c->u[idx - 2 * c->stride_y];
                off_y = (16.0/5.0) * u_yp + 2.0 * u_ym - (1.0/5.0) * u_ff;
            }
        }
        double diag_z = 2.0, off_z = u_zm + u_zp;
        if (z_sp) {
            diag_z = 5.0;
            if (k == 1) {
                const double u_ff = c->u[idx + 2 * c->stride_z];
                off_z = (16.0/5.0) * u_zm + 2.0 * u_zp - (1.0/5.0) * u_ff;
            } else {
                const double u_ff = c->u[idx - 2 * c->stride_z];
                off_z = (16.0/5.0) * u_zp + 2.0 * u_zm - (1.0/5.0) * u_ff;
            }
        }
        const double diag = diag_x * c->inv_dx2 + diag_y * c->inv_dy2 + diag_z * c->inv_dz2;
        const double off  = off_x  * c->inv_dx2 + off_y  * c->inv_dy2 + off_z  * c->inv_dz2;
        u_new = (off - c->rhs[idx]) / diag;
    }

    c->u[idx] = (1.0 - c->omega) * c->u[idx] + c->omega * u_new;
}

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
     * stencil is replaced on that axis by the 4-point Lagrange form
     * (doc/history/CellCentred_plan.md sec. 4.2). */
    const struct sor_ctx_3d ctx = {
        .u   = gfs->vars[VAR_SOL]->val,
        .rhs = gfs->vars[VAR_RHS]->val,

        .nx = nx, .ny = ny, .nz = nz,
        /* Index strides matching gf_indx_3d: idx = i + (j + k*ny)*nx */
        .stride_y = nx,
        .stride_z = nx * ny,

        .cx = dy2dz2 / denom,           /* weight for u[i+/-1, j, k] */
        .cy = dx2dz2 / denom,           /* weight for u[i, j+/-1, k] */
        .cz = dx2dy2 / denom,           /* weight for u[i, j, k+/-1] */
        .cs = dx2dy2 * dz2 / denom,     /* weight for source         */

        .inv_dx2 = 1.0 / dx2,
        .inv_dy2 = 1.0 / dy2,
        .inv_dz2 = 1.0 / dz2,

        .omega = omega,

        .x_lower_d_v = (gfs->domain.lower_x_rank == INVALID_RANK)
                       && !gfs->domain.neumann_lower_x
                       &&  gfs->domain.neumann_upper_x,
        .x_upper_d_v = (gfs->domain.upper_x_rank == INVALID_RANK)
                       &&  gfs->domain.neumann_lower_x
                       && !gfs->domain.neumann_upper_x,
        .y_lower_d_v = (gfs->domain.lower_y_rank == INVALID_RANK)
                       && !gfs->domain.neumann_lower_y
                       &&  gfs->domain.neumann_upper_y,
        .y_upper_d_v = (gfs->domain.upper_y_rank == INVALID_RANK)
                       &&  gfs->domain.neumann_lower_y
                       && !gfs->domain.neumann_upper_y,
        .z_lower_d_v = (gfs->domain.lower_z_rank == INVALID_RANK)
                       && !gfs->domain.neumann_lower_z
                       &&  gfs->domain.neumann_upper_z,
        .z_upper_d_v = (gfs->domain.upper_z_rank == INVALID_RANK)
                       &&  gfs->domain.neumann_lower_z
                       && !gfs->domain.neumann_upper_z,
    };

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
                    sor_update_3d(idx, i, j, k, &ctx);
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
                    sor_update_3d(idx, i, j, k, &ctx);
                }
            }
        }
        sync_var_3d(gfs, VAR_SOL);
        apply_bc_3d(gfs, VAR_SOL);
    }
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
                 * (u_- + u_+ - 2 u_self) for uniform spacing; the
                 * 4-point Lagrange extrapolation
                 * (1/5)(16 u_v - 25 u_self + 10 u_far - u_far_far) at
                 * the cell adjacent to a hybrid Dirichlet vertex.
                 * See gauss_seidel_3d for the derivation. */
                double Lx_raw, Ly_raw, Lz_raw;
                if (x_sp) {
                    if (i == 1) {
                        const double u_ff = u[idx + 2];
                        Lx_raw = (16.0 * u_xm - 25.0 * u_self + 10.0 * u_xp - u_ff) / 5.0;
                    } else {
                        const double u_ff = u[idx - 2];
                        Lx_raw = (16.0 * u_xp - 25.0 * u_self + 10.0 * u_xm - u_ff) / 5.0;
                    }
                } else {
                    Lx_raw = u_xm + u_xp - 2.0 * u_self;
                }
                if (y_sp) {
                    if (j == 1) {
                        const double u_ff = u[idx + 2 * stride_y];
                        Ly_raw = (16.0 * u_ym - 25.0 * u_self + 10.0 * u_yp - u_ff) / 5.0;
                    } else {
                        const double u_ff = u[idx - 2 * stride_y];
                        Ly_raw = (16.0 * u_yp - 25.0 * u_self + 10.0 * u_ym - u_ff) / 5.0;
                    }
                } else {
                    Ly_raw = u_ym + u_yp - 2.0 * u_self;
                }
                if (z_sp) {
                    if (k == 1) {
                        const double u_ff = u[idx + 2 * stride_z];
                        Lz_raw = (16.0 * u_zm - 25.0 * u_self + 10.0 * u_zp - u_ff) / 5.0;
                    } else {
                        const double u_ff = u[idx - 2 * stride_z];
                        Lz_raw = (16.0 * u_zp - 25.0 * u_self + 10.0 * u_zm - u_ff) / 5.0;
                    }
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
