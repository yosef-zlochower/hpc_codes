#include "gauss_seidel.h"
#include "comm.h"
#include "domain.h"
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>

/* ----- Per-rank sweep bounds for the smoother & defect kernels ----------
 *
 * On each rank, every face that is (a) a physical-boundary face and
 * (b) carries a Neumann condition forces the boundary node to be an
 * unknown rather than a constraint; the sweep range widens by one on
 * that side to include it.  Dirichlet faces (and faces handed off to
 * an MPI neighbour) keep the historical [gs, n-gs) range.
 *
 * The kernel uses the per-face Neumann flags to decide, at each
 * boundary node it visits, whether to mirror the off-domain neighbour
 * rather than read u[idx-1] / u[idx+1] etc., which would index
 * outside the array.
 */
struct sweep_bounds_3d
{
    int64_t i_lo, i_hi;
    int64_t j_lo, j_hi;
    int64_t k_lo, k_hi;
    bool    nx_lo_neumann, nx_hi_neumann;
    bool    ny_lo_neumann, ny_hi_neumann;
    bool    nz_lo_neumann, nz_hi_neumann;
    /* Per-face Neumann data callback (q = du/dn).  NULL on homogeneous
     * Neumann faces -- the kernel skips the 2 h q term in that case
     * and the mirror reduces to the homogeneous form u_ghost = u_int. */
    bc_fn_t q_x_lo, q_x_hi;
    bc_fn_t q_y_lo, q_y_hi;
    bc_fn_t q_z_lo, q_z_hi;
};

/* Pull the inhomogeneous-Neumann callback for face f from gfs->bc, or
 * NULL when the face is either non-Neumann or Neumann-homogeneous.
 * Coarse levels get NULL on every face because bc_spec_homogenize sets
 * homogeneous = true and value = NULL during hierarchy construction. */
static inline bc_fn_t neumann_q(const struct ngfs_3d *gfs, face_id_t f)
{
    if (!gfs->bc) return NULL;
    const struct bc_face_t *fb = &gfs->bc->face[f];
    if (fb->kind != BC_NEUMANN) return NULL;
    if (fb->homogeneous)        return NULL;
    return fb->value;
}

static struct sweep_bounds_3d compute_sweep_bounds(const struct ngfs_3d *gfs)
{
    struct sweep_bounds_3d sb;
    const int     gs = gfs->gs;
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t nz = gfs->nz;

    /* Default: interior-only sweep, no Neumann mirrors, no q callbacks. */
    sb.i_lo = gs; sb.i_hi = nx - gs;
    sb.j_lo = gs; sb.j_hi = ny - gs;
    sb.k_lo = gs; sb.k_hi = nz - gs;
    sb.nx_lo_neumann = sb.nx_hi_neumann = false;
    sb.ny_lo_neumann = sb.ny_hi_neumann = false;
    sb.nz_lo_neumann = sb.nz_hi_neumann = false;
    sb.q_x_lo = sb.q_x_hi = NULL;
    sb.q_y_lo = sb.q_y_hi = NULL;
    sb.q_z_lo = sb.q_z_hi = NULL;

    if (!gfs->bc) return sb;  /* default = homogeneous Dirichlet everywhere */

    if (gfs->domain.lower_x_rank == INVALID_RANK
            && gfs->bc->face[FACE_LOWER_X].kind == BC_NEUMANN) {
        sb.i_lo = 0;
        sb.nx_lo_neumann = true;
        sb.q_x_lo = neumann_q(gfs, FACE_LOWER_X);
    }
    if (gfs->domain.upper_x_rank == INVALID_RANK
            && gfs->bc->face[FACE_UPPER_X].kind == BC_NEUMANN) {
        sb.i_hi = nx;
        sb.nx_hi_neumann = true;
        sb.q_x_hi = neumann_q(gfs, FACE_UPPER_X);
    }
    if (gfs->domain.lower_y_rank == INVALID_RANK
            && gfs->bc->face[FACE_LOWER_Y].kind == BC_NEUMANN) {
        sb.j_lo = 0;
        sb.ny_lo_neumann = true;
        sb.q_y_lo = neumann_q(gfs, FACE_LOWER_Y);
    }
    if (gfs->domain.upper_y_rank == INVALID_RANK
            && gfs->bc->face[FACE_UPPER_Y].kind == BC_NEUMANN) {
        sb.j_hi = ny;
        sb.ny_hi_neumann = true;
        sb.q_y_hi = neumann_q(gfs, FACE_UPPER_Y);
    }
    if (gfs->domain.lower_z_rank == INVALID_RANK
            && gfs->bc->face[FACE_LOWER_Z].kind == BC_NEUMANN) {
        sb.k_lo = 0;
        sb.nz_lo_neumann = true;
        sb.q_z_lo = neumann_q(gfs, FACE_LOWER_Z);
    }
    if (gfs->domain.upper_z_rank == INVALID_RANK
            && gfs->bc->face[FACE_UPPER_Z].kind == BC_NEUMANN) {
        sb.k_hi = nz;
        sb.nz_hi_neumann = true;
        sb.q_z_hi = neumann_q(gfs, FACE_UPPER_Z);
    }

    return sb;
}

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
*       * Dirichlet: face nodes are written.  When the face is
*         homogeneous (or var != VAR_SOL, since the defect carries no
*         inhomogeneous data of its own) the value is 0; otherwise the
*         per-face callback supplies u(x,y,z).
*       * Neumann: not yet implemented in Phase 2.  The function aborts
*         via MPI_Abort if a Neumann face is encountered (Phase 3 adds
*         the ghost-row write).
*
*     When gfs->bc == NULL, every face defaults to homogeneous Dirichlet
*     -- the historical behaviour that operator-level unit tests rely on.
* Input Variables:
*     gfs: struct ngfs_3d*, 3D grid function container
*     var: int, index of the variable in gfs->vars[] to apply BCs to
* Output Variables:
*     gfs->vars[var]->val: double*, physical-boundary face points are
*         updated according to the per-face spec
* Return Values and indicators of success / failure
*     (none).  May call MPI_Abort if a Neumann face is encountered
*     before Phase 3 lands the implementation.
*******************************************************************/
void apply_bc_3d(struct ngfs_3d *gfs, int var)
{
    double *v  = gfs->vars[var]->val;
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t nz = gfs->nz;

    /* Lower-x face: i = 0 */
    if (gfs->domain.lower_x_rank == INVALID_RANK)
    {
        const face_id_t f = FACE_LOWER_X;
        if (face_kind(gfs, f) == BC_DIRICHLET)
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  x  = gfs->x0;            /* i = 0 */
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
            /* BC_NEUMANN: face nodes are unknowns updated by the
             * smoother sweep, which substitutes the mirror neighbour
             * via compute_sweep_bounds().  No write needed here. */
        }
    }

    /* Upper-x face: i = nx-1 */
    if (gfs->domain.upper_x_rank == INVALID_RANK)
    {
        const face_id_t f = FACE_UPPER_X;
        if (face_kind(gfs, f) == BC_DIRICHLET)
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  x  = gfs->x0 + (nx - 1) * gfs->dx;
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
            /* BC_NEUMANN: face nodes are unknowns updated by the
             * smoother sweep, which substitutes the mirror neighbour
             * via compute_sweep_bounds().  No write needed here. */
        }
    }

    /* Lower-y face: j = 0 */
    if (gfs->domain.lower_y_rank == INVALID_RANK)
    {
        const face_id_t f = FACE_LOWER_Y;
        if (face_kind(gfs, f) == BC_DIRICHLET)
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  y  = gfs->y0;            /* j = 0 */
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
            /* BC_NEUMANN: face nodes are unknowns updated by the
             * smoother sweep, which substitutes the mirror neighbour
             * via compute_sweep_bounds().  No write needed here. */
        }
    }

    /* Upper-y face: j = ny-1 */
    if (gfs->domain.upper_y_rank == INVALID_RANK)
    {
        const face_id_t f = FACE_UPPER_Y;
        if (face_kind(gfs, f) == BC_DIRICHLET)
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  y  = gfs->y0 + (ny - 1) * gfs->dy;
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
            /* BC_NEUMANN: face nodes are unknowns updated by the
             * smoother sweep, which substitutes the mirror neighbour
             * via compute_sweep_bounds().  No write needed here. */
        }
    }

    /* Lower-z face: k = 0 */
    if (gfs->domain.lower_z_rank == INVALID_RANK)
    {
        const face_id_t f = FACE_LOWER_Z;
        if (face_kind(gfs, f) == BC_DIRICHLET)
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  z  = gfs->z0;            /* k = 0 */
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
            /* BC_NEUMANN: face nodes are unknowns updated by the
             * smoother sweep, which substitutes the mirror neighbour
             * via compute_sweep_bounds().  No write needed here. */
        }
    }

    /* Upper-z face: k = nz-1 */
    if (gfs->domain.upper_z_rank == INVALID_RANK)
    {
        const face_id_t f = FACE_UPPER_Z;
        if (face_kind(gfs, f) == BC_DIRICHLET)
        {
            const bc_fn_t cb = face_value(gfs, f, var);
            const double  z  = gfs->z0 + (nz - 1) * gfs->dz;
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
            /* BC_NEUMANN: face nodes are unknowns updated by the
             * smoother sweep, which substitutes the mirror neighbour
             * via compute_sweep_bounds().  No write needed here. */
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

    /* Per-rank sweep bounds: widened by one on every Neumann face this
     * rank owns.  Dirichlet faces and faces shared with an MPI
     * neighbour keep the [gs, n-gs) interior-only range. */
    const struct sweep_bounds_3d sb = compute_sweep_bounds(gfs);

    /* Pre-compute the constant face-coordinates used by the q-callback
     * lookup at boundary nodes.  Outside the boundary planes these are
     * unused; computing them once avoids re-deriving them inside the
     * hot loop. */
    const double x_face_lo = gfs->x0;
    const double x_face_hi = gfs->x0 + (nx - 1) * gfs->dx;
    const double y_face_lo = gfs->y0;
    const double y_face_hi = gfs->y0 + (ny - 1) * gfs->dy;
    const double z_face_lo = gfs->z0;
    const double z_face_hi = gfs->z0 + (nz - 1) * gfs->dz;
    (void)y_face_lo; (void)y_face_hi;  /* Used only by mirrors below */
    (void)z_face_lo; (void)z_face_hi;
    (void)x_face_lo; (void)x_face_hi;

    /* Mirror neighbour at a Neumann boundary face.
     *   u_ghost = u_interior_neighbour + 2 * h * q(face point)
     * with q = du/dn (outward-normal derivative).  q_cb may be NULL
     * for homogeneous Neumann; in that case the 2 h q term vanishes
     * and the formula reduces to u_ghost = u_interior. */
    #define NEUMANN_MIRROR(u_int_neighbour, h, q_cb, x_arg, y_arg, z_arg, face_id)   \
        ((q_cb) ? ((u_int_neighbour) + 2.0 * (h) * (q_cb)((x_arg),(y_arg),(z_arg),(face_id))) \
                : (u_int_neighbour))

    for (int iter = 0; iter < n_smooth; iter++)
    {
        /* ---- Red sweep ---- */
        for (int64_t k = sb.k_lo; k < sb.k_hi; k++)
        {
            const bool   k_at_lo = (k == 0);
            const bool   k_at_hi = (k == nz - 1);
            const double z       = gfs->z0 + k * gfs->dz;
            for (int64_t j = sb.j_lo; j < sb.j_hi; j++)
            {
                const bool   j_at_lo = (j == 0);
                const bool   j_at_hi = (j == ny - 1);
                const double y       = gfs->y0 + j * gfs->dy;
                /* Red points have (global_i + global_j + global_k) even. */
                const int jk_parity      = (local_j0 + j + local_k0 + k) % 2;
                const int first_is_black = (local_i0 + sb.i_lo + jk_parity) % 2;
                const int64_t red_start  = sb.i_lo + first_is_black;

                for (int64_t i = red_start; i < sb.i_hi; i += 2)
                {
                    const int64_t idx = i + (j + k * ny) * nx;
                    const double  x   = gfs->x0 + i * gfs->dx;

                    const double u_xm = (i == 0      && sb.nx_lo_neumann)
                        ? NEUMANN_MIRROR(u[idx + 1], gfs->dx, sb.q_x_lo,
                                         x_face_lo, y, z, FACE_LOWER_X)
                        : u[idx - 1];
                    const double u_xp = (i == nx - 1 && sb.nx_hi_neumann)
                        ? NEUMANN_MIRROR(u[idx - 1], gfs->dx, sb.q_x_hi,
                                         x_face_hi, y, z, FACE_UPPER_X)
                        : u[idx + 1];
                    const double u_ym = (j_at_lo     && sb.ny_lo_neumann)
                        ? NEUMANN_MIRROR(u[idx + stride_y], gfs->dy, sb.q_y_lo,
                                         x, y_face_lo, z, FACE_LOWER_Y)
                        : u[idx - stride_y];
                    const double u_yp = (j_at_hi     && sb.ny_hi_neumann)
                        ? NEUMANN_MIRROR(u[idx - stride_y], gfs->dy, sb.q_y_hi,
                                         x, y_face_hi, z, FACE_UPPER_Y)
                        : u[idx + stride_y];
                    const double u_zm = (k_at_lo     && sb.nz_lo_neumann)
                        ? NEUMANN_MIRROR(u[idx + stride_z], gfs->dz, sb.q_z_lo,
                                         x, y, z_face_lo, FACE_LOWER_Z)
                        : u[idx - stride_z];
                    const double u_zp = (k_at_hi     && sb.nz_hi_neumann)
                        ? NEUMANN_MIRROR(u[idx - stride_z], gfs->dz, sb.q_z_hi,
                                         x, y, z_face_hi, FACE_UPPER_Z)
                        : u[idx + stride_z];

                    const double u_new =
                          cx * (u_xp + u_xm)
                        + cy * (u_yp + u_ym)
                        + cz * (u_zp + u_zm)
                        - cs * rhs[idx];
                    u[idx] = (1.0 - omega) * u[idx] + omega * u_new;
                }
            }
        }
        sync_var_3d(gfs, VAR_SOL);
        apply_bc_3d(gfs, VAR_SOL);

        /* ---- Black sweep ---- */
        for (int64_t k = sb.k_lo; k < sb.k_hi; k++)
        {
            const bool   k_at_lo = (k == 0);
            const bool   k_at_hi = (k == nz - 1);
            const double z       = gfs->z0 + k * gfs->dz;
            for (int64_t j = sb.j_lo; j < sb.j_hi; j++)
            {
                const bool   j_at_lo = (j == 0);
                const bool   j_at_hi = (j == ny - 1);
                const double y       = gfs->y0 + j * gfs->dy;
                const int jk_parity       = (local_j0 + j + local_k0 + k) % 2;
                const int first_is_black  = (local_i0 + sb.i_lo + jk_parity) % 2;
                const int64_t black_start = sb.i_lo + 1 - first_is_black;

                for (int64_t i = black_start; i < sb.i_hi; i += 2)
                {
                    const int64_t idx = i + (j + k * ny) * nx;
                    const double  x   = gfs->x0 + i * gfs->dx;

                    const double u_xm = (i == 0      && sb.nx_lo_neumann)
                        ? NEUMANN_MIRROR(u[idx + 1], gfs->dx, sb.q_x_lo,
                                         x_face_lo, y, z, FACE_LOWER_X)
                        : u[idx - 1];
                    const double u_xp = (i == nx - 1 && sb.nx_hi_neumann)
                        ? NEUMANN_MIRROR(u[idx - 1], gfs->dx, sb.q_x_hi,
                                         x_face_hi, y, z, FACE_UPPER_X)
                        : u[idx + 1];
                    const double u_ym = (j_at_lo     && sb.ny_lo_neumann)
                        ? NEUMANN_MIRROR(u[idx + stride_y], gfs->dy, sb.q_y_lo,
                                         x, y_face_lo, z, FACE_LOWER_Y)
                        : u[idx - stride_y];
                    const double u_yp = (j_at_hi     && sb.ny_hi_neumann)
                        ? NEUMANN_MIRROR(u[idx - stride_y], gfs->dy, sb.q_y_hi,
                                         x, y_face_hi, z, FACE_UPPER_Y)
                        : u[idx + stride_y];
                    const double u_zm = (k_at_lo     && sb.nz_lo_neumann)
                        ? NEUMANN_MIRROR(u[idx + stride_z], gfs->dz, sb.q_z_lo,
                                         x, y, z_face_lo, FACE_LOWER_Z)
                        : u[idx - stride_z];
                    const double u_zp = (k_at_hi     && sb.nz_hi_neumann)
                        ? NEUMANN_MIRROR(u[idx - stride_z], gfs->dz, sb.q_z_hi,
                                         x, y, z_face_hi, FACE_UPPER_Z)
                        : u[idx + stride_z];

                    const double u_new =
                          cx * (u_xp + u_xm)
                        + cy * (u_yp + u_ym)
                        + cz * (u_zp + u_zm)
                        - cs * rhs[idx];
                    u[idx] = (1.0 - omega) * u[idx] + omega * u_new;
                }
            }
        }
        sync_var_3d(gfs, VAR_SOL);
        apply_bc_3d(gfs, VAR_SOL);
    }

    #undef NEUMANN_MIRROR
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

    /* Sweep range mirrors the smoother's: include Neumann boundary
     * nodes (their defect is the residual of the boundary equation,
     * encoded via the same ghost-row substitution the smoother uses),
     * exclude Dirichlet boundary nodes (whose defect is identically
     * zero -- apply_bc_3d below writes 0 there). */
    const struct sweep_bounds_3d sb = compute_sweep_bounds(gfs);

    /* Constant face-coordinates used by the q-callback at boundary
     * nodes.  Only relevant on inhomogeneous Neumann faces. */
    const double x_face_lo = gfs->x0;
    const double x_face_hi = gfs->x0 + (nx - 1) * gfs->dx;
    const double y_face_lo = gfs->y0;
    const double y_face_hi = gfs->y0 + (ny - 1) * gfs->dy;
    const double z_face_lo = gfs->z0;
    const double z_face_hi = gfs->z0 + (nz - 1) * gfs->dz;

    /* Same ghost-mirror formula as the smoother kernel; q_cb is NULL
     * on homogeneous Neumann faces, in which case the 2 h q term
     * vanishes. */
    #define NEUMANN_MIRROR(u_int_neighbour, h, q_cb, x_arg, y_arg, z_arg, face_id)   \
        ((q_cb) ? ((u_int_neighbour) + 2.0 * (h) * (q_cb)((x_arg),(y_arg),(z_arg),(face_id))) \
                : (u_int_neighbour))

    double local_max = 0.0;

    for (int64_t k = sb.k_lo; k < sb.k_hi; k++)
    {
        const bool   k_at_lo = (k == 0);
        const bool   k_at_hi = (k == nz - 1);
        const double z       = gfs->z0 + k * gfs->dz;
        for (int64_t j = sb.j_lo; j < sb.j_hi; j++)
        {
            const bool   j_at_lo = (j == 0);
            const bool   j_at_hi = (j == ny - 1);
            const double y       = gfs->y0 + j * gfs->dy;
            for (int64_t i = sb.i_lo; i < sb.i_hi; i++)
            {
                const int64_t idx = i + (j + k * ny) * nx;
                const double  x   = gfs->x0 + i * gfs->dx;

                const double u_xm = (i == 0      && sb.nx_lo_neumann)
                    ? NEUMANN_MIRROR(u[idx + 1], gfs->dx, sb.q_x_lo,
                                     x_face_lo, y, z, FACE_LOWER_X)
                    : u[idx - 1];
                const double u_xp = (i == nx - 1 && sb.nx_hi_neumann)
                    ? NEUMANN_MIRROR(u[idx - 1], gfs->dx, sb.q_x_hi,
                                     x_face_hi, y, z, FACE_UPPER_X)
                    : u[idx + 1];
                const double u_ym = (j_at_lo     && sb.ny_lo_neumann)
                    ? NEUMANN_MIRROR(u[idx + stride_y], gfs->dy, sb.q_y_lo,
                                     x, y_face_lo, z, FACE_LOWER_Y)
                    : u[idx - stride_y];
                const double u_yp = (j_at_hi     && sb.ny_hi_neumann)
                    ? NEUMANN_MIRROR(u[idx - stride_y], gfs->dy, sb.q_y_hi,
                                     x, y_face_hi, z, FACE_UPPER_Y)
                    : u[idx + stride_y];
                const double u_zm = (k_at_lo     && sb.nz_lo_neumann)
                    ? NEUMANN_MIRROR(u[idx + stride_z], gfs->dz, sb.q_z_lo,
                                     x, y, z_face_lo, FACE_LOWER_Z)
                    : u[idx - stride_z];
                const double u_zp = (k_at_hi     && sb.nz_hi_neumann)
                    ? NEUMANN_MIRROR(u[idx - stride_z], gfs->dz, sb.q_z_hi,
                                     x, y, z_face_hi, FACE_UPPER_Z)
                    : u[idx + stride_z];

                const double Lu =
                      (u_xp + u_xm - 2.0 * u[idx]) * idx2
                    + (u_yp + u_ym - 2.0 * u[idx]) * idy2
                    + (u_zp + u_zm - 2.0 * u[idx]) * idz2;
                def[idx] = Lu - rhs[idx];
                const double absval = fabs(def[idx]);
                if (absval > local_max)
                    local_max = absval;
            }
        }
    }

    #undef NEUMANN_MIRROR

    apply_bc_3d(gfs, VAR_DEF);

    double global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                  gfs->domain.cart_comm);

    sync_var_3d(gfs, VAR_DEF);

    return global_max;
}
