#include "gauss_seidel.h"
#include "comm.h"
#include "domain.h"
#include <math.h>
#include <mpi.h>

/******************************************************************
* Purpose: Enforce homogeneous Dirichlet boundary conditions (u = 0) on
*     variable `var` at all physical boundary faces. A face is a physical
*     boundary when this rank has no MPI neighbour on that side
*     (lower/upper_*_rank == INVALID_RANK). Ghost-zone faces that do have
*     MPI neighbours are not touched.
* Input Variables:
*     gfs: struct ngfs_3d*, 3D grid function container
*     var: int, index of the variable in gfs->vars[] to apply BCs to
* Output Variables:
*     gfs->vars[var]->val: double*, boundary-face points set to 0.0
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void apply_bc_3d(struct ngfs_3d *gfs, int var)
{
    double *v  = gfs->vars[var]->val;
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t nz = gfs->nz;

    /* Lower-x face: i = 0 */
    if (gfs->domain.lower_x_rank == INVALID_RANK)
        for (int k = 0; k < nz; k++)
            for (int j = 0; j < ny; j++)
                v[gf_indx_3d(gfs, 0, j, k)] = 0.0;

    /* Upper-x face: i = nx-1 */
    if (gfs->domain.upper_x_rank == INVALID_RANK)
        for (int k = 0; k < nz; k++)
            for (int j = 0; j < ny; j++)
                v[gf_indx_3d(gfs, nx - 1, j, k)] = 0.0;

    /* Lower-y face: j = 0 */
    if (gfs->domain.lower_y_rank == INVALID_RANK)
        for (int k = 0; k < nz; k++)
            for (int i = 0; i < nx; i++)
                v[gf_indx_3d(gfs, i, 0, k)] = 0.0;

    /* Upper-y face: j = ny-1 */
    if (gfs->domain.upper_y_rank == INVALID_RANK)
        for (int k = 0; k < nz; k++)
            for (int i = 0; i < nx; i++)
                v[gf_indx_3d(gfs, i, ny - 1, k)] = 0.0;

    /* Lower-z face: k = 0 */
    if (gfs->domain.lower_z_rank == INVALID_RANK)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                v[gf_indx_3d(gfs, i, j, 0)] = 0.0;

    /* Upper-z face: k = nz-1 */
    if (gfs->domain.upper_z_rank == INVALID_RANK)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                v[gf_indx_3d(gfs, i, j, nz - 1)] = 0.0;
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
    const int gs = gfs->gs;
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

    for (int iter = 0; iter < n_smooth; iter++)
    {
        /* ---- Red sweep ---- */
        for (int k = gs; k < nz - gs; k++)
        {
            for (int j = gs; j < ny - gs; j++)
            {
                /*
                 * Determine which i to start for red points in this row.
                 * Red: (global_i + global_j + global_k) is even.
                 * Global parity of j+k for this row:
                 */
                const int jk_parity    = (local_j0 + j + local_k0 + k) % 2;
                /*
                 * Is local i=gs (the first interior point) red?
                 * global_i at i=gs is (local_i0 + gs), which is >= 1 for gs=1.
                 */
                const int first_is_black = (local_i0 + gs + jk_parity) % 2;
                const int red_start      = gs + first_is_black;

                for (int i = red_start; i < nx - gs; i += 2)
                {
                    const int64_t idx  = i + (j + k * ny) * nx;
                    const double u_new =
                          cx * (u[idx + 1]        + u[idx - 1])
                        + cy * (u[idx + stride_y] + u[idx - stride_y])
                        + cz * (u[idx + stride_z] + u[idx - stride_z])
                        - cs * rhs[idx];
                    u[idx] = (1.0 - omega) * u[idx] + omega * u_new;
                }
            }
        }
        sync_var_3d(gfs, VAR_SOL);
        apply_bc_3d(gfs, VAR_SOL);

        /* ---- Black sweep ---- */
        for (int k = gs; k < nz - gs; k++)
        {
            for (int j = gs; j < ny - gs; j++)
            {
                const int jk_parity    = (local_j0 + j + local_k0 + k) % 2;
                const int first_is_black = (local_i0 + gs + jk_parity) % 2;
                const int black_start    = gs + 1 - first_is_black;

                for (int i = black_start; i < nx - gs; i += 2)
                {
                    const int64_t idx  = i + (j + k * ny) * nx;
                    const double u_new =
                          cx * (u[idx + 1]        + u[idx - 1])
                        + cy * (u[idx + stride_y] + u[idx - stride_y])
                        + cz * (u[idx + stride_z] + u[idx - stride_z])
                        - cs * rhs[idx];
                    u[idx] = (1.0 - omega) * u[idx] + omega * u_new;
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
    const int gs = gfs->gs;
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

    double local_max = 0.0;

    for (int k = gs; k < nz - gs; k++)
    {
        for (int j = gs; j < ny - gs; j++)
        {
            for (int i = gs; i < nx - gs; i++)
            {
                const int64_t idx = i + (j + k * ny) * nx;
                const double Lu =
                      (u[idx + 1]        + u[idx - 1]        - 2.0 * u[idx]) * idx2
                    + (u[idx + stride_y] + u[idx - stride_y] - 2.0 * u[idx]) * idy2
                    + (u[idx + stride_z] + u[idx - stride_z] - 2.0 * u[idx]) * idz2;
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
