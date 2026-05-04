#ifndef GAUSS_SEIDEL_H
#define GAUSS_SEIDEL_H

#include "gf.h"

/* Variable indices in the ngfs_3d vars[] array */
#define VAR_SOL 0  /* solution u             */
#define VAR_RHS 1  /* right-hand side / source f */
#define VAR_DEF 2  /* defect  r = Lu - f     */

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
void apply_bc_3d(struct ngfs_3d *gfs, int var);

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
void gauss_seidel_3d(struct ngfs_3d *gfs, int n_smooth, double omega);

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
double calc_defect_3d(struct ngfs_3d *gfs);

#endif /* GAUSS_SEIDEL_H */
