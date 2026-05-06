#ifndef MULTIGRID_H
#define MULTIGRID_H


#include "gf.h"

/******************************************************************
* Purpose: Attempt to create one coarser-grid level below `parent` in a 2D
*     multigrid hierarchy. The child grid has half the cell count in each
*     direction (grid points = cells/2 + 1). Coarsening is rejected if any
*     direction has an odd cell count or if the child would have fewer than
*     min_cells_per_direction interior cells on any rank (checked via
*     MPI_Allreduce). On success, parent->child and child->parent are linked.
* Input Variables:
*     parent: struct ngfs_2d*, fine grid; must be non-NULL with
*         parent->child == NULL
*     min_cells_per_direction: int, minimum number of interior cells per rank
*         per direction before coarsening stops
* Output Variables:
*     parent->child: struct ngfs_2d*, newly allocated and initialised coarser
*         level, linked bidirectionally to parent
* Return Values and indicators of success / failure
*     0 on success (child created), 1 if final depth reached (odd cell count
*     or too few cells — not an error), -1 on error (parent NULL,
*     parent->child already set, or allocation failure)
*******************************************************************/
int ngfs_2d_create_child(struct ngfs_2d *parent, int min_cells_per_direction);
/******************************************************************
* Purpose: Attempt to create one coarser-grid level below `parent` in a 3D
*     multigrid hierarchy. Analogous to ngfs_2d_create_child.
* Input Variables:
*     parent: struct ngfs_3d*, fine grid; must be non-NULL with
*         parent->child == NULL
*     min_cells_per_direction: int, minimum interior cells per rank per axis
* Output Variables:
*     parent->child: struct ngfs_3d*, newly allocated and initialised coarser
*         level, linked bidirectionally to parent
* Return Values and indicators of success / failure
*     0 on success, 1 at final depth, -1 on error
*******************************************************************/
int ngfs_3d_create_child(struct ngfs_3d *parent, int min_cells_per_direction);

/******************************************************************
* Purpose: Build the complete coarse-grid hierarchy rooted at `root` by
*     repeatedly calling ngfs_2d_create_child until final depth is reached.
* Input Variables:
*     root: struct ngfs_2d*, finest grid; must be non-NULL
*     min_cells_per_direction: int, forwarded to ngfs_2d_create_child
* Output Variables:
*     root->child->child->...: chain of progressively coarser grids
*         allocated and linked
* Return Values and indicators of success / failure
*     0 on success (hierarchy complete), -1 on error
*******************************************************************/
int ngfs_2d_create_hierarchy(struct ngfs_2d *root, int min_cells_per_direction);
/******************************************************************
* Purpose: Build the complete coarse-grid hierarchy rooted at `root` by
*     repeatedly calling ngfs_3d_create_child until final depth is reached.
* Input Variables:
*     root: struct ngfs_3d*, finest grid; must be non-NULL
*     min_cells_per_direction: int, forwarded to ngfs_3d_create_child
* Output Variables:
*     root->child->child->...: chain of progressively coarser grids
* Return Values and indicators of success / failure
*     0 on success, -1 on error
*******************************************************************/
int ngfs_3d_create_hierarchy(struct ngfs_3d *root, int min_cells_per_direction);

/******************************************************************
* Purpose: Injection transfer from a fine-grid variable to a coarse-grid
*     variable in 2D. Each coarse-grid point that coincides with a fine-grid
*     point (child spacing = 2 * parent spacing) receives the fine-grid value
*     directly. Covers all non-ghost child points (including physical boundary
*     points).
* Input Variables:
*     parent: struct ngfs_2d*, fine grid
*     pvar: int, fine-grid variable index
*     child: struct ngfs_2d*, coarse grid
*     cvar: int, coarse-grid variable index to write
* Output Variables:
*     child->vars[cvar]->val: double*, coarse-grid points updated from parent
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void inject_var_2d(struct ngfs_2d *parent, int pvar,
                   struct ngfs_2d *child, int cvar);
/******************************************************************
* Purpose: Injection transfer from a fine-grid variable to a coarse-grid
*     variable in 3D. Analogous to inject_var_2d.
* Input Variables:
*     parent: struct ngfs_3d*, fine grid
*     pvar: int, fine-grid variable index
*     child: struct ngfs_3d*, coarse grid
*     cvar: int, coarse-grid variable index to write
* Output Variables:
*     child->vars[cvar]->val: double*, updated
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void inject_var_3d(struct ngfs_3d *parent, int pvar,
                   struct ngfs_3d *child, int cvar);

/******************************************************************
* Purpose: Full-weighting restriction from fine-grid variable pvar to
*     coarse-grid variable cvar in 2D. Interior coarse-grid points receive a
*     weighted average of the surrounding 3x3 fine-grid stencil: weight 4 at
*     the coincident point, 2 on face-adjacent points, 1 at corners, total
*     weight 16. Physical boundary points of the coarse grid are skipped
*     (call inject_var_2d first to initialise them).
* Input Variables:
*     parent: struct ngfs_2d*, fine grid
*     pvar: int, fine-grid variable index
*     child: struct ngfs_2d*, coarse grid
*     cvar: int, coarse-grid variable index to write
* Output Variables:
*     child->vars[cvar]->val: double*, interior coarse-grid points updated
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void restrict_var_2d(struct ngfs_2d *parent, int pvar,
                     struct ngfs_2d *child, int cvar);
/******************************************************************
* Purpose: Full-weighting restriction from fine-grid variable pvar to
*     coarse-grid variable cvar in 3D. Interior coarse-grid points receive a
*     weighted average of the surrounding 3x3x3 fine-grid stencil: weight 8
*     at the coincident point, 4 on face-adjacent, 2 on edge-adjacent, 1 on
*     corner-adjacent; total weight 64. Physical boundary points are skipped.
* Input Variables:
*     parent: struct ngfs_3d*, fine grid
*     pvar: int, fine-grid variable index
*     child: struct ngfs_3d*, coarse grid
*     cvar: int, coarse-grid variable index to write
* Output Variables:
*     child->vars[cvar]->val: double*, interior coarse-grid points updated
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void restrict_var_3d(struct ngfs_3d *parent, int pvar,
                     struct ngfs_3d *child, int cvar);

/******************************************************************
* Purpose: Prolongation (correction) operator from coarse-grid variable cvar
*     to fine-grid variable pvar in 2D. Subtracts the prolongated
*     coarse-grid values from the fine-grid variable
*     (pval -= interp(cval)) at all interior fine-grid points. The
*     interpolation uses bilinear stencils: coincident points (both indices
*     even) copy directly; axis-midpoints average two neighbours;
*     cell-centre midpoints average four neighbours. Physical boundary
*     fine-grid points are skipped to preserve Dirichlet BCs. Prerequisite:
*     ghost zones on both grids must be valid (call sync_var before this
*     function).
* Input Variables:
*     child: struct ngfs_2d*, coarse grid
*     cvar: int, coarse correction variable index
*     parent: struct ngfs_2d*, fine grid
*     pvar: int, fine variable index to update
* Output Variables:
*     parent->vars[pvar]->val: double*, interior fine-grid points
*         decremented by the prolongated correction
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void prolong_var_2d(struct ngfs_2d *child, int cvar,
                    struct ngfs_2d *parent, int pvar);
/******************************************************************
* Purpose: Prolongation (correction) operator from coarse-grid variable cvar
*     to fine-grid variable pvar in 3D. Subtracts the prolongated
*     coarse-grid values from the fine-grid variable using trilinear
*     interpolation: coincident (copy), face-midpoints (average 2),
*     edge-midpoints (average 4), cell-centres (average 8). Physical
*     boundary fine-grid points are skipped. Prerequisite: ghost zones on
*     both grids must be valid.
* Input Variables:
*     child: struct ngfs_3d*, coarse grid
*     cvar: int, coarse correction variable index
*     parent: struct ngfs_3d*, fine grid
*     pvar: int, fine variable index to update
* Output Variables:
*     parent->vars[pvar]->val: double*, interior fine-grid points
*         decremented by the prolongated correction
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void prolong_var_3d(struct ngfs_3d *child, int cvar,
                    struct ngfs_3d *parent, int pvar);

/******************************************************************
* Purpose: Perform one multigrid V-cycle at level `gfs`. Pre-smooths with
*     n_smooth Gauss-Seidel SOR iterations, computes the defect, and if a
*     coarser level exists and the defect exceeds tol: zeroes the child
*     correction, restricts the defect to the child RHS, recurses into the
*     child, post-smooths with omega and omega=1.0 passes, and recomputes
*     the defect (repeated up to subcycles times). On return, if gfs has a
*     parent, the correction in VAR_SOL is prolongated to the parent.
* Input Variables:
*     gfs: struct ngfs_3d*, current multigrid level; VAR_SOL must be
*         initialised and BCs applied; VAR_RHS must be set
*     n_smooth: int, smoothing steps per sweep
*     omega: double, SOR relaxation parameter
*     tol: double, defect norm below which coarse-grid correction is skipped
*     subcycles: int, maximum number of coarse-grid sub-cycles per V-cycle
*         level
*     verbose: int, 1 = print per-level defect trace and "Starting Vcycle"
*         banner; 0 = silent
* Output Variables:
*     gfs->vars[VAR_SOL]->val: double*, solution updated by smoothing and
*         coarse-grid correction; if gfs->parent != NULL, also
*         gfs->parent->vars[VAR_SOL]->val (double*, decremented by
*         prolongated correction)
* Return Values and indicators of success / failure
*     double, L-infinity norm of the defect at this level after all
*     smoothing steps
*******************************************************************/
double vcycle_3d(struct ngfs_3d *gfs, int n_smooth, double omega,
                 double tol, int subcycles, int verbose);

#endif
