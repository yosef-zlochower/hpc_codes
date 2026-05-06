#include "gf.h"
#include "comm.h"
#include "multigrid.h"
#include "gauss_seidel.h"
#include <assert.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
int ngfs_2d_create_child(struct ngfs_2d *parent, int min_cells_per_direction)
{
    if (!parent || parent->child != NULL)
        return -1;

    const int64_t cells_x = parent->domain.global_ni - 1;
    const int64_t cells_y = parent->domain.global_nj - 1;

    /* Final depth: odd cell count in any direction prevents exact coarsening */
    if (cells_x % 2 != 0 || cells_y % 2 != 0)
        return 1;

    int cart_dims[2], cart_periods[2], cart_coords[2];
    MPI_Cart_get(parent->domain.cart_comm, 2, cart_dims, cart_periods, cart_coords);
    const int ny_cpu = cart_dims[0];
    const int nx_cpu = cart_dims[1];

    const int64_t global_nx_child = cells_x / 2 + 1;
    const int64_t global_ny_child = cells_y / 2 + 1;
    const double dx_child = 2.0 * parent->domain.dx;
    const double dy_child = 2.0 * parent->domain.dy;

    struct ngfs_2d *child = calloc(1, sizeof(struct ngfs_2d));
    if (!child)
        return -1;
    child->vars = NULL;

    if (setup_2d_domain(nx_cpu, ny_cpu, parent->domain.rank,
                        global_nx_child, global_ny_child,
                        parent->gs,
                        parent->domain.global_x0, parent->domain.global_y0,
                        dx_child, dy_child,
                        &child->domain) != 0)
    {
        free(child);
        return -1;
    }

    /* Final depth: check the minimum local interior cell count across all
     * ranks.  Interior (owned) cells exclude ghost zones on neighbour faces. */
    const int gs = parent->gs;
    const int64_t local_x = child->domain.local_nx
        - (child->domain.lower_x_rank != INVALID_RANK ? gs : 0)
        - (child->domain.upper_x_rank != INVALID_RANK ? gs : 0);
    const int64_t local_y = child->domain.local_ny
        - (child->domain.lower_y_rank != INVALID_RANK ? gs : 0)
        - (child->domain.upper_y_rank != INVALID_RANK ? gs : 0);
    int64_t local_min = ((local_x < local_y) ? local_x : local_y);
    int64_t global_min;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_INT64_T, MPI_MIN,
                  child->domain.cart_comm);

    if (global_min < min_cells_per_direction)
    {
        cleanup_2d_domain(&child->domain);
        free(child);
        return 1;
    }

    ngfs_2d_allocate(parent->nvars, child);

    child->parent = parent;
    parent->child = child;

    return 0;
}

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
int ngfs_2d_create_hierarchy(struct ngfs_2d *root, int min_cells_per_direction)
{
    if (!root)
        return -1;
    struct ngfs_2d *cur = root;
    while (1)
    {
        int ret = ngfs_2d_create_child(cur, min_cells_per_direction);
        if (ret == 1)
            return 0;  /* final depth reached: hierarchy is complete */
        if (ret != 0)
            return -1; /* error */
        cur = cur->child;
    }
}

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
int ngfs_3d_create_child(struct ngfs_3d *parent, int min_cells_per_direction)
{
    if (!parent || parent->child != NULL)
        return -1;

    const int64_t cells_x = parent->domain.global_ni - 1;
    const int64_t cells_y = parent->domain.global_nj - 1;
    const int64_t cells_z = parent->domain.global_nk - 1;

    /* Final depth: odd cell count in any direction prevents exact coarsening */
    if (cells_x % 2 != 0 || cells_y % 2 != 0 || cells_z % 2 != 0)
        return 1;

    int cart_dims[3], cart_periods[3], cart_coords[3];
    MPI_Cart_get(parent->domain.cart_comm, 3, cart_dims, cart_periods, cart_coords);
    const int nz_cpu = cart_dims[0];
    const int ny_cpu = cart_dims[1];
    const int nx_cpu = cart_dims[2];

    const int64_t global_nx_child = cells_x / 2 + 1;
    const int64_t global_ny_child = cells_y / 2 + 1;
    const int64_t global_nz_child = cells_z / 2 + 1;
    const double dx_child = 2.0 * parent->domain.dx;
    const double dy_child = 2.0 * parent->domain.dy;
    const double dz_child = 2.0 * parent->domain.dz;

    struct ngfs_3d *child = calloc(1, sizeof(struct ngfs_3d));
    if (!child)
        return -1;
    child->vars = NULL;

    if (setup_3d_domain(nx_cpu, ny_cpu, nz_cpu, parent->domain.rank,
                        global_nx_child, global_ny_child, global_nz_child,
                        parent->gs,
                        parent->domain.global_x0, parent->domain.global_y0,
                        parent->domain.global_z0,
                        dx_child, dy_child, dz_child,
                        &child->domain) != 0)
    {
        free(child);
        return -1;
    }

    /* Final depth: check the minimum local interior cell count across all
     * ranks.  Interior (owned) cells exclude ghost zones on neighbour faces. */
    const int gs = parent->gs;
    const int64_t local_x = child->domain.local_nx
        - (child->domain.lower_x_rank != INVALID_RANK ? gs : 0)
        - (child->domain.upper_x_rank != INVALID_RANK ? gs : 0);
    const int64_t local_y = child->domain.local_ny
        - (child->domain.lower_y_rank != INVALID_RANK ? gs : 0)
        - (child->domain.upper_y_rank != INVALID_RANK ? gs : 0);
    const int64_t local_z = child->domain.local_nz
        - (child->domain.lower_z_rank != INVALID_RANK ? gs : 0)
        - (child->domain.upper_z_rank != INVALID_RANK ? gs : 0);
    int64_t local_min = (int64_t)local_x;
    if (local_y < local_min) local_min = (int64_t)local_y;
    if (local_z < local_min) local_min = (int64_t)local_z;
    int64_t global_min;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_INT64_T, MPI_MIN,
                  child->domain.cart_comm);

    if (global_min < min_cells_per_direction)
    {
        cleanup_3d_domain(&child->domain);
        free(child);
        return 1;
    }

    ngfs_3d_allocate(parent->nvars, child);

    child->parent = parent;
    parent->child = child;

    /* Coarse levels carry the homogeneous variant of the parent's BC
     * kind: the unknown on a coarse grid is the correction e_H, whose
     * boundary condition is always homogeneous (e=0 on Dirichlet,
     * de/dn = 0 on Neumann) regardless of the inhomogeneous data the
     * user supplied at the fine level. */
    if (parent->bc) {
        child->bc = malloc(sizeof(struct bc_spec_t));
        if (child->bc)
            bc_spec_homogenize(parent->bc, child->bc);
    }

    return 0;
}

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
int ngfs_3d_create_hierarchy(struct ngfs_3d *root, int min_cells_per_direction)
{
    if (!root)
        return -1;
    struct ngfs_3d *cur = root;
    while (1)
    {
        int ret = ngfs_3d_create_child(cur, min_cells_per_direction);
        if (ret == 1)
            return 0;  /* final depth reached: hierarchy is complete */
        if (ret != 0)
            return -1; /* error */
        cur = cur->child;
    }
}

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
                   struct ngfs_2d *child, int cvar)
{
    const int64_t pnx = parent->nx;
    const double *pval = parent->vars[pvar]->val;
    double *cval = child->vars[cvar]->val;

    const int gs  = child->gs;
    const int64_t cnx = child->nx;
    const int64_t cny = child->ny;

    /* Include physical boundaries (start at 0); skip ghost zones only when an
     * MPI neighbour is present on that face. */
    const int64_t ic_lo = (child->domain.lower_x_rank != INVALID_RANK) ? gs : 0;
    const int64_t ic_hi = cnx - ((child->domain.upper_x_rank != INVALID_RANK) ? gs : 0);
    const int64_t jc_lo = (child->domain.lower_y_rank != INVALID_RANK) ? gs : 0;
    const int64_t jc_hi = cny - ((child->domain.upper_y_rank != INVALID_RANK) ? gs : 0);

    for (int64_t jc = jc_lo; jc < jc_hi; jc++)
    {
        const int64_t jp = 2 * (child->domain.local_j0 + jc) - parent->domain.local_j0;
        for (int64_t ic = ic_lo; ic < ic_hi; ic++)
        {
            const int64_t ip = 2 * (child->domain.local_i0 + ic) - parent->domain.local_i0;
            cval[ic + jc * cnx] = pval[ip + jp * pnx];
        }
    }
}

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
                     struct ngfs_2d *child, int cvar)
{
    const int64_t pnx = parent->nx;
    const double *pval = parent->vars[pvar]->val;
    double *cval = child->vars[cvar]->val;

    const int gs  = child->gs;
    const int64_t cnx = child->nx;
    const int64_t cny = child->ny;

    /* Interior only: skip physical boundary points (offset by 1) and ghost
     * zones (offset by gs) on sides that have MPI neighbours. */
    const int64_t ic_lo = (child->domain.lower_x_rank != INVALID_RANK) ? gs : 1;
    const int64_t ic_hi = cnx - ((child->domain.upper_x_rank != INVALID_RANK) ? gs : 1);
    const int64_t jc_lo = (child->domain.lower_y_rank != INVALID_RANK) ? gs : 1;
    const int64_t jc_hi = cny - ((child->domain.upper_y_rank != INVALID_RANK) ? gs : 1);

    for (int64_t jc = jc_lo; jc < jc_hi; jc++)
    {
        const int64_t jp = 2 * (child->domain.local_j0 + jc) - parent->domain.local_j0;
        for (int64_t ic = ic_lo; ic < ic_hi; ic++)
        {
            const int64_t ip = 2 * (child->domain.local_i0 + ic) - parent->domain.local_i0;
#define P2(di, dj) pval[(ip + (di)) + (jp + (dj)) * pnx]
            cval[ic + jc * cnx] = (
                4.0 * P2( 0, 0)
                + 2.0 * (P2(+1, 0) + P2(-1, 0) + P2(0,+1) + P2(0,-1))
                + (P2(+1,+1) + P2(+1,-1) + P2(-1,+1) + P2(-1,-1))
            ) / 16.0;
#undef P2
        }
    }
}

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
                   struct ngfs_3d *child, int cvar)
{
    const int64_t pnx = parent->nx;
    const int64_t pny = parent->ny;
    const double *pval = parent->vars[pvar]->val;
    double *cval = child->vars[cvar]->val;

    const int gs  = child->gs;
    const int64_t cnx = child->nx;
    const int64_t cny = child->ny;
    const int64_t cnz = child->nz;

    const int64_t ic_lo = (child->domain.lower_x_rank != INVALID_RANK) ? gs : 0;
    const int64_t ic_hi = cnx - ((child->domain.upper_x_rank != INVALID_RANK) ? gs : 0);
    const int64_t jc_lo = (child->domain.lower_y_rank != INVALID_RANK) ? gs : 0;
    const int64_t jc_hi = cny - ((child->domain.upper_y_rank != INVALID_RANK) ? gs : 0);
    const int64_t kc_lo = (child->domain.lower_z_rank != INVALID_RANK) ? gs : 0;
    const int64_t kc_hi = cnz - ((child->domain.upper_z_rank != INVALID_RANK) ? gs : 0);

    for (int64_t kc = kc_lo; kc < kc_hi; kc++)
    {
        const int64_t kp = 2 * (child->domain.local_k0 + kc) - parent->domain.local_k0;
        for (int64_t jc = jc_lo; jc < jc_hi; jc++)
        {
            const int64_t jp = 2 * (child->domain.local_j0 + jc) - parent->domain.local_j0;
            for (int64_t ic = ic_lo; ic < ic_hi; ic++)
            {
                const int64_t ip = 2 * (child->domain.local_i0 + ic) - parent->domain.local_i0;
                cval[ic + (jc + kc * cny) * cnx] =
                    pval[ip + (jp + kp * pny) * pnx];
            }
        }
    }
}

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
                     struct ngfs_3d *child, int cvar)
{
    const int64_t pnx = parent->nx;
    const int64_t pny = parent->ny;
    const double *pval = parent->vars[pvar]->val;
    double *cval = child->vars[cvar]->val;

    const int gs  = child->gs;
    const int64_t cnx = child->nx;
    const int64_t cny = child->ny;
    const int64_t cnz = child->nz;

    const int64_t ic_lo = (child->domain.lower_x_rank != INVALID_RANK) ? gs : 1;
    const int64_t ic_hi = cnx - ((child->domain.upper_x_rank != INVALID_RANK) ? gs : 1);
    const int64_t jc_lo = (child->domain.lower_y_rank != INVALID_RANK) ? gs : 1;
    const int64_t jc_hi = cny - ((child->domain.upper_y_rank != INVALID_RANK) ? gs : 1);
    const int64_t kc_lo = (child->domain.lower_z_rank != INVALID_RANK) ? gs : 1;
    const int64_t kc_hi = cnz - ((child->domain.upper_z_rank != INVALID_RANK) ? gs : 1);

    for (int64_t kc = kc_lo; kc < kc_hi; kc++)
    {
        const int64_t kp = 2 * (child->domain.local_k0 + kc) - parent->domain.local_k0;
        for (int64_t jc = jc_lo; jc < jc_hi; jc++)
        {
            const int64_t jp = 2 * (child->domain.local_j0 + jc) - parent->domain.local_j0;
            for (int64_t ic = ic_lo; ic < ic_hi; ic++)
            {
                const int64_t ip = 2 * (child->domain.local_i0 + ic) - parent->domain.local_i0;
#define P3(di, dj, dk) pval[(ip+(di)) + ((jp+(dj)) + (kp+(dk)) * pny) * pnx]
                cval[ic + (jc + kc * cny) * cnx] = (
                    8.0 * P3( 0, 0, 0)
                    + 4.0 * (P3(+1, 0, 0) + P3(-1, 0, 0)
                           + P3( 0,+1, 0) + P3( 0,-1, 0)
                           + P3( 0, 0,+1) + P3( 0, 0,-1))
                    + 2.0 * (P3(+1,+1, 0) + P3(+1,-1, 0)
                           + P3(-1,+1, 0) + P3(-1,-1, 0)
                           + P3(+1, 0,+1) + P3(+1, 0,-1)
                           + P3(-1, 0,+1) + P3(-1, 0,-1)
                           + P3( 0,+1,+1) + P3( 0,+1,-1)
                           + P3( 0,-1,+1) + P3( 0,-1,-1))
                    + 1.0 * (P3(+1,+1,+1) + P3(+1,+1,-1)
                           + P3(+1,-1,+1) + P3(+1,-1,-1)
                           + P3(-1,+1,+1) + P3(-1,+1,-1)
                           + P3(-1,-1,+1) + P3(-1,-1,-1))
                ) / 64.0;
#undef P3
            }
        }
    }
}

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
                    struct ngfs_2d *parent, int pvar)
{
    const int64_t pnx = parent->nx;
    const int64_t cnx = child->nx;
    const double *cval = child->vars[cvar]->val;
    double *pval = parent->vars[pvar]->val;

    const int gs = parent->gs;

    /* Parent owned range. */
    const int64_t pp_lo_x = (parent->domain.lower_x_rank != INVALID_RANK) ? gs : 0;
    const int64_t pp_hi_x = parent->nx - ((parent->domain.upper_x_rank != INVALID_RANK) ? gs : 0);
    const int64_t pp_lo_y = (parent->domain.lower_y_rank != INVALID_RANK) ? gs : 0;
    const int64_t pp_hi_y = parent->ny - ((parent->domain.upper_y_rank != INVALID_RANK) ? gs : 0);

    /* Loop over every owned parent point.  For each point classify it by the
     * parity of its global index (coincident / axis-midpoint / face-midpoint)
     * and read the required child values — ghost zones are valid after
     * sync_var.  Physical boundary points (global index 0 or global_n-1) are
     * never updated so that Dirichlet boundary values are preserved. */
    const int64_t gni = parent->domain.global_ni;
    const int64_t gnj = parent->domain.global_nj;
    for (int64_t jp = pp_lo_y; jp < pp_hi_y; jp++)
    {
        const int64_t pg_y = parent->domain.local_j0 + jp;
        if (pg_y == 0 || pg_y == gnj - 1) continue;

        const int y_even = (pg_y % 2 == 0);
        /* For even pg_y: coincident child jc = pg_y/2.
         * For odd  pg_y: interpolated, use child jc = (pg_y+1)/2. */
        const int64_t cg_y = (pg_y + (y_even ? 0 : 1)) / 2;
        const int64_t jc   = cg_y - child->domain.local_j0;

        for (int64_t ip = pp_lo_x; ip < pp_hi_x; ip++)
        {
            const int64_t pg_x = parent->domain.local_i0 + ip;
            if (pg_x == 0 || pg_x == gni - 1) continue;

            const int x_even = (pg_x % 2 == 0);
            const int64_t cg_x = (pg_x + (x_even ? 0 : 1)) / 2;
            const int64_t ic   = cg_x - child->domain.local_i0;

#define C2(di, dj) cval[(ic + (di)) + (jc + (dj)) * cnx]
            double update;
            if (x_even && y_even)
                update = C2(0, 0);
            else if (!x_even && y_even)
                update = 0.5  * (C2(-1, 0) + C2(0, 0));
            else if (x_even && !y_even)
                update = 0.5  * (C2(0, -1) + C2(0, 0));
            else
                update = 0.25 * (C2(-1, -1) + C2(-1, 0) + C2(0, -1) + C2(0, 0));
#undef C2

            pval[ip + jp * pnx] -= update;
        }
    }
}

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
                    struct ngfs_3d *parent, int pvar)
{
    const int64_t pnx = parent->nx;
    const int64_t pny = parent->ny;
    const int64_t cnx = child->nx;
    const int64_t cny = child->ny;
    const double *cval = child->vars[cvar]->val;
    double *pval = parent->vars[pvar]->val;

    const int gs = parent->gs;

    const int64_t pp_lo_x = (parent->domain.lower_x_rank != INVALID_RANK) ? gs : 0;
    const int64_t pp_hi_x = parent->nx - ((parent->domain.upper_x_rank != INVALID_RANK) ? gs : 0);
    const int64_t pp_lo_y = (parent->domain.lower_y_rank != INVALID_RANK) ? gs : 0;
    const int64_t pp_hi_y = parent->ny - ((parent->domain.upper_y_rank != INVALID_RANK) ? gs : 0);
    const int64_t pp_lo_z = (parent->domain.lower_z_rank != INVALID_RANK) ? gs : 0;
    const int64_t pp_hi_z = parent->nz - ((parent->domain.upper_z_rank != INVALID_RANK) ? gs : 0);

    /* Per-face skip flags.  Skip the fine-grid boundary node only on
     * Dirichlet faces (where the correction must remain zero so the
     * fine-grid Dirichlet data is preserved across coarse-grid
     * corrections); on Neumann faces the boundary node IS an unknown
     * and the correction is non-zero there.  When parent->bc is NULL
     * we fall back to the historical default (homogeneous Dirichlet
     * on every face) so operator-level unit tests keep working. */
    const bool skip_lo_x = (!parent->bc || parent->bc->face[FACE_LOWER_X].kind == BC_DIRICHLET);
    const bool skip_hi_x = (!parent->bc || parent->bc->face[FACE_UPPER_X].kind == BC_DIRICHLET);
    const bool skip_lo_y = (!parent->bc || parent->bc->face[FACE_LOWER_Y].kind == BC_DIRICHLET);
    const bool skip_hi_y = (!parent->bc || parent->bc->face[FACE_UPPER_Y].kind == BC_DIRICHLET);
    const bool skip_lo_z = (!parent->bc || parent->bc->face[FACE_LOWER_Z].kind == BC_DIRICHLET);
    const bool skip_hi_z = (!parent->bc || parent->bc->face[FACE_UPPER_Z].kind == BC_DIRICHLET);

    const int64_t gni3 = parent->domain.global_ni;
    const int64_t gnj3 = parent->domain.global_nj;
    const int64_t gnk3 = parent->domain.global_nk;
    for (int64_t kp = pp_lo_z; kp < pp_hi_z; kp++)
    {
        const int64_t pg_z = parent->domain.local_k0 + kp;
        if (skip_lo_z && pg_z == 0)         continue;
        if (skip_hi_z && pg_z == gnk3 - 1)  continue;

        const int z_even = (pg_z % 2 == 0);
        const int64_t cg_z = (pg_z + (z_even ? 0 : 1)) / 2;
        const int64_t kc   = cg_z - child->domain.local_k0;

        for (int64_t jp = pp_lo_y; jp < pp_hi_y; jp++)
        {
            const int64_t pg_y = parent->domain.local_j0 + jp;
            if (skip_lo_y && pg_y == 0)         continue;
            if (skip_hi_y && pg_y == gnj3 - 1)  continue;

            const int y_even = (pg_y % 2 == 0);
            const int64_t cg_y = (pg_y + (y_even ? 0 : 1)) / 2;
            const int64_t jc   = cg_y - child->domain.local_j0;

            for (int64_t ip = pp_lo_x; ip < pp_hi_x; ip++)
            {
                const int64_t pg_x = parent->domain.local_i0 + ip;
                if (skip_lo_x && pg_x == 0)         continue;
                if (skip_hi_x && pg_x == gni3 - 1)  continue;

                const int x_even = (pg_x % 2 == 0);
                const int64_t cg_x = (pg_x + (x_even ? 0 : 1)) / 2;
                const int64_t ic   = cg_x - child->domain.local_i0;

#define C3(di, dj, dk) cval[(ic+(di)) + ((jc+(dj)) + (kc+(dk)) * cny) * cnx]
                double update;
                if (x_even && y_even && z_even)
                    update = C3(0, 0, 0);
                else if (!x_even && y_even && z_even)
                    update = 0.5   * (C3(-1, 0, 0) + C3(0, 0, 0));
                else if (x_even && !y_even && z_even)
                    update = 0.5   * (C3(0, -1, 0) + C3(0, 0, 0));
                else if (x_even && y_even && !z_even)
                    update = 0.5   * (C3(0, 0, -1) + C3(0, 0, 0));
                else if (!x_even && !y_even && z_even)
                    update = 0.25  * (C3(-1,-1, 0) + C3(-1, 0, 0) +
                                      C3( 0,-1, 0) + C3( 0, 0, 0));
                else if (!x_even && y_even && !z_even)
                    update = 0.25  * (C3(-1, 0,-1) + C3(-1, 0, 0) +
                                      C3( 0, 0,-1) + C3( 0, 0, 0));
                else if (x_even && !y_even && !z_even)
                    update = 0.25  * (C3(0,-1,-1) + C3(0,-1, 0) +
                                      C3(0, 0,-1) + C3(0, 0, 0));
                else
                    update = 0.125 * (C3(-1,-1,-1) + C3(-1,-1, 0) +
                                      C3(-1, 0,-1) + C3(-1, 0, 0) +
                                      C3( 0,-1,-1) + C3( 0,-1, 0) +
                                      C3( 0, 0,-1) + C3( 0, 0, 0));
#undef C3

                pval[ip + (jp + kp * pny) * pnx] -= update;
            }
        }
    }
}

/******************************************************************
* Purpose: Return the depth of `gfs` in its multigrid hierarchy, where 0 is
*     the finest (root) level. Determined by counting parent pointers.
* Input Variables:
*     gfs: const struct ngfs_3d*, any level of the hierarchy
* Output Variables:
*     (none)
* Return Values and indicators of success / failure
*     int, depth (0 = finest, increasing toward coarsest)
*******************************************************************/
static int gf_level(const struct ngfs_3d *gfs)
{
    int lev = 0;
    const struct ngfs_3d *p = gfs;
    while (p->parent) { lev++; p = p->parent; }
    return lev;
}

/******************************************************************
* Purpose: Print one indented debug line to stdout on MPI rank 0 only,
*     mimicking the Python multigrid logger.debug(" * " * level + " " + msg)
*     format. Flushed immediately so output appears in order during parallel
*     runs.
* Input Variables:
*     rank: int, this MPI rank; output is suppressed unless rank == 0
*     level: int, hierarchy depth used to determine indentation
*     fmt: const char*, printf-style format string
*     ...: variadic arguments for the format string
* Output Variables:
*     (none — writes to stdout)
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
static void vcycle_debug(int rank, int level, const char *fmt, ...)
{
    if (rank != 0) return;
    for (int i = 0; i < level; i++) fputs(" * ", stdout);
    fputs(" ", stdout);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    putchar('\n');
    fflush(stdout);
}

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
                 double tol, int subcycles, int verbose)
{
    const int level = gf_level(gfs);
    const int rank  = gfs->domain.rank;

    if (verbose && level == 0 && rank == 0)
    { puts("Starting Vcycle"); fflush(stdout); }

    /* Pre-smoothing.  gauss_seidel_3d syncs VAR_SOL on exit. */
    gauss_seidel_3d(gfs, n_smooth, omega);

    /* Compute defect d = Lu - f.  calc_defect_3d syncs VAR_DEF on exit. */
    double defect_norm = calc_defect_3d(gfs);
    if (verbose)
        vcycle_debug(rank, level, "defect = %12.6e  (level %d, %ld x %ld x %ld)",
                     defect_norm, level,
                     (long)gfs->domain.global_ni - 1,
                     (long)gfs->domain.global_nj - 1,
                     (long)gfs->domain.global_nk - 1);

    if (gfs->child != NULL && defect_norm > tol)
    {
        for (int it = 0; it < subcycles; it++)
        {
            struct ngfs_3d *child = gfs->child;

            if (verbose)
                vcycle_debug(rank, level, "restrict at %ld",
                             (long)gfs->domain.global_ni);

            /* Zero child correction; apply homogeneous-Dirichlet BCs. */
            memset(child->vars[VAR_SOL]->val, 0, (size_t)child->n * sizeof(double));
            apply_bc_3d(child, VAR_SOL);

            /* Restrict parent defect → child RHS.
             * VAR_DEF on gfs is already synced by calc_defect_3d. */
            inject_var_3d(gfs, VAR_DEF, child, VAR_RHS);
            restrict_var_3d(gfs, VAR_DEF, child, VAR_RHS);

            /* Recursive V-cycle.  On return, the child has prolonged its
             * correction into gfs->vars[VAR_SOL]. */
            vcycle_3d(child, n_smooth, omega, tol, subcycles, verbose);

            /* Synchronise VAR_SOL ghost zones: prolong_var_3d modifies
             * interior fine-grid points but leaves ghost zones stale at
             * rank boundaries.  Must sync before post-smoothing reads them. */
            sync_var_3d(gfs, VAR_SOL);

            /* Post-smoothing: two passes.
             *
             *   1. SOR pass with the user-supplied omega.  When omega > 1
             *      this attacks the smooth (low-frequency) error
             *      components efficiently but, depending on omega, can
             *      have a poorer smoothing factor on the very highest
             *      frequencies than plain Gauss-Seidel.
             *   2. Plain Gauss-Seidel pass (omega = 1.0).  Red-black
             *      Gauss-Seidel has a smoothing factor of ~1/4 on the
             *      seven-point Laplacian -- independent of h -- so this
             *      pass is the one we rely on to damp the high-frequency
             *      content before the defect is restricted (or the cycle
             *      returns to its parent).
             *
             * The two-pass post-smooth is therefore deliberate: the SOR
             * pass accelerates convergence on the slowly-varying error,
             * the plain-GS pass guarantees a robust high-frequency
             * smoothing factor regardless of the user's choice of omega.
             * See the "smoothing factor of relaxation" discussion in
             * Trottenberg, Oosterlee & Schueller, Multigrid (2001), Sec. 2.1.
             *
             * gauss_seidel_3d syncs VAR_SOL on exit of each call. */
            gauss_seidel_3d(gfs, n_smooth, omega);
            gauss_seidel_3d(gfs, n_smooth, 1.0);

            defect_norm = calc_defect_3d(gfs);
            if (verbose)
                vcycle_debug(rank, level,
                             "post-smooth defect = %12.6e  (level %d)",
                             defect_norm, level);

            if (defect_norm <= tol)
                break;
        }
    }

    /* Prolong correction to the parent level.
     * VAR_SOL is synced by the last gauss_seidel_3d call above (or by the
     * pre-smooth if the child loop was skipped). */
    if (gfs->parent != NULL)
    {
        if (verbose)
            vcycle_debug(rank, level, "prolongate at %ld",
                         (long)gfs->domain.global_ni);
        prolong_var_3d(gfs, VAR_SOL, gfs->parent, VAR_SOL);
    }

    return defect_norm;
}
