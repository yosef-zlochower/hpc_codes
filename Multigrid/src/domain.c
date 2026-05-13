#include "domain.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

/******************************************************************
* Purpose: Partition nproc MPI processes across ndim dimensions using a
*     greedy prime-factorisation algorithm. The longest dimension receives
*     each prime factor in turn.
* Input Variables:
*     ndim: int, number of spatial dimensions
*     dims: size_t[], size of global grid in each dimension, length ndim
*     nproc: size_t, total number of MPI processes
* Output Variables:
*     topology: size_t[], number of processes per dimension, length ndim
*               (must be pre-allocated by caller)
* Return Values and indicators of success / failure
*     0 on success, -1 on invalid input (ndim <= 0, nproc == 0, ndim > 16)
*******************************************************************/
int automatic_topology(int ndim, const size_t dims[], size_t nproc,
                       size_t topology[])
{

#ifdef MAX_DIMS
#undef MAX_DIMS
#endif
#ifdef MAX_FACTORS
#undef MAX_FACTORS
#endif
#define MAX_NDIMS 16
#define MAX_FACTORS 64 // can't be more than 64 factors in a 64 bit integer.

    if (ndim <= 0 || nproc == 0)
    {
        fprintf(stderr,
                "Error: invalid values for ndim %d and / or nproc %lu.\n", ndim,
                nproc);
        return -1;
    }

    if (ndim > MAX_NDIMS) // Reasonable hard limit for physics domains
    {
        fprintf(stderr, "Error: ndim %d exceeds limit.\n", ndim);
        return -1;
    }

    /* 64 is sufficient to store all prime factors (with multiplicity) for a 64
     * bit integer
     */
    size_t factors[MAX_FACTORS];
    int num_factors = 0;
    size_t temp_n = nproc;

    // Handle 2 separately to allow stepping by 2 later
    while (temp_n % 2 == 0)
    {
        factors[num_factors++] = 2;
        temp_n /= 2;
    }

    // Handle odd factors
    size_t limit = (size_t)sqrt((double)temp_n);
    for (size_t s = 3; s <= limit; s += 2)
    {
        while (temp_n % s == 0)
        {
            if (num_factors >= MAX_FACTORS)
                return -1; // Safety break
            factors[num_factors++] = s;
            temp_n /= s;
        }
    }

    if (temp_n > 1)
    {
        if (num_factors >= MAX_FACTORS)
            return -1;
        factors[num_factors++] = temp_n;
    }

    // Setup greedy distribution
    double current_dims[MAX_NDIMS]; // double precision because dim % factor
                                    // will be non-zero, generically.
    for (int i = 0; i < ndim; i++)
    {
        current_dims[i] = (double)dims[i];
        topology[i] = 1;
    }

    /* Distribute factors (Process usually largest factors first for better
     * balance).  To minimize surface area, we usually want to apply factors to
     * the currently largest dimension. The order of application matters
     * slightly. Processing from largest factor to smallest is generally
     * preferred.

     * Iterate backwards through our list (which is Small -> Large)
     * so we process Large -> Small.
     */
    for (int k = num_factors - 1; k >= 0; k--)
    {
        size_t factor = factors[k];

        int best_dim = -1;
        double max_len = -1.0;

        // Find dimension with largest current length
        for (int i = 0; i < ndim; i++)
        {
            /* we use >= here because splitting under z preferred to y,
             * splitting y preferred to x.
             */
            if (current_dims[i] >= max_len)
            {
                max_len = current_dims[i];
                best_dim = i;
            }
        }

        // Apply factor
        topology[best_dim] *= factor;
        current_dims[best_dim] /= (double)factor;
    }

    for (int i = 0; i < ndim; i++)
    {
        if (topology[i] > INT_MAX)
        {
            fprintf(stderr, "Error: Topology dim %d exceeds MPI int limits.\n",
                    i);
            return -1;
        }
    }

    return 0;
}

#undef MAX_DIMS
#undef MAX_FACTORS

/* Consider a grid of the form (11 points):
 *   o o o o o o o o o o o
 *   that will be distributed among 3 MPI processes (x = ghostzone)
 *
 *   o o o o x x   (4 non-ghostzones, 6 total points)
 *       x x o o o o x x  (4 non-ghostzones, 8 total points )
 *               x x o o o  (3 non-ghostzones, 5 total points)
 *
 * That is 11/3 = 3, and 11 - 3*3 = 2, meaning two MPI processes
 * need to have an extra point.
 * Proc 0: local0 = 0, localn = 6.
 * Proc 1: local0 = 2, localn = 8.
 * Proc 2: local0 = 7, localn = 5.
 */
/******************************************************************
* Purpose: Compute the local extent and ghost-zone layout for one axis of
*     the domain decomposition. Distributes nglobal interior points across
*     ncpu_per_direction ranks, giving floor(nglobal/ncpu) points to each
*     rank and one extra point to the first (nglobal mod ncpu) ranks. Ghost
*     zones of width gs are added on sides that have MPI neighbours.
* Input Variables:
*     ncpu_per_direction: int, number of processes along this axis
*     direction_rank: int, this rank's coordinate along this axis (0-based)
*     nglobal: int64_t, total number of grid points along this axis
*     gs: int, ghost zone width
* Output Variables:
*     domain: struct domain1d_st*, filled with local0, n, lower/upper ranks
*             and sizes, gs, bbox
* Return Values and indicators of success / failure
*     0 on success
*******************************************************************/
int setup_1d_domain(const int ncpu_per_direction, const int direction_rank,
                    const int64_t nglobal, const int gs, struct domain1d_st *domain)
{
    const int rank = direction_rank;
    const int nprocs = ncpu_per_direction;

    domain->rank = rank;
    /* ln starts as the approximate number of non-ghostzones per proces
     */
    int64_t ln = nglobal / nprocs;

    /* Since ln * nprocs < nglobal, there are processes that will need
     * to have more points. The number of such processes is stored in
     * "under"
     */
    const int64_t under = nglobal - ln * nprocs;

    domain->bbox.lower = 1;
    domain->bbox.upper = 1;
    domain->local0 = 0;

    /* Here we try to find the starting index (on the global grid) of
     * the first non-ghostzone on our local grid. We do this by summing
     * up the non-ghostzone points owned by all processes with rank less
     * than our rank. All ranks < under will have an extra point.
     */
    for (int64_t i = 0; i < under && i < rank; i++)
    {
        domain->local0 += (ln + 1);
    }
    for (int64_t i = under; i < rank; i++)
    {
        domain->local0 += ln;
    }

    /* If our rank < under, then we also have an extra point */
    if (rank < under)
    {
        ln++;
    }

    domain->lower_rank = INVALID_RANK;
    domain->upper_rank = INVALID_RANK;
    domain->lower_size = 0;
    domain->upper_size = 0;

    /* add ghostzones except at bdry */
    if (domain->local0 > 0)
    {
        domain->bbox.lower = 0;
        domain->lower_rank = rank - 1;
        domain->local0 -= gs;
        domain->lower_size = gs;
        ln += gs;
    }
    else
    {
        domain->lower_rank = INVALID_RANK;
    }
    if (domain->local0 + ln < nglobal)
    {
        domain->bbox.upper = 0;
        domain->upper_rank = rank + 1;
        ln += gs;
        domain->upper_size = gs;
    }
    else
    {
        domain->upper_rank = INVALID_RANK;
    }

    /* At this point, ln now contains the number of points, including
     * all ghost zones and local0 contains the index of the first point
     * (which is typically a ghost point).
     */
    domain->n = ln;
    domain->gs = gs;
    return 0;
}

/******************************************************************
* Purpose: Set up a 3D Cartesian MPI domain decomposition. MPI Cartesian
*     dims are ordered {nz_cpu, ny_cpu, nx_cpu} so that
*     rank = rank_x + rank_y*nx_cpu + rank_z*nx_cpu*ny_cpu.
* Input Variables:
*     nx_cpu: int, processes in x
*     ny_cpu: int, processes in y
*     nz_cpu: int, processes in z
*     rank: int, MPI_COMM_WORLD rank of this process
*     nx_global: int64_t, global grid size in x
*     ny_global: int64_t, global grid size in y
*     nz_global: int64_t, global grid size in z
*     gs: int, ghost zone width
*     global_x0: double, global origin in x
*     global_y0: double, global origin in y
*     global_z0: double, global origin in z
*     dx: double, grid spacing in x
*     dy: double, grid spacing in y
*     dz: double, grid spacing in z
* Output Variables:
*     domain: struct domain3d_st*, fully initialised
* Return Values and indicators of success / failure
*     0 on success, -1 if MPI_Cart_create returns MPI_COMM_NULL
*******************************************************************/
int setup_3d_domain(const int nx_cpu, const int ny_cpu, const int nz_cpu,
                    const int rank, const int64_t global_nx_cells,
                    const int64_t global_ny_cells, const int64_t global_nz_cells,
                    const bool neumann_face[6],
                    const int gs, const double global_x0,
                    const double global_y0, const double global_z0,
                    const double dx, const double dy, const double dz,
                    struct domain3d_st *domain)
{
    struct domain1d_st domain_1d;

    /* Per-face Neumann flags (NULL = all Dirichlet, the legacy
     * test-suite behaviour). */
    const bool n_lo_x = neumann_face ? neumann_face[0] : false;
    const bool n_hi_x = neumann_face ? neumann_face[1] : false;
    const bool n_lo_y = neumann_face ? neumann_face[2] : false;
    const bool n_hi_y = neumann_face ? neumann_face[3] : false;
    const bool n_lo_z = neumann_face ? neumann_face[4] : false;
    const bool n_hi_z = neumann_face ? neumann_face[5] : false;

    /* Phase 3 layout: any axis with at least one Neumann face is
     * cell-centred -- it has N+2 grid points (N interior cells plus
     * one extra point at each end).  On a pure D-D axis there is no
     * extra point at either end and the layout stays at N+1.
     *
     * Pure NN: ghost at each end (lowest at a-h/2, highest at b+h/2).
     * D-N    : Dirichlet vertex at lower end (at exactly a),
     *          Neumann ghost at upper end (at b+h/2).  The gap
     *          between vertex and the first interior cell centre is
     *          h/2 -- the unavoidable half-step (CellCentred_plan.md
     *          sec. 2.2).
     * N-D    : mirror of D-N.
     *
     * Origin shift: gfs->x0 is set so that the standard formula
     * x = gfs->x0 + i*dx gives the correct physical coordinate at
     * every *interior cell centre and at every Neumann ghost*.
     * Dirichlet vertices on hybrid axes are at coordinate
     * gfs->x0 + (vertex_index +- 1/2)*dx -- apply_bc_3d handles the
     * +-h/2 offset explicitly. */
    const bool axis_x_has_n = n_lo_x || n_hi_x;
    const bool axis_y_has_n = n_lo_y || n_hi_y;
    const bool axis_z_has_n = n_lo_z || n_hi_z;

    domain->global_nx_cells = global_nx_cells;
    domain->global_ny_cells = global_ny_cells;
    domain->global_nz_cells = global_nz_cells;
    const int64_t nx_points = global_nx_cells + (axis_x_has_n ? 2 : 1);
    const int64_t ny_points = global_ny_cells + (axis_y_has_n ? 2 : 1);
    const int64_t nz_points = global_nz_cells + (axis_z_has_n ? 2 : 1);

    domain->neumann_lower_x = n_lo_x;
    domain->neumann_upper_x = n_hi_x;
    domain->neumann_lower_y = n_lo_y;
    domain->neumann_upper_y = n_hi_y;
    domain->neumann_lower_z = n_lo_z;
    domain->neumann_upper_z = n_hi_z;
    domain->gs = gs;

    /* Origin shifted by -h/2 whenever the axis has any Neumann face
     * (so that the standard x = x0 + i*dx formula gives correct cell
     * coordinates).  Dirichlet vertices are at i=0 (D-N) or i=N+1
     * (N-D) and need an explicit +-h/2 fix-up in apply_bc. */
    domain->global_x0 = axis_x_has_n ? global_x0 - dx / 2.0 : global_x0;
    domain->global_y0 = axis_y_has_n ? global_y0 - dy / 2.0 : global_y0;
    domain->global_z0 = axis_z_has_n ? global_z0 - dz / 2.0 : global_z0;

    domain->dx = dx;
    domain->dy = dy;
    domain->dz = dz;

    domain->mpi_size = nx_cpu * ny_cpu * nz_cpu;

    /* Create Cartesian topology */
    /* Note: MPI uses row-major ordering, so dims are {nz_cpu, ny_cpu, nx_cpu}
     * to match the old rank formula: rank = rank_x + rank_y * nx_cpu + rank_z *
     * nx_cpu * ny_cpu */
    int dims[3] = { nz_cpu, ny_cpu, nx_cpu };
    int periods[3] = { 0, 0, 0 }; /* non-periodic boundaries */
    int reorder = 0;              /* preserve rank ordering */
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);

    if (cart_comm == MPI_COMM_NULL)
    {
        return -1; /* Error creating Cartesian communicator */
    }

    domain->cart_comm = cart_comm;

    /* Get coordinates in Cartesian grid */
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    const int rank_z = coords[0];
    const int rank_y = coords[1];
    const int rank_x = coords[2];

    /* Get neighbor ranks using MPI_Cart_shift */
    int lower_x, upper_x, lower_y, upper_y, lower_z, upper_z;
    /* Direction 0 is Z, direction 1 is Y, direction 2 is X */
    MPI_Cart_shift(cart_comm, 2, 1, &lower_x, &upper_x);
    MPI_Cart_shift(cart_comm, 1, 1, &lower_y, &upper_y);
    MPI_Cart_shift(cart_comm, 0, 1, &lower_z, &upper_z);

    /* Setup 1D domains as before */
    setup_1d_domain(nx_cpu, rank_x, nx_points, gs, &domain_1d);

    domain->rank = rank;
    domain->local_nx = domain_1d.n;
    domain->local_i0 = domain_1d.local0;

    /* Convert MPI_PROC_NULL to INVALID_RANK for backward compatibility */
    domain->lower_x_rank = (lower_x == MPI_PROC_NULL) ? INVALID_RANK : lower_x;
    domain->upper_x_rank = (upper_x == MPI_PROC_NULL) ? INVALID_RANK : upper_x;

    setup_1d_domain(ny_cpu, rank_y, ny_points, gs, &domain_1d);
    domain->local_ny = domain_1d.n;
    domain->local_j0 = domain_1d.local0;

    domain->lower_y_rank = (lower_y == MPI_PROC_NULL) ? INVALID_RANK : lower_y;
    domain->upper_y_rank = (upper_y == MPI_PROC_NULL) ? INVALID_RANK : upper_y;

    setup_1d_domain(nz_cpu, rank_z, nz_points, gs, &domain_1d);
    domain->local_nz = domain_1d.n;
    domain->local_k0 = domain_1d.local0;

    domain->lower_z_rank = (lower_z == MPI_PROC_NULL) ? INVALID_RANK : lower_z;
    domain->upper_z_rank = (upper_z == MPI_PROC_NULL) ? INVALID_RANK : upper_z;

    return 0;
}

/******************************************************************
* Purpose: Free the MPI Cartesian communicator associated with a 3D domain.
* Input Variables:
*     domain: struct domain3d_st*, domain to clean up
* Output Variables:
*     (none)
* Return Values and indicators of success / failure
*     0
*******************************************************************/
int cleanup_3d_domain(struct domain3d_st *domain)
{
    if (domain->cart_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&domain->cart_comm);
        domain->cart_comm = MPI_COMM_NULL;
    }
    return 0;
}
