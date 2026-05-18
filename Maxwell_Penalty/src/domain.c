#include "domain.h"
#include "mpi_check.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>

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
int setup_1d_domain(const int ncpu_per_direction, const int direction_rank,
                    const int64_t nglobal, const int gs,
                    struct domain1d_st *domain,
                    const enum bdry_type bdry)
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
    for (int i = 0; i < under && i < rank; i++)
    {
        domain->local0 += (ln + 1);
    }
    for (int i = under; i < rank; i++)
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

    /* add ghostzones except at bdry. EDIT: for periodic BC, all
     * processes get boundary points */
    if (domain->local0 > 0)
    {
        domain->bbox.lower = 0;
        domain->lower_rank = rank - 1;
        domain->local0 -= gs;
        ln += gs;
        domain->lower_size = gs;
    }
    else
    {
        if (bdry == PERIODIC)
        {
            domain->bbox.lower = 0;
            domain->lower_rank = nprocs - 1;
            domain->local0 -= gs;
            ln += gs;
            domain->lower_size = gs;
        }
        else
        {
            domain->lower_rank = INVALID_RANK;
            domain->lower_size = 0;
        }
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
        if (bdry == PERIODIC)
        {
            domain->bbox.upper = 0;
            domain->upper_rank = 0;
            ln += gs;
            domain->upper_size = gs;
        }
        else
        {
            domain->upper_rank = INVALID_RANK;
            domain->upper_size = 0;
        }
    }

    /* At this point, ln now contains the number of points, including
     * all ghost zones and local0 contains the index of the first point
     * (which is typically a ghost point).
     */
    domain->n = ln;
    domain->gs = gs;
    return 0;
}

int setup_3d_domain(const int nx_cpu, const int ny_cpu, const int nz_cpu,
                    const int rank, const int64_t nx_global,
                    const int64_t ny_global, const int64_t nz_global,
                    const int gs, const double global_x0,
                    const double global_y0, const double global_z0,
                    const double global_xn, const double global_yn,
                    const double global_zn, struct domain3d_st *domain,
                    const enum bdry_type bdry_x, const enum bdry_type bdry_y,
                    const enum bdry_type bdry_z)
{
    struct domain1d_st domain_1d;

    domain->global_ni = nx_global + (bdry_x == PERIODIC ? 0 : 1);
    domain->global_nj = ny_global + (bdry_y == PERIODIC ? 0 : 1);
    domain->global_nk = nz_global + (bdry_z == PERIODIC ? 0 : 1);
    domain->gs = gs;

    domain->global_x0 = global_x0;
    domain->global_y0 = global_y0;
    domain->global_z0 = global_z0;

    domain->global_xn = global_xn;
    domain->global_yn = global_yn;
    domain->global_zn = global_zn;

    domain->dx = (global_xn - global_x0) / nx_global;
    domain->dy = (global_yn - global_y0) / ny_global;
    domain->dz = (global_zn - global_z0) / nz_global;

    domain->mpi_size = nx_cpu * ny_cpu * nz_cpu;

    /* Create MPI Cartesian topology.
     * MPI uses row-major ordering, so dims are {nz, ny, nx} to get
     * rank = rank_x + rank_y * nx_cpu + rank_z * nx_cpu * ny_cpu. */
    int dims[3] = { nz_cpu, ny_cpu, nx_cpu };
    int periods[3] = { bdry_z == PERIODIC, bdry_y == PERIODIC,
                       bdry_x == PERIODIC };
    int reorder = 0;
    MPI_ERROR(MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder,
                              &domain->cart_comm));

    if (domain->cart_comm == MPI_COMM_NULL)
    {
        return -1;
    }

    /* Get this rank's coordinates in the Cartesian grid */
    int coords[3];
    MPI_ERROR(MPI_Cart_coords(domain->cart_comm, rank, 3, coords));
    const int rank_z = coords[0];
    const int rank_y = coords[1];
    const int rank_x = coords[2];

    /* Get neighbor ranks via MPI_Cart_shift.
     * Dimension 0 = z, 1 = y, 2 = x (matches dims[] ordering). */
    int lower_x, upper_x, lower_y, upper_y, lower_z, upper_z;
    MPI_ERROR(MPI_Cart_shift(domain->cart_comm, 2, 1, &lower_x, &upper_x));
    MPI_ERROR(MPI_Cart_shift(domain->cart_comm, 1, 1, &lower_y, &upper_y));
    MPI_ERROR(MPI_Cart_shift(domain->cart_comm, 0, 1, &lower_z, &upper_z));

    /* setup_1d_domain computes local sizes and ghost zone counts.
     * We still pass bdry_type so it correctly adds ghost zones at
     * periodic boundaries. */
    domain->rank = rank;

    setup_1d_domain(nx_cpu, rank_x, domain->global_ni, gs, &domain_1d, bdry_x);
    domain->nx = domain_1d.n;
    domain->local_i0 = domain_1d.local0;

    /* Use MPI_Cart_shift results for neighbor ranks.
     * MPI_PROC_NULL → INVALID_RANK for a consistent sentinel. */
    domain->lower_x_rank = (lower_x == MPI_PROC_NULL) ? INVALID_RANK : lower_x;
    domain->upper_x_rank = (upper_x == MPI_PROC_NULL) ? INVALID_RANK : upper_x;

    /* Derive bbox from neighbor ranks: physical boundary iff no neighbor */
    domain->bbox.x.lower = (domain->lower_x_rank == INVALID_RANK) ? 1 : 0;
    domain->bbox.x.upper = (domain->upper_x_rank == INVALID_RANK) ? 1 : 0;

    setup_1d_domain(ny_cpu, rank_y, domain->global_nj, gs, &domain_1d, bdry_y);
    domain->ny = domain_1d.n;
    domain->local_j0 = domain_1d.local0;

    domain->lower_y_rank = (lower_y == MPI_PROC_NULL) ? INVALID_RANK : lower_y;
    domain->upper_y_rank = (upper_y == MPI_PROC_NULL) ? INVALID_RANK : upper_y;

    domain->bbox.y.lower = (domain->lower_y_rank == INVALID_RANK) ? 1 : 0;
    domain->bbox.y.upper = (domain->upper_y_rank == INVALID_RANK) ? 1 : 0;

    setup_1d_domain(nz_cpu, rank_z, domain->global_nk, gs, &domain_1d, bdry_z);
    domain->nz = domain_1d.n;
    domain->local_k0 = domain_1d.local0;

    domain->lower_z_rank = (lower_z == MPI_PROC_NULL) ? INVALID_RANK : lower_z;
    domain->upper_z_rank = (upper_z == MPI_PROC_NULL) ? INVALID_RANK : upper_z;

    domain->bbox.z.lower = (domain->lower_z_rank == INVALID_RANK) ? 1 : 0;
    domain->bbox.z.upper = (domain->upper_z_rank == INVALID_RANK) ? 1 : 0;

    return 0;
}

int cleanup_3d_domain(struct domain3d_st *domain)
{
    if (domain->cart_comm != MPI_COMM_NULL)
    {
        MPI_ERROR(MPI_Comm_free(&domain->cart_comm));
        domain->cart_comm = MPI_COMM_NULL;
    }
    return 0;
}

#define MAX_NDIMS 16
#define MAX_FACTORS 64

int automatic_topology(int ndim, const size_t dims[], size_t nproc,
                       size_t topology[])
{
    if (ndim <= 0 || nproc == 0)
    {
        fprintf(stderr,
                "Error: invalid values for ndim %d and / or nproc %lu.\n", ndim,
                nproc);
        return -1;
    }

    if (ndim > MAX_NDIMS)
    {
        fprintf(stderr, "Error: ndim %d exceeds limit.\n", ndim);
        return -1;
    }

    /* Factorise nproc into primes (stored smallest-to-largest) */
    size_t factors[MAX_FACTORS];
    int num_factors = 0;
    size_t temp_n = nproc;

    while (temp_n % 2 == 0)
    {
        factors[num_factors++] = 2;
        temp_n /= 2;
    }

    size_t limit = (size_t)sqrt((double)temp_n);
    for (size_t s = 3; s <= limit; s += 2)
    {
        while (temp_n % s == 0)
        {
            if (num_factors >= MAX_FACTORS)
                return -1;
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

    /* Greedy distribution: assign each prime factor (largest first)
     * to the currently longest dimension. */
    double current_dims[MAX_NDIMS];
    for (int i = 0; i < ndim; i++)
    {
        current_dims[i] = (double)dims[i];
        topology[i] = 1;
    }

    for (int k = num_factors - 1; k >= 0; k--)
    {
        size_t factor = factors[k];

        int best_dim = -1;
        double max_len = -1.0;

        for (int i = 0; i < ndim; i++)
        {
            /* >= so that ties prefer the last (highest) dimension */
            if (current_dims[i] >= max_len)
            {
                max_len = current_dims[i];
                best_dim = i;
            }
        }

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

#undef MAX_NDIMS
#undef MAX_FACTORS
