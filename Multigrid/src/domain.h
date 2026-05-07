#ifndef DOMAIN_H
#define DOMAIN_H
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <mpi.h>

#define INVALID_RANK -1
struct bbox_st
{
    int lower;
    int upper;
};

struct domain1d_st
{
    int rank;
    struct bbox_st bbox;
    int64_t n;
    int lower_size;
    int lower_rank;
    int upper_size;
    int upper_rank;
    int64_t local0;
    int gs;
};

struct domain2d_st
{
    int rank;         /* this processes rank */
    int mpi_size;     /* Total number of mpi processes */
    int gs;           /* number of ghost zones */
    int64_t local_nx; /* number of points in x direction of local grid */
    int64_t local_ny; /* number of points in y direction of local grid */
    int64_t local_i0; /* global grid i index of the local grid i=0 */
    int64_t local_j0; /* global grid j index of the local grid j=0 */
    int lower_x_rank; /* rank of lower x neighboor */
    int upper_x_rank; /* rank of upper x neighboor */
    int lower_y_rank; /* rank of lower y neighboor */
    int upper_y_rank; /* rank of upper y neighboor */
    MPI_Comm cart_comm; /* Cartesian communicator */
    int64_t global_nx_cells; /* number of cells in x of the global grid */
    int64_t global_ny_cells; /* number of cells in y of the global grid */
    double global_x0;  /* global minimum x */
    double global_y0;  /* global minimum y */
    double dx;        /* grid spacing */
    double dy;        /* grid spacing */
};

struct domain3d_st
{
    int rank;         /* this processes rank */
    int mpi_size;     /* Total number of mpi processes */
    int gs;           /* number of ghost zones */
    int64_t local_nx; /* number of points in x direction of local grid */
    int64_t local_ny; /* number of points in y direction of local grid */
    int64_t local_nz; /* number of points in z direction of local grid */
    int64_t local_i0; /* global grid i index of the local grid i=0 */
    int64_t local_j0; /* global grid j index of the local grid j=0 */
    int64_t local_k0; /* global grid k index of the local grid k=0 */
    int lower_x_rank; /* rank of lower x neighboor */
    int upper_x_rank; /* rank of upper x neighboor */
    int lower_y_rank; /* rank of lower y neighboor */
    int upper_y_rank; /* rank of upper y neighboor */
    int lower_z_rank; /* rank of lower z neighboor */
    int upper_z_rank; /* rank of upper z neighboor */
    MPI_Comm cart_comm; /* Cartesian communicator */
    int64_t global_nx_cells; /* number of cells in x of the global grid */
    int64_t global_ny_cells; /* number of cells in y of the global grid */
    int64_t global_nz_cells; /* number of cells in z of the global grid */
    /* Per-face Neumann flags (Phase 2 of CellCentred_plan).  An axis is
     * cell-centred (extra ghost cell at each end, origin shifted by
     * -h/2) iff *both* its ends are Neumann; hybrid axes (D-N or N-D)
     * stay vertex-centred in Phase 2 and use the legacy in-stencil
     * mirror at the Neumann end (Phase 3 will move them to cell-centred
     * with a half-step at the Dirichlet vertex). */
    bool neumann_lower_x, neumann_upper_x;
    bool neumann_lower_y, neumann_upper_y;
    bool neumann_lower_z, neumann_upper_z;
    double global_x0;  /* global minimum x */
    double global_y0;  /* global minimum y */
    double global_z0;  /* global minimum z */
    double dx;        /* grid spacing */
    double dy;        /* grid spacing */
    double dz;        /* grid spacing */
};

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
                    const int64_t nglobal, const int gs,
                    struct domain1d_st *domain);

/******************************************************************
* Purpose: Set up a 2D Cartesian MPI domain decomposition. Creates an MPI
*     Cartesian communicator, queries neighbour ranks via MPI_Cart_shift,
*     calls setup_1d_domain for each axis, and fills all fields of the
*     domain2d_st struct.
* Input Variables:
*     nx_cpu: int, processes in x
*     ny_cpu: int, processes in y
*     rank: int, MPI_COMM_WORLD rank of this process
*     global_nx_cells: int64_t, global cell count in x
*     global_ny_cells: int64_t, global cell count in y
*     gs: int, ghost zone width
*     global_x0: double, global domain origin in x
*     global_y0: double, global domain origin in y
*     dx: double, grid spacing in x
*     dy: double, grid spacing in y
* Output Variables:
*     domain: struct domain2d_st*, fully initialised
* Return Values and indicators of success / failure
*     0 on success, -1 if MPI_Cart_create returns MPI_COMM_NULL
*******************************************************************/
int setup_2d_domain(const int nx_cpu, const int ny_cpu, const int rank,
                    const int64_t global_nx_cells, const int64_t global_ny_cells,
                    const int gs, const double global_x0, const double global_y0,
                    const double dx, const double dy,
                    struct domain2d_st *domain);

/******************************************************************
* Purpose: Set up a 3D Cartesian MPI domain decomposition. Analogous to
*     setup_2d_domain but for three axes. MPI Cartesian dims are ordered
*     {nz_cpu, ny_cpu, nx_cpu} so that
*     rank = rank_x + rank_y*nx_cpu + rank_z*nx_cpu*ny_cpu.
* Input Variables:
*     nx_cpu: int, processes in x
*     ny_cpu: int, processes in y
*     nz_cpu: int, processes in z
*     rank: int, MPI_COMM_WORLD rank of this process
*     global_nx_cells: int64_t, global cell count in x
*     global_ny_cells: int64_t, global cell count in y
*     global_nz_cells: int64_t, global cell count in z
*     neumann_face: const bool[6], per-face Neumann flag, indexed
*         {LOWER_X, UPPER_X, LOWER_Y, UPPER_Y, LOWER_Z, UPPER_Z}
*         (matches face_id_t from bc.h).  An axis with *both* ends
*         marked Neumann becomes cell-centred: extra ghost cell at
*         each end (point count = N+2 instead of N+1) and origin
*         shifted by -h/2.  Other axes (DD or hybrid) keep the
*         vertex-centred layout, point count = N+1.  Pass NULL for
*         the legacy all-Dirichlet behaviour (used by operator-level
*         unit tests).
*     gs: int, ghost zone width
*     global_x0: double, global origin in x (user lower bound a_x; the
*         half-cell shift for cell-centred axes is applied internally)
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
                    struct domain3d_st *domain);

/******************************************************************
* Purpose: Free the MPI Cartesian communicator associated with a 2D domain.
* Input Variables:
*     domain: struct domain2d_st*, domain to clean up
* Output Variables:
*     (none)
* Return Values and indicators of success / failure
*     0
*******************************************************************/
int cleanup_2d_domain(struct domain2d_st *domain);
/******************************************************************
* Purpose: Free the MPI Cartesian communicator associated with a 3D domain.
* Input Variables:
*     domain: struct domain3d_st*, domain to clean up
* Output Variables:
*     (none)
* Return Values and indicators of success / failure
*     0
*******************************************************************/
int cleanup_3d_domain(struct domain3d_st *domain);
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
                           size_t topology[]);

#endif
