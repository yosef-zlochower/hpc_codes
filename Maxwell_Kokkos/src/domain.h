#ifndef DOMAIN_H
#define DOMAIN_H
#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>

/* Sentinel value for neighbour ranks that do not exist (physical boundary) */
#define INVALID_RANK -1

/* Per-axis flag: 1 = this face is a physical boundary, 0 = MPI neighbour */
struct bbox_st
{
    int lower;
    int upper;
};

struct bbox3D_st
{
    struct bbox_st x;
    struct bbox_st y;
    struct bbox_st z;
};

/* 1D domain decomposition along a single axis */
struct domain1d_st
{
    int rank;            /* this rank's coordinate along this axis */
    struct bbox_st bbox; /* physical boundary flags */
    int64_t n;           /* local point count (interior + ghost zones) */
    int lower_size;      /* number of lower ghost points (0 or gs) */
    int lower_rank;      /* 1D rank of lower neighbour (or INVALID_RANK) */
    int upper_size;      /* number of upper ghost points (0 or gs) */
    int upper_rank;      /* 1D rank of upper neighbour (or INVALID_RANK) */
    int64_t local0;      /* global index of this rank's first local point */
    int gs;              /* ghost zone width */
};

/* 3D Cartesian domain decomposition across MPI ranks */
struct domain3d_st
{
    int rank;            /* MPI rank of this process */
    int mpi_size;        /* total number of MPI processes */
    int gs;              /* ghost zone width */
    int64_t nx;          /* local x points (interior + ghost zones) */
    int64_t ny;          /* local y points (interior + ghost zones) */
    int64_t nz;          /* local z points (interior + ghost zones) */
    int64_t local_i0;    /* global i-index of this rank's first point */
    int64_t local_j0;    /* global j-index of this rank's first point */
    int64_t local_k0;    /* global k-index of this rank's first point */
    int lower_x_rank;    /* MPI rank of lower x neighbour (or INVALID_RANK) */
    int upper_x_rank;    /* MPI rank of upper x neighbour (or INVALID_RANK) */
    int lower_y_rank;    /* MPI rank of lower y neighbour (or INVALID_RANK) */
    int upper_y_rank;    /* MPI rank of upper y neighbour (or INVALID_RANK) */
    int lower_z_rank;    /* MPI rank of lower z neighbour (or INVALID_RANK) */
    int upper_z_rank;    /* MPI rank of upper z neighbour (or INVALID_RANK) */
    int64_t global_ni;   /* global point count in x */
    int64_t global_nj;   /* global point count in y */
    int64_t global_nk;   /* global point count in z */
    double global_x0;    /* domain lower x boundary */
    double global_y0;    /* domain lower y boundary */
    double global_z0;    /* domain lower z boundary */
    double global_xn;    /* domain upper x boundary */
    double global_yn;    /* domain upper y boundary */
    double global_zn;    /* domain upper z boundary */
    double dx;           /* grid spacing in x */
    double dy;           /* grid spacing in y */
    double dz;           /* grid spacing in z */
    struct bbox3D_st bbox; /* physical boundary flags (derived from ranks) */
    MPI_Comm cart_comm;    /* MPI Cartesian communicator */
};

enum bdry_type
{
    NON_PERIODIC = 0,
    PERIODIC
};

/* Distribute nglobal points along one axis across ncpu_per_direction ranks.
 * Fills domain with local size, ghost zone counts, and neighbour ranks. */
int setup_1d_domain(const int ncpu_per_direction, const int direction_rank,
                    const int64_t nglobal, const int gs,
                    struct domain1d_st *domain,
                    const enum bdry_type periodic);

/* Create a 3D MPI Cartesian domain decomposition.  Builds an MPI Cartesian
 * communicator, queries neighbour ranks via MPI_Cart_shift, and calls
 * setup_1d_domain for each axis. */
int setup_3d_domain(const int nx_cpu, const int ny_cpu, const int nz_cpu,
                    const int rank, const int64_t nx_global,
                    const int64_t ny_global, const int64_t nz_global,
                    const int gs, const double global_x0,
                    const double global_y0, const double global_z0,
                    const double global_xn, const double global_yn,
                    const double global_zn, struct domain3d_st *domain,
                    const enum bdry_type periodic_x,
                    const enum bdry_type periodic_y,
                    const enum bdry_type periodic_z);

/* Free the MPI Cartesian communicator in domain->cart_comm. */
int cleanup_3d_domain(struct domain3d_st *domain);

/* Partition nproc MPI processes across ndim dimensions using a greedy
 * prime-factorisation algorithm.  The longest dimension receives each
 * prime factor in turn, minimising surface-area-to-volume ratio.
 *
 *   ndim      -- number of spatial dimensions (1..16)
 *   dims[]    -- global grid size in each dimension, length ndim
 *   nproc     -- total number of MPI processes
 *   topology[] -- output: processes per dimension, length ndim
 *                 (must be pre-allocated by caller)
 *
 * Returns 0 on success, -1 on invalid input. */
int automatic_topology(int ndim, const size_t dims[], size_t nproc,
                       size_t topology[]);

#endif
