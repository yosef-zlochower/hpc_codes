#ifndef GF_H
#define GF_H

#include "bc.h"
#include "domain.h"
#include <stdint.h>
#include <stdlib.h>

/* struct gf holds the data for a single gridfunction */
struct gf
{
    int64_t n;       /* length of each array */
    int gs;               /* Ghost size of algorithm */
    double *restrict val; /* array to store data */
    char *vname;          /* A name could be helpful for IO */
};

/* struct ngfs holds data for all gridfunctions */
struct ngfs_3d
{
    int nvars;                 /* How many variables */
    double x0;                 /* local x coordinate "origin" */
    double y0;                 /* local y coordinate "origin" */
    double z0;                 /* local z coordinate "origin" */
    double dx;                 /* x coordinate grid spacing */
    double dy;                 /* y coordinate grid spacing */
    double dz;                 /* z coordinate grid spacing */
    int64_t n;            /* length of arrays */
    int64_t nx;           /* length of arrays */
    int64_t ny;           /* length of arrays */
    int64_t nz;           /* length of arrays */
    int gs;                    /* ghost size */
    struct gf **vars;          /* pointer to nvars gf structures */
    struct domain3d_st domain; /* Domain structure */
    size_t buff_x_size;
    size_t buff_y_size;
    size_t buff_z_size;

    double *lower_x_src;
    double *lower_x_dst;
    double *upper_x_src;
    double *upper_x_dst;

    double *lower_y_src;
    double *lower_y_dst;
    double *upper_y_src;
    double *upper_y_dst;

    double *lower_z_src;
    double *lower_z_dst;
    double *upper_z_src;
    double *upper_z_dst;

    struct ngfs_3d *parent; /* coarser grid, NULL if this is the coarsest */
    struct ngfs_3d *child;  /* finer grid, NULL if this is the finest */

    /* Per-face boundary conditions.  Owned by this struct (allocated
     * either by the driver or by ngfs_3d_create_child) and freed in
     * ngfs_3d_deallocate.  NULL means "homogeneous Dirichlet on every
     * physical-boundary face" -- the historical default that keeps
     * code paths which never set BCs explicitly (e.g. operator-level
     * unit tests) working unchanged. */
    struct bc_spec_t *bc;
};

/******************************************************************
* Purpose: Compute the flat 1D array index for grid point (i, j, k)
*     using row-major ordering with i as the fastest-varying index.
* Input Variables:
*     gfs: struct ngfs_3d*, grid function container, used to read nx and ny
*     i: int64_t, local x-index
*     j: int64_t, local y-index
*     k: int64_t, local z-index
* Output Variables:
*     (none — pure function)
* Return Values and indicators of success / failure
*     int64_t flat index: i + (j + k * gfs->ny) * gfs->nx
*******************************************************************/
static inline int64_t gf_indx_3d(struct ngfs_3d *gfs, int64_t i, int64_t j, int64_t k)
{
    return i + (j + k * gfs->ny) * gfs->nx;
}

/******************************************************************
* Purpose: Allocate the data array and optional name string for a
*     single grid function struct. Called internally by
*     ngfs_3d_allocate for each variable slot.
* Input Variables:
*     n: int64_t, total number of grid points to allocate
*     gs: int, ghost zone width, stored for reference
*     vname: char*, optional variable name; if non-NULL a copy is
*         made; if NULL no name is stored
* Output Variables:
*     gptr: struct gf*, fields n, gs, val, and vname are written
* Return Values and indicators of success / failure
*     0 on success. Asserts (abort) if calloc fails.
*******************************************************************/
int gf_allocate(int64_t n, int gs, struct gf *gptr, char *vname);

/******************************************************************
* Purpose: Free the data array and name string held by a grid
*     function struct. Does not free the struct itself.
* Input Variables:
*     gptr: struct gf*, grid function to free
* Output Variables:
*     gptr: struct gf*, val and vname are set to NULL, n and gs
*         are zeroed
* Return Values and indicators of success / failure
*     0
*******************************************************************/
int gf_deallocate(struct gf *gptr);

/******************************************************************
* Purpose: Initialise a 3D grid function container after its domain
*     has been set up. Copies local dimensions and spacing from the
*     embedded domain struct, allocates nvars grid function slots
*     (each named "Var%d"), allocates send/receive communication
*     buffers for all six faces, and sets parent/child pointers to
*     NULL.
* Input Variables:
*     nvars: int, number of variable slots to allocate
*     ptr: struct ngfs_3d*, container whose domain field must already
*         be initialised; ptr->vars must be NULL
* Output Variables:
*     ptr: struct ngfs_3d*, all fields populated:
*         nx/ny/nz/n/gs/dx/dy/dz/x0/y0/z0, vars[], all buffer
*         pointers, parent, child
* Return Values and indicators of success / failure
*     0 on success. Asserts (abort) on allocation failure.
*******************************************************************/
int ngfs_3d_allocate(int nvars, struct ngfs_3d *ptr);

/******************************************************************
* Purpose: Free all variable slots and communication buffers owned
*     by a 3D container, and recursively free any child hierarchy
*     via ngfs_3d_free. Does not free the container struct itself or
*     its domain.
* Input Variables:
*     ptr: struct ngfs_3d*, container to deallocate; ptr->vars must
*         be non-NULL
* Output Variables:
*     ptr: struct ngfs_3d*, vars set to NULL, nvars/n/gs zeroed,
*         all buffer pointers freed
* Return Values and indicators of success / failure
*     0
*******************************************************************/
int ngfs_3d_deallocate(struct ngfs_3d *ptr);

/******************************************************************
* Purpose: Replace the name string of a grid function with a new
*     copy of `name`.
* Input Variables:
*     gptr: struct gf*, grid function to rename
*     name: const char*, new name; must be 1-512 characters
* Output Variables:
*     gptr: struct gf*, vname updated to a newly allocated copy of
*         name
* Return Values and indicators of success / failure
*     0 on success, -1 if name is empty, -2 if name exceeds 512
*     characters.
*******************************************************************/
int gf_rename(struct gf *gptr, const char *name);

/******************************************************************
* Purpose: Recursively free a dynamically-allocated 3D grid function
*     container and its entire child hierarchy. Calls
*     ngfs_3d_deallocate to release variable data, then
*     cleanup_3d_domain, then frees the struct pointer itself. Safe
*     to call with NULL.
* Input Variables:
*     gfs: struct ngfs_3d*, root of the hierarchy to free; may be
*         NULL
* Output Variables:
*     (none — the pointed-to memory is freed)
* Return Values and indicators of success / failure
*     0
*******************************************************************/
int ngfs_3d_free(struct ngfs_3d *gfs);

#endif
