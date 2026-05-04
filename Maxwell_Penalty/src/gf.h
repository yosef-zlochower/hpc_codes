#ifndef GF_H
#define GF_H

#include "domain.h"
#include <stdint.h>
#include <stdlib.h>

/* struct gf holds the data for a single gridfunction */
struct gf
{
    int64_t n;            /* length of each array */
    int gs;               /* Ghost size of algorithm */
    double *restrict old; /* for old values */
    double *restrict new; /* for updated values */
    double *restrict dot; /* time derivatives  */
    double *restrict K1;  /* RK variables */
    double *restrict K2;  /* RK variables */
    double *restrict K3;  /* RK variables */
    double *restrict K4;  /* RK variables */
    char *vname;
};

/* Buffer pair used by one MPI face of one axis:  src holds data we pack
 * to send to that neighbour, dst holds data we unpack after recv. */
struct face_buffers
{
    double *src;
    double *dst;
};

/* Per-axis ghost-exchange state.  Both face_buffers pairs hold messages
 * of identical size (face_size doubles) because the lower and upper
 * neighbours on one axis exchange slabs of the same shape. */
struct comm_axis
{
    size_t               face_size;   /* per-rank buffer size in doubles */
    struct face_buffers  lower;       /* toward the lower-index neighbour */
    struct face_buffers  upper;       /* toward the upper-index neighbour */
};

/* struct ngfs holds data for all gridfunctions */
struct ngfs
{
    int n_evol_vars;           /* How many variables */
    int n_aux_vars;            /* How many variables */
    double x0;                 /* local x coordinate "origin" */
    double y0;                 /* local y coordinate "origin" */
    double z0;                 /* local z coordinate "origin" */
    double dx;                 /* x coordinate grid spacing */
    double dy;                 /* y coordinate grid spacing */
    double dz;                 /* z coordinate grid spacing */
    int64_t n_tot;             /* total number of points */
    int64_t nx;                /* local points in x (including ghosts) */
    int64_t ny;                /* local points in y (including ghosts) */
    int64_t nz;                /* local points in z (including ghosts) */
    int gs;                    /* ghost size */
    struct gf **vars;          /* pointer to nvars gf structures */
    struct gf **auxvars;       /* pointer to nvars gf structures */
    struct domain3d_st domain; /* Domain structure */

    /* Pre-allocated ghost-zone exchange buffers, one struct per axis.
     * Each struct holds the face size and the src/dst pointers for both
     * the lower and upper neighbour on that axis (4 pointers per axis,
     * 12 in total, grouped for clarity). */
    struct comm_axis comm_x;
    struct comm_axis comm_y;
    struct comm_axis comm_z;
};

int gf_allocate(int64_t n, int gs, struct gf *gptr, char *vname);
int gf_aux_allocate(int64_t n, int gs, struct gf *gptr, char *vname);
int gf_deallocate(struct gf *gptr);
int gf_aux_deallocate(struct gf *gptr);

int ngfs_allocate(int n_evol_vars, int n_aux_vars, struct ngfs *ptr);

int ngfs_deallocate(struct ngfs *ptr);

int gf_rename(struct gf *gptr, const char *name);

static inline int64_t ijk_indx(int64_t i, int64_t j, int64_t k,
                                const struct ngfs *ptr)
{
    return i + j * ptr->nx + k * ptr->nx * ptr->ny;
}
#endif
