#include "gf.h"
#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
int ngfs_3d_allocate(int nvars, struct ngfs_3d *ptr)
{
    assert(ptr->vars == NULL);
    ptr->nvars = nvars;
    ptr->nx = ptr->domain.local_nx;
    ptr->ny = ptr->domain.local_ny;
    ptr->nz = ptr->domain.local_nz;
    ptr->n = ptr->domain.local_nx * ptr->domain.local_ny * ptr->domain.local_nz;
    ptr->gs = ptr->domain.gs;
    ptr->dx = ptr->domain.dx;
    ptr->dy = ptr->domain.dy;
    ptr->dz = ptr->domain.dz;
    ptr->x0 = ptr->domain.global_x0 + ptr->domain.local_i0 * ptr->domain.dx;
    ptr->y0 = ptr->domain.global_y0 + ptr->domain.local_j0 * ptr->domain.dy;
    ptr->z0 = ptr->domain.global_z0 + ptr->domain.local_k0 * ptr->domain.dz;

    ptr->vars = calloc(nvars, sizeof(struct gf *));
    assert(ptr->nvars);

    char *vname = NULL;
    const size_t name_length = 20;
    for (int i = 0; i < nvars; i++)
    {
        vname = calloc(name_length, sizeof(char));
        snprintf(vname, name_length, "Var%d", i);
        ptr->vars[i] = calloc(1, sizeof(struct gf));
        gf_allocate(ptr->nx * ptr->ny * ptr->nz, ptr->gs, ptr->vars[i], vname);
        free(vname);
        vname = NULL;
    }

    /* Calculate buffer sizes */
    ptr->buff_x_size = (size_t)(ptr->gs * ptr->ny * ptr->nz);
    ptr->buff_y_size = (size_t)(ptr->nx * ptr->gs * ptr->nz);
    ptr->buff_z_size = (size_t)(ptr->nx * ptr->ny * ptr->gs);

    ptr->lower_x_src = malloc(ptr->buff_x_size * sizeof(double));
    ptr->lower_x_dst = malloc(ptr->buff_x_size * sizeof(double));
    ptr->upper_x_src = malloc(ptr->buff_x_size * sizeof(double));
    ptr->upper_x_dst = malloc(ptr->buff_x_size * sizeof(double));

    ptr->lower_y_src = malloc(ptr->buff_y_size * sizeof(double));
    ptr->lower_y_dst = malloc(ptr->buff_y_size * sizeof(double));
    ptr->upper_y_src = malloc(ptr->buff_y_size * sizeof(double));
    ptr->upper_y_dst = malloc(ptr->buff_y_size * sizeof(double));

    ptr->lower_z_src = malloc(ptr->buff_z_size * sizeof(double));
    ptr->lower_z_dst = malloc(ptr->buff_z_size * sizeof(double));
    ptr->upper_z_src = malloc(ptr->buff_z_size * sizeof(double));
    ptr->upper_z_dst = malloc(ptr->buff_z_size * sizeof(double));


    assert(ptr->lower_x_src && ptr->lower_x_dst && ptr->upper_x_src && ptr->upper_x_dst);
    assert(ptr->lower_y_src && ptr->lower_y_dst && ptr->upper_y_src && ptr->upper_y_dst);
    assert(ptr->lower_z_src && ptr->lower_z_dst && ptr->upper_z_src && ptr->upper_z_dst);

    ptr->parent = NULL;
    ptr->child  = NULL;

    return 0;
}

/******************************************************************
* Purpose: Initialise a 2D grid function container after its domain
*     has been set up. Analogous to ngfs_3d_allocate but for two
*     dimensions; allocates buffers for four faces (x and y).
* Input Variables:
*     nvars: int, number of variable slots
*     ptr: struct ngfs_2d*, container whose domain field must already
*         be initialised; ptr->vars must be NULL
* Output Variables:
*     ptr: struct ngfs_2d*, all fields populated
* Return Values and indicators of success / failure
*     0 on success. Asserts (abort) on allocation failure.
*******************************************************************/
int ngfs_2d_allocate(int nvars, struct ngfs_2d *ptr)
{
    assert(ptr->vars == NULL);
    ptr->nvars = nvars;
    ptr->nx = ptr->domain.local_nx;
    ptr->ny = ptr->domain.local_ny;
    ptr->n = ptr->domain.local_nx * ptr->domain.local_ny;
    ptr->gs = ptr->domain.gs;
    ptr->dx = ptr->domain.dx;
    ptr->dy = ptr->domain.dy;
    ptr->x0 = ptr->domain.global_x0 + ptr->domain.local_i0 * ptr->domain.dx;
    ptr->y0 = ptr->domain.global_y0 + ptr->domain.local_j0 * ptr->domain.dy;

    ptr->vars = calloc(nvars, sizeof(struct gf *));
    assert(ptr->nvars);

    char *vname = NULL;
    const size_t name_length = 20;
    for (int i = 0; i < nvars; i++)
    {
        vname = calloc(name_length, sizeof(char));
        snprintf(vname, name_length, "Var%d", i);
        ptr->vars[i] = calloc(1, sizeof(struct gf));
        gf_allocate(ptr->nx * ptr->ny, ptr->gs, ptr->vars[i], vname);
        free(vname);
        vname = NULL;
    }

    /* Calculate buffer sizes */
    ptr->buff_x_size = (size_t)(ptr->gs * ptr->ny);
    ptr->buff_y_size = (size_t)(ptr->gs * ptr->nx);

    ptr->lower_x_src = malloc(ptr->buff_x_size * sizeof(double));
    ptr->lower_x_dst = malloc(ptr->buff_x_size * sizeof(double));
    ptr->upper_x_src = malloc(ptr->buff_x_size * sizeof(double));
    ptr->upper_x_dst = malloc(ptr->buff_x_size * sizeof(double));

    ptr->lower_y_src = malloc(ptr->buff_y_size * sizeof(double));
    ptr->lower_y_dst = malloc(ptr->buff_y_size * sizeof(double));
    ptr->upper_y_src = malloc(ptr->buff_y_size * sizeof(double));
    ptr->upper_y_dst = malloc(ptr->buff_y_size * sizeof(double));

    assert(ptr->lower_x_src && ptr->lower_x_dst && ptr->upper_x_src && ptr->upper_x_dst);
    assert(ptr->lower_y_src && ptr->lower_y_dst && ptr->upper_y_src && ptr->upper_y_dst);

    ptr->parent = NULL;
    ptr->child  = NULL;

    return 0;
}

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
int ngfs_3d_deallocate(struct ngfs_3d *ptr)
{
    if (ptr->child)
        ngfs_3d_free(ptr->child);

    assert(ptr->vars);

    for (int i = 0; i < ptr->nvars; i++)
    {
        gf_deallocate(ptr->vars[i]);
        free(ptr->vars[i]);
        ptr->vars[i] = NULL;
    }
    free(ptr->vars);
    ptr->nvars = 0;
    ptr->n = 0;
    ptr->gs = 0;
    ptr->vars = NULL;

    free(ptr->lower_x_src);
    free(ptr->lower_x_dst);
    free(ptr->upper_x_src);
    free(ptr->upper_x_dst);

    free(ptr->lower_y_src);
    free(ptr->lower_y_dst);
    free(ptr->upper_y_src);
    free(ptr->upper_y_dst);

    free(ptr->lower_z_src);
    free(ptr->lower_z_dst);
    free(ptr->upper_z_src);
    free(ptr->upper_z_dst);

    return 0;
}

/******************************************************************
* Purpose: Free all variable slots and communication buffers owned
*     by a 2D container, and recursively free any child hierarchy
*     via ngfs_2d_free. Does not free the container struct itself or
*     its domain.
* Input Variables:
*     ptr: struct ngfs_2d*, container to deallocate; ptr->vars must
*         be non-NULL
* Output Variables:
*     ptr: struct ngfs_2d*, vars set to NULL, nvars/n/gs zeroed,
*         all buffer pointers freed
* Return Values and indicators of success / failure
*     0
*******************************************************************/
int ngfs_2d_deallocate(struct ngfs_2d *ptr)
{
    if (ptr->child)
        ngfs_2d_free(ptr->child);

    assert(ptr->vars);

    for (int i = 0; i < ptr->nvars; i++)
    {
        gf_deallocate(ptr->vars[i]);
        free(ptr->vars[i]);
        ptr->vars[i] = NULL;
    }
    free(ptr->vars);
    ptr->nvars = 0;
    ptr->n = 0;
    ptr->gs = 0;
    ptr->vars = NULL;

    free(ptr->lower_x_src);
    free(ptr->lower_x_dst);
    free(ptr->upper_x_src);
    free(ptr->upper_x_dst);

    free(ptr->lower_y_src);
    free(ptr->lower_y_dst);
    free(ptr->upper_y_src);
    free(ptr->upper_y_dst);

    return 0;
}

/******************************************************************
* Purpose: Allocate the data array and optional name string for a
*     single grid function struct. Called internally by
*     ngfs_2d/3d_allocate for each variable slot.
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
int gf_allocate(int64_t n, int gs, struct gf *gptr, char *vname)
{
    gptr->n = n;
    gptr->gs = gs;
    gptr->val = calloc((size_t)n, sizeof(double));
    assert(gptr->val);

    if (vname)
    {
        const size_t slen = strlen(vname) + 1;
        gptr->vname = calloc(slen, sizeof(char));

        assert(gptr->vname);

        strncpy(gptr->vname, vname, slen);
        gptr->vname[slen-1] = '\0';
    }
    else
    {
        gptr->vname = NULL;
    }

    return 0;
}

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
int gf_rename(struct gf *gptr, const char *name)
{
    const size_t s_len = strlen(name);

    if (s_len < 1)
    {
        fprintf(stderr, "Invalid variable name\n");
        return -1;
    }

    if (s_len > 512)
    {
        fprintf(stderr, "Invalid variable name\n");
        return -2;
    }

    if (gptr->vname)
    {
        free(gptr->vname);
        gptr->vname = NULL;
    }

    char *newname = calloc(s_len + 1, sizeof(char));
    strncpy(newname, name, s_len + 1);
    gptr->vname = newname;

    return 0;
}

/******************************************************************
* Purpose: Recursively free a dynamically-allocated 2D grid function
*     container and its entire child hierarchy. Calls
*     ngfs_2d_deallocate to release variable data, then
*     cleanup_2d_domain, then frees the struct pointer itself. Safe
*     to call with NULL.
* Input Variables:
*     gfs: struct ngfs_2d*, root of the hierarchy to free; may be
*         NULL
* Output Variables:
*     (none — the pointed-to memory is freed)
* Return Values and indicators of success / failure
*     0
*******************************************************************/
int ngfs_2d_free(struct ngfs_2d *gfs)
{
    if (!gfs)
        return 0;
    if (gfs->child)
        ngfs_2d_free(gfs->child);
    if (gfs->parent)
        gfs->parent->child = NULL;
    if (gfs->vars)
        ngfs_2d_deallocate(gfs);
    cleanup_2d_domain(&gfs->domain);
    free(gfs);
    return 0;
}

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
int ngfs_3d_free(struct ngfs_3d *gfs)
{
    if (!gfs)
        return 0;
    if (gfs->child)
        ngfs_3d_free(gfs->child);
    if (gfs->parent)
        gfs->parent->child = NULL;
    if (gfs->vars)
        ngfs_3d_deallocate(gfs);
    cleanup_3d_domain(&gfs->domain);
    free(gfs);
    return 0;
}

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
int gf_deallocate(struct gf *gptr)
{
    free(gptr->val);
    gptr->val = NULL;

    if (gptr->vname)
    {
        free(gptr->vname);
        gptr->vname = NULL;
    }
    gptr->n = 0;
    gptr->gs = 0;

    return 0;
}
