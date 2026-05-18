#include "gf.h"
#include "alloc_check.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Allocate / free the four ghost-zone exchange buffers for one axis. */
static void comm_axis_alloc(struct comm_axis *ax, size_t face_size)
{
    ax->face_size = face_size;
    ax->lower.src = calloc(face_size, sizeof(double));
    CHECK_ALLOC(ax->lower.src, "comm axis lower.src");
    ax->lower.dst = calloc(face_size, sizeof(double));
    CHECK_ALLOC(ax->lower.dst, "comm axis lower.dst");
    ax->upper.src = calloc(face_size, sizeof(double));
    CHECK_ALLOC(ax->upper.src, "comm axis upper.src");
    ax->upper.dst = calloc(face_size, sizeof(double));
    CHECK_ALLOC(ax->upper.dst, "comm axis upper.dst");
}

static void comm_axis_free(struct comm_axis *ax)
{
    free(ax->lower.src); ax->lower.src = NULL;
    free(ax->lower.dst); ax->lower.dst = NULL;
    free(ax->upper.src); ax->upper.src = NULL;
    free(ax->upper.dst); ax->upper.dst = NULL;
    ax->face_size = 0;
}

/* Allocate gridfunctions. External routines call this function */
int ngfs_allocate(int n_evol_vars, int n_aux_vars, struct ngfs *ptr)
{
    assert(ptr->vars == NULL);
    assert(ptr->auxvars == NULL);
    ptr->n_evol_vars = n_evol_vars;
    ptr->n_aux_vars = n_aux_vars;
    ptr->nx = ptr->domain.nx;
    ptr->ny = ptr->domain.ny;
    ptr->nz = ptr->domain.nz;
    ptr->n_tot = ptr->domain.nx * ptr->domain.ny * ptr->domain.nz;
    ptr->gs = ptr->domain.gs;
    ptr->dx = ptr->domain.dx;
    ptr->dy = ptr->domain.dy;
    ptr->dz = ptr->domain.dz;
    ptr->x0 = ptr->domain.global_x0 + ptr->domain.local_i0 * ptr->domain.dx;
    ptr->y0 = ptr->domain.global_y0 + ptr->domain.local_j0 * ptr->domain.dy;
    ptr->z0 = ptr->domain.global_z0 + ptr->domain.local_k0 * ptr->domain.dz;

    /* Pre-allocate communication buffers for ghost-zone exchange.
     * Each buffer holds data for max(n_evol_vars, n_aux_vars) variables
     * packed into a single MPI message per face. */
    const size_t maxvars = n_evol_vars > n_aux_vars ? n_evol_vars : n_aux_vars;
    comm_axis_alloc(&ptr->comm_x, ptr->gs * ptr->ny * ptr->nz * maxvars);
    comm_axis_alloc(&ptr->comm_y, ptr->nx * ptr->gs * ptr->nz * maxvars);
    comm_axis_alloc(&ptr->comm_z, ptr->nx * ptr->ny * ptr->gs * maxvars);

    ptr->vars = calloc(n_evol_vars + n_aux_vars, sizeof(struct gf *));
    CHECK_ALLOC(ptr->vars, "ngfs vars table");

    char *vname = NULL;
    const size_t name_length = 20;
    for (int i = 0; i < n_evol_vars; i++)
    {
        vname = calloc(name_length, sizeof(char));
        CHECK_ALLOC(vname, "evolved gf name");
        snprintf(vname, name_length, "Var%d", i);
        ptr->vars[i] = calloc(1, sizeof(struct gf));
        CHECK_ALLOC(ptr->vars[i], "evolved gf struct");
        gf_allocate(ptr->nx * ptr->ny * ptr->nz, ptr->gs, ptr->vars[i], vname);
        vname = NULL;
    }

    ptr->auxvars = ptr->vars + n_evol_vars;

    for (int i = 0; i < n_aux_vars; i++)
    {
        vname = calloc(name_length, sizeof(char));
        CHECK_ALLOC(vname, "aux gf name");
        snprintf(vname, name_length, "Aux%d", i);
        ptr->auxvars[i] = calloc(1, sizeof(struct gf));
        CHECK_ALLOC(ptr->auxvars[i], "aux gf struct");
        gf_aux_allocate(ptr->nx * ptr->ny * ptr->nz, ptr->gs, ptr->auxvars[i],
                        vname);
        vname = NULL;
    }

    return 0;
}

/* free gridfunctions */
int ngfs_deallocate(struct ngfs *ptr)
{
    assert(ptr->vars);

    for (int i = 0; i < ptr->n_evol_vars; i++)
    {
        gf_deallocate(ptr->vars[i]);
        free(ptr->vars[i]);
        ptr->vars[i] = NULL;
    }
    for (int i = 0; i < ptr->n_aux_vars; i++)
    {
        gf_aux_deallocate(ptr->vars[i + ptr->n_evol_vars]);
        free(ptr->vars[i + ptr->n_evol_vars]);
        ptr->vars[i + ptr->n_evol_vars] = NULL;
    }
    free(ptr->vars);
    ptr->n_evol_vars = 0;
    ptr->n_aux_vars = 0;

    ptr->n_tot = 0;
    ptr->gs = 0;
    ptr->vars = NULL;
    ptr->auxvars = NULL;

    /* Free ghost-zone exchange buffers */
    comm_axis_free(&ptr->comm_x);
    comm_axis_free(&ptr->comm_y);
    comm_axis_free(&ptr->comm_z);

    return 0;
}

/* Allocate individual gridfunction. External routines generally don't call this
 * function */
int gf_allocate(int64_t n, int gs, struct gf *gptr, char *vname)
{
    gptr->n = n;
    gptr->gs = gs;
    gptr->old = calloc(n, sizeof(double));
    CHECK_ALLOC(gptr->old, "evolved gf->old");
    gptr->new = calloc(n, sizeof(double));
    CHECK_ALLOC(gptr->new, "evolved gf->new");
    gptr->K1 = calloc(n, sizeof(double));
    CHECK_ALLOC(gptr->K1, "evolved gf->K1");
    gptr->K2 = calloc(n, sizeof(double));
    CHECK_ALLOC(gptr->K2, "evolved gf->K2");
    gptr->K3 = calloc(n, sizeof(double));
    CHECK_ALLOC(gptr->K3, "evolved gf->K3");
    gptr->K4 = calloc(n, sizeof(double));
    CHECK_ALLOC(gptr->K4, "evolved gf->K4");
    gptr->dot = NULL;

    gptr->vname = vname;

    return 0;
}
int gf_aux_allocate(int64_t n, int gs, struct gf *gptr, char *vname)
{
    gptr->n = n;
    gptr->gs = gs;
    gptr->new = calloc(n, sizeof(double));
    CHECK_ALLOC(gptr->new, "aux gf->new");
    gptr->old = NULL;
    gptr->dot = NULL;
    gptr->K1 = NULL;
    gptr->K2 = NULL;
    gptr->K3 = NULL;
    gptr->K4 = NULL;
    gptr->vname = vname;
    gptr->dot = gptr->new; /* aux vars alias dot → new so that
                            * sync_vars() can sync their data using
                            * the same ->dot interface as evolved vars */
    return 0;
}

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
    CHECK_ALLOC(newname, "gf_rename newname");
    strncpy(newname, name, s_len + 1);
    gptr->vname = newname;

    return 0;
}

/* deallocate individual gridfunction. External routines generally don't call
 * this function */
int gf_deallocate(struct gf *gptr)
{
    free(gptr->old);
    free(gptr->new);
    free(gptr->K1);
    free(gptr->K2);
    free(gptr->K3);
    free(gptr->K4);

    if (gptr->vname)
    {
        free(gptr->vname);
        gptr->vname = NULL;
    }
    gptr->n = 0;
    gptr->gs = 0;
    gptr->old = NULL;
    gptr->new = NULL;
    gptr->dot = NULL;
    gptr->K1 = NULL;
    gptr->K2 = NULL;
    gptr->K3 = NULL;
    gptr->K4 = NULL;

    return 0;
}

int gf_aux_deallocate(struct gf *gptr)
{
    free(gptr->new);

    if (gptr->vname)
    {
        free(gptr->vname);
        gptr->vname = NULL;
    }
    gptr->n = 0;
    gptr->gs = 0;
    gptr->old = NULL;
    gptr->new = NULL;
    gptr->dot = NULL;
    gptr->K1 = NULL;
    gptr->K2 = NULL;
    gptr->K3 = NULL;
    gptr->K4 = NULL;

    return 0;
}
