#include "gf.h"
#include "domain.h"
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

/******************************************************************
* Purpose: Write the local patch of a 2D grid variable to a JSON file named
*     "<vname>_rank_<rank>.json" (or "VAR_<var>_rank_<rank>.json" if the
*     variable has no name). The JSON includes grid metadata (nx, ny, dx, dy,
*     x0, y0, local and global offsets, rank info, ghost presence flags) and
*     the full local data array as a nested JSON array [ny][nx].
* Input Variables:
*     gfs: struct ngfs_2d*, grid function container
*     var: int, index of the variable to output
* Output:
*     A JSON file written to the current working directory.
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void output_2d_gf(struct ngfs_2d *gfs, int var, const char *dir)
{
    char fname[256];
    const char *prefix = (dir && dir[0]) ? dir : ".";
    if (gfs->vars[var]->vname)
    {
        snprintf(fname, sizeof(fname), "%s/%s_rank_%d.json",
                 prefix, gfs->vars[var]->vname, gfs->domain.rank);
    }
    else
    {
        snprintf(fname, sizeof(fname), "%s/VAR_%d_rank_%d.json",
                 prefix, var, gfs->domain.rank);
    }
    FILE *f = fopen(fname, "w");
    if (!f)
    {
        fprintf(stderr, "rank %d: cannot open '%s' for writing: %s\n",
                gfs->domain.rank, fname, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    fprintf(f, "{\n");
    fprintf(f, "    \"nx\": %ld,\n", (long)gfs->nx);
    fprintf(f, "    \"ny\": %ld,\n", (long)gfs->ny);
    fprintf(f, "    \"dx\": %20.16e,\n", gfs->dx);
    fprintf(f, "    \"dy\": %20.16e,\n", gfs->dy);
    fprintf(f, "    \"x0\": %20.16e,\n", gfs->x0);
    fprintf(f, "    \"y0\": %20.16e,\n", gfs->y0);
    fprintf(f, "    \"local_i0\": %ld,\n", (long)gfs->domain.local_i0);
    fprintf(f, "    \"local_j0\": %ld,\n", (long)gfs->domain.local_j0);
    fprintf(f, "    \"global_ni\": %ld,\n", (long)gfs->domain.global_ni);
    fprintf(f, "    \"global_nj\": %ld,\n", (long)gfs->domain.global_nj);
    fprintf(f, "    \"global_x0\": %20.16e,\n", gfs->domain.global_x0);
    fprintf(f, "    \"global_y0\": %20.16e,\n", gfs->domain.global_y0);
    fprintf(f, "    \"rank\": %d,\n", gfs->domain.rank);
    fprintf(f, "    \"mpi_size\": %d,\n", gfs->domain.mpi_size);
    fprintf(f, "    \"gs\": %d,\n", gfs->gs);
    fprintf(f, "    \"lower_x_ghost\": %s,\n",
            (gfs->domain.lower_x_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"upper_x_ghost\": %s,\n",
            (gfs->domain.upper_x_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"lower_y_ghost\": %s,\n",
            (gfs->domain.lower_y_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"upper_y_ghost\": %s,\n",
            (gfs->domain.upper_y_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"data\": [ ");
    for (int64_t j = 0; j < gfs->ny; j++)
    {
        const char *jend = (j != gfs->ny - 1) ? "," : "";
        fprintf(f, "[\n");
        for (int64_t i = 0; i < gfs->nx; i++)
        {
            const char *iend = (i != gfs->nx - 1) ? "," : "";
            fprintf(f, "%20.16e%s", gfs->vars[var]->val[gf_indx_2d(gfs, i, j)], iend);
        }
        fprintf(f, "]%s", jend);
    }
    fprintf(f, "]\n}\n");
    fclose(f);
}

/******************************************************************
* Purpose: Write the local patch of a 3D grid variable to a JSON file named
*     "<vname>_rank_<rank>.json" (or "VAR_<var>_rank_<rank>.json" if the
*     variable has no name). The JSON includes grid metadata (nx, ny, nz, dx,
*     dy, dz, x0, y0, z0, local and global offsets, rank info, ghost presence
*     flags) and the full local data array as a nested JSON array [nz][ny][nx].
* Input Variables:
*     gfs: struct ngfs_3d*, grid function container
*     var: int, index of the variable to output
* Output:
*     A JSON file written to the current working directory.
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void output_3d_gf(struct ngfs_3d *gfs, int var, const char *dir)
{
    char fname[256];
    const char *prefix = (dir && dir[0]) ? dir : ".";
    if (gfs->vars[var]->vname)
    {
        snprintf(fname, sizeof(fname), "%s/%s_rank_%d.json",
                 prefix, gfs->vars[var]->vname, gfs->domain.rank);
    }
    else
    {
        snprintf(fname, sizeof(fname), "%s/VAR_%d_rank_%d.json",
                 prefix, var, gfs->domain.rank);
    }

    FILE *f = fopen(fname, "w");
    if (!f)
    {
        fprintf(stderr, "rank %d: cannot open '%s' for writing: %s\n",
                gfs->domain.rank, fname, strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    fprintf(f, "{\n");
    fprintf(f, "    \"nx\": %ld,\n", (long)gfs->nx);
    fprintf(f, "    \"ny\": %ld,\n", (long)gfs->ny);
    fprintf(f, "    \"nz\": %ld,\n", (long)gfs->nz);
    fprintf(f, "    \"dx\": %20.16e,\n", gfs->dx);
    fprintf(f, "    \"dy\": %20.16e,\n", gfs->dy);
    fprintf(f, "    \"dz\": %20.16e,\n", gfs->dz);
    fprintf(f, "    \"x0\": %20.16e,\n", gfs->x0);
    fprintf(f, "    \"y0\": %20.16e,\n", gfs->y0);
    fprintf(f, "    \"z0\": %20.16e,\n", gfs->z0);
    fprintf(f, "    \"local_i0\": %ld,\n", (long)gfs->domain.local_i0);
    fprintf(f, "    \"local_j0\": %ld,\n", (long)gfs->domain.local_j0);
    fprintf(f, "    \"local_k0\": %ld,\n", (long)gfs->domain.local_k0);
    fprintf(f, "    \"global_ni\": %ld,\n", (long)gfs->domain.global_ni);
    fprintf(f, "    \"global_nj\": %ld,\n", (long)gfs->domain.global_nj);
    fprintf(f, "    \"global_nk\": %ld,\n", (long)gfs->domain.global_nk);
    fprintf(f, "    \"global_x0\": %20.16e,\n", gfs->domain.global_x0);
    fprintf(f, "    \"global_y0\": %20.16e,\n", gfs->domain.global_y0);
    fprintf(f, "    \"global_z0\": %20.16e,\n", gfs->domain.global_z0);
    fprintf(f, "    \"rank\": %d,\n", gfs->domain.rank);
    fprintf(f, "    \"mpi_size\": %d,\n", gfs->domain.mpi_size);
    fprintf(f, "    \"gs\": %d,\n", gfs->gs);
    fprintf(f, "    \"lower_x_ghost\": %s,\n",
            (gfs->domain.lower_x_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"upper_x_ghost\": %s,\n",
            (gfs->domain.upper_x_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"lower_y_ghost\": %s,\n",
            (gfs->domain.lower_y_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"upper_y_ghost\": %s,\n",
            (gfs->domain.upper_y_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"lower_z_ghost\": %s,\n",
            (gfs->domain.lower_z_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"upper_z_ghost\": %s,\n",
            (gfs->domain.upper_z_rank != INVALID_RANK) ? "true" : "false");
    fprintf(f, "    \"data\": [ ");
    for (int64_t k = 0; k < gfs->nz; k++)
    {
        const char *kend = (k != gfs->nz - 1) ? "," : "";
        fprintf(f, "[\n");
        for (int64_t j = 0; j < gfs->ny; j++)
        {
            const char *jend = (j != gfs->ny - 1) ? "," : "";
            fprintf(f, "[\n");
            for (int64_t i = 0; i < gfs->nx; i++)
            {
                const char *iend = (i != gfs->nx - 1) ? "," : "";
                fprintf(f, "%20.16e%s",
                        gfs->vars[var]->val[gf_indx_3d(gfs, i, j, k)], iend);
            }
            fprintf(f, "]%s", jend);
        }
        fprintf(f, "]%s", kend);
    }
    fprintf(f, "]\n}\n");
    fclose(f);
}
