#include "io.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "HDF5BinaryWrite.h"
#include "gf.h"

/* Build the per-rank output filename used by output_3d_gf.
 * Filename pattern:
 *
 *     <dir>/rank_<R>.h5
 *
 * Each rank writes to its own file; multiple variables (multiple
 * calls to output_*_gf for the same rank) are appended into the same
 * file as separate datasets under `/`. */
static void build_rank_filename(char *buf, size_t buflen,
                                const char *dir, int rank)
{
    const char *prefix = (dir && dir[0]) ? dir : ".";
    snprintf(buf, buflen, "%s/rank_%d.h5", prefix, rank);
}

/* Dataset name for variable `var` inside the per-rank HDF5 file.
 * Prefers the gf's own vname (always set to "VarN" by gf_allocate) but
 * falls back to "VAR_<var>" if that slot is somehow null. */
static const char *dataset_name(const char *vname, int var, char *fallback,
                                size_t fbuf_len)
{
    if (vname && vname[0]) return vname;
    snprintf(fallback, fbuf_len, "VAR_%d", var);
    return fallback;
}

void output_3d_gf(struct ngfs_3d *gfs, int var, const char *dir)
{
    char filename[256];
    char fallback[64];
    build_rank_filename(filename, sizeof(filename), dir, gfs->domain.rank);
    const char *dsname = dataset_name(gfs->vars[var]->vname, var,
                                      fallback, sizeof(fallback));

    /* HDF5 dataset dims are slowest-axis first: { nz, ny, nx }.
     * The buffer is i-fastest (gf_indx_3d order); HDF5 reads it in
     * row-major / C order so the dim list above places nz outermost. */
    const size_t local_dim[3] = { (size_t)gfs->nz, (size_t)gfs->ny,
                                  (size_t)gfs->nx };

    if (BinaryWriteArray_3d(filename, dsname, local_dim,
                            gfs->vars[var]->val, gfs) != 0)
    {
        fprintf(stderr, "rank %d: failed to write '%s' to '%s'\n",
                gfs->domain.rank, dsname, filename);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}
