#ifndef IO_H
#define IO_H
#include "gf.h"

/******************************************************************
* Purpose: Append the local patch of a 3D grid variable as a dataset
*     to this rank's output file `<dir>/rank_<R>.h5` with dims
*     `{ nz, ny, nx }` (slowest axis first).  Buffer layout is
*     i-fastest (matches gf_indx_3d).  The dataset name is the
*     variable's vname slot (always "VarN" for variables allocated
*     through ngfs_3d_allocate; falls back to "VAR_<var>" if vname is
*     null).  The first call from a given rank also writes a
*     `/metadata` group with grid dims, spacings, origins, ghost-zone
*     count, per-face Neumann flags, and per-face has-neighbour
*     flags; subsequent calls reuse the existing metadata and just
*     append their dataset under `/`.
* Input Variables:
*     gfs: struct ngfs_3d*, grid function container
*     var: int, index of the variable to output
*     dir: const char*, output directory path; NULL or "" means cwd.
*         The caller is responsible for ensuring the directory exists.
* Output:
*     A per-rank HDF5 file under the chosen directory; aborts on any
*     HDF5 error via MPI_Abort.
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void output_3d_gf(struct ngfs_3d *gfs, int var, const char *dir);
#endif
