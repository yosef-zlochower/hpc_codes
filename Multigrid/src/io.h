#ifndef IO_H
#define IO_H
#include "gf.h"

/******************************************************************
* Purpose: Write the local patch of a 2D grid variable to a JSON file
*     named "<dir>/<vname>_rank_<rank>.json" (or "<dir>/VAR_<var>_rank_<rank>.json"
*     if the variable has no name).  The JSON includes grid metadata
*     (nx, ny, dx, dy, x0, y0, local and global offsets, rank info,
*     ghost presence flags) and the full local data array as a nested
*     JSON array [ny][nx].
* Input Variables:
*     gfs: struct ngfs_2d*, grid function container
*     var: int, index of the variable to output
*     dir: const char*, output directory path; NULL or "" means cwd.
*         The caller is responsible for ensuring the directory exists
*         (e.g. by calling mkdir on rank 0 followed by MPI_Barrier).
* Output:
*     A JSON file written under the chosen directory.
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void output_2d_gf(struct ngfs_2d *gfs, int var, const char *dir);

/******************************************************************
* Purpose: Write the local patch of a 3D grid variable to a JSON file
*     named "<dir>/<vname>_rank_<rank>.json" (or "<dir>/VAR_<var>_rank_<rank>.json"
*     if the variable has no name).  The JSON includes grid metadata
*     (nx, ny, nz, dx, dy, dz, x0, y0, z0, local and global offsets,
*     rank info, ghost presence flags) and the full local data array
*     as a nested JSON array [nz][ny][nx].
* Input Variables:
*     gfs: struct ngfs_3d*, grid function container
*     var: int, index of the variable to output
*     dir: const char*, output directory path; NULL or "" means cwd.
*         The caller is responsible for ensuring the directory exists.
* Output:
*     A JSON file written under the chosen directory.
* Return Values and indicators of success / failure
*     (none)
*******************************************************************/
void output_3d_gf(struct ngfs_3d *gfs, int var, const char *dir);
#endif
