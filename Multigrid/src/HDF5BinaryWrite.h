#ifndef HDF5BINARYWRITE_H
#define HDF5BINARYWRITE_H

#include "gf.h"

/* Write the local patch of a 3D variable to the per-rank HDF5 file at
 * `filename`, as a dataset named `/<datasetname>`.
 *
 * On the first call (file does not yet exist) the file is created and
 * a `/metadata` group is written with grid dimensions, spacings,
 * origins, ghost-zone count, per-axis Neumann flags, and per-axis
 * has-neighbour flags.  Subsequent calls append additional datasets
 * to the same file without re-writing metadata.  This lets the same
 * rank emit multiple variables (each with its own `/<vname>` slot)
 * via repeated calls.
 *
 * `local_dim` is the dataset shape in HDF5 (slowest-axis first):
 * `{ nz, ny, nx }`.  `data` must point to a contiguous nz*ny*nx
 * array in i-fastest order (matching gf_indx_3d).
 *
 * Returns 0 on success, non-zero on any HDF5 error. */
int BinaryWriteArray_3d(const char *filename, const char *datasetname,
                        const size_t local_dim[3], const double *data,
                        const struct ngfs_3d *gfs);

/* 2D variant.  `local_dim = { ny, nx }`. */
int BinaryWriteArray_2d(const char *filename, const char *datasetname,
                        const size_t local_dim[2], const double *data,
                        const struct ngfs_2d *gfs);

#endif /* HDF5BINARYWRITE_H */
