#ifndef HDF5BINARYWRITE_HPP
#define HDF5BINARYWRITE_HPP
#include "gf.hpp"
#include <stddef.h>

/* Append (groupname, datasetname) to filename, creating it if missing.
 * `data` is a host pointer to a contiguous slab matching local_dim. The
 * caller is responsible for deep-copying the device View to a host
 * mirror first (see io.cpp). The first call to a fresh file also writes
 * the run-wide metadata block; subsequent calls reuse it. */
int BinaryWriteArray(const char *filename, const char *groupname,
                     const char *datasetname, int ndim,
                     const size_t local_dim[], const double *data,
                     const NGFS *gfs);

#endif
