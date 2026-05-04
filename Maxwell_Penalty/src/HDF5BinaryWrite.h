#ifndef HDF5BINARYWRITE_H
#define HDF5BINARYWRITE_H
#include "gf.h"
#include "domain.h"
int BinaryWriteArray(const char *filename, const char *groupname,
                     const char *datasetname, const int ndim,
                     const size_t local_dim[], const double *data,
                     const struct ngfs *gfs);
#endif
