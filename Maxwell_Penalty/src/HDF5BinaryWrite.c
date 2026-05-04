#include "HDF5BinaryWrite.h"
#include <assert.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <stdlib.h>
#include <string.h>

#include "gf.h"
#include "domain.h"

#define HDF5_ERROR(fn_call)                                                    \
    do                                                                         \
    {                                                                          \
        int _error_code = fn_call;                                             \
                                                                               \
        if (_error_code < 0)                                                   \
        {                                                                      \
            fprintf(stderr, "HDF5 call '%s' returned error code %d", #fn_call, \
                    _error_code);                                              \
            error_count++;                                                     \
        }                                                                      \
    } while (0)

#include <hdf5.h>
#include <hdf5_hl.h>

static void create_string_attribute(hid_t id, const char *name, const char *val)
{
    int error_count = 0;
    hid_t dataspace_id;
    hid_t attribute_id;
    hid_t datatype;
    hsize_t oD1 = 1;

    HDF5_ERROR(datatype = H5Tcopy(H5T_C_S1));
    HDF5_ERROR(H5Tset_size(datatype, strlen(val) + 1));
    HDF5_ERROR(dataspace_id = H5Screate_simple(1, &oD1, NULL));
    HDF5_ERROR(attribute_id = H5Acreate(id, name, datatype, dataspace_id,
                                        H5P_DEFAULT, H5P_DEFAULT));
    HDF5_ERROR(H5Awrite(attribute_id, datatype, val));
    HDF5_ERROR(H5Aclose(attribute_id));
    HDF5_ERROR(H5Sclose(dataspace_id));
    HDF5_ERROR(H5Tclose(datatype));
}

static void create_int_attribute(hid_t id, const char *name, const int val)
{
    int error_count = 0;
    hid_t dataspace_id;
    hid_t attribute_id;
    hsize_t oD1 = 1;

    HDF5_ERROR(dataspace_id = H5Screate_simple(1, &oD1, NULL));
    HDF5_ERROR(attribute_id = H5Acreate(id, name, H5T_NATIVE_INT, dataspace_id,
                                        H5P_DEFAULT, H5P_DEFAULT));
    HDF5_ERROR(H5Awrite(attribute_id, H5T_NATIVE_INT, &val));
    HDF5_ERROR(H5Aclose(attribute_id));
    HDF5_ERROR(H5Sclose(dataspace_id));
}

static void create_double_attribute(hid_t id, const char *name,
                                    const double val)
{
    int error_count = 0;
    hid_t dataspace_id;
    hid_t attribute_id;
    hsize_t oD1 = 1;

    HDF5_ERROR(dataspace_id = H5Screate_simple(1, &oD1, NULL));
    HDF5_ERROR(attribute_id =
                   H5Acreate(id, name, H5T_NATIVE_DOUBLE, dataspace_id,
                             H5P_DEFAULT, H5P_DEFAULT));
    HDF5_ERROR(H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &val));
    HDF5_ERROR(H5Aclose(attribute_id));
    HDF5_ERROR(H5Sclose(dataspace_id));
}

static void write_metadata(hid_t file_id, const struct ngfs *gfs)
{
    int error_count = 0;
    const int local_ni = (int)gfs->nx;
    const int local_nj = (int)gfs->ny;
    const int local_nk = (int)gfs->nz;

    const int local_i0 = (int)gfs->domain.local_i0;
    const int local_j0 = (int)gfs->domain.local_j0;
    const int local_k0 = (int)gfs->domain.local_k0;

    const int global_ni = (int)gfs->domain.global_ni;
    const int global_nj = (int)gfs->domain.global_nj;
    const int global_nk = (int)gfs->domain.global_nk;

    const double local_x0 = gfs->x0;
    const double local_y0 = gfs->y0;
    const double local_z0 = gfs->z0;

    const double global_x0 = gfs->domain.global_x0;
    const double global_y0 = gfs->domain.global_y0;
    const double global_z0 = gfs->domain.global_z0;

    const double dx = gfs->dx;
    const double dy = gfs->dy;
    const double dz = gfs->dz;

    hid_t group_id;
    char metaname[] = "/metadata";
    HDF5_ERROR(group_id = H5Gcreate(file_id, metaname, H5P_DEFAULT, H5P_DEFAULT,
                                    H5P_DEFAULT));

    create_int_attribute(group_id, "ghost_zones", gfs->gs);
    create_int_attribute(group_id, "mpi_size", gfs->domain.mpi_size);

    create_int_attribute(group_id, "n_evol_vars", gfs->n_evol_vars);
    create_int_attribute(group_id, "n_aux_vars", gfs->n_aux_vars);

    create_int_attribute(group_id, "local_ni", local_ni);
    create_int_attribute(group_id, "local_nj", local_nj);
    create_int_attribute(group_id, "local_nk", local_nk);

    create_int_attribute(group_id, "global_ni", global_ni);
    create_int_attribute(group_id, "global_nj", global_nj);
    create_int_attribute(group_id, "global_nk", global_nk);

    create_int_attribute(group_id, "local_i0", local_i0);
    create_int_attribute(group_id, "local_j0", local_j0);
    create_int_attribute(group_id, "local_k0", local_k0);

    create_double_attribute(group_id, "local_x0", local_x0);
    create_double_attribute(group_id, "local_y0", local_y0);
    create_double_attribute(group_id, "local_z0", local_z0);

    create_double_attribute(group_id, "global_x0", global_x0);
    create_double_attribute(group_id, "global_y0", global_y0);
    create_double_attribute(group_id, "global_z0", global_z0);

    create_double_attribute(group_id, "dx", dx);
    create_double_attribute(group_id, "dy", dy);
    create_double_attribute(group_id, "dz", dz);

    for (int i = 0; i < gfs->n_evol_vars; i++)
    {
#define NAME_LENGTH 100
        char name[NAME_LENGTH];
        snprintf(name, NAME_LENGTH, "var_%d", i);
        create_string_attribute(group_id, name, gfs->vars[i]->vname);
    }

    for (int i = 0; i < gfs->n_aux_vars; i++)
    {
#define NAME_LENGTH 100
        char name[NAME_LENGTH];
        snprintf(name, NAME_LENGTH, "aux_%d", i);
        create_string_attribute(group_id, name, gfs->auxvars[i]->vname);
    }

    HDF5_ERROR(H5Gclose(group_id));
}

/******************************************************************************
 * BinaryWriteArray:
 * purpose: Write N-dimensional data to an hdf5 file
 *
 * filename: (char *, input) pointer to string contanining the hdf5 filename.
 * groupname: (char *, input) pointer to string contanining the groupname
 *                            where the dataset will be stored
 * datasetname: (char *, input) pointer to string with the dataset name
 * ndim: (int, input) number of dimensions in the dataset
 * local_dim: (size_t array, input) size of the dataset in each
 * dimension. Note the ordering local_dim = {...,ny, nx}, NOT {nx, ny, ...}
 *
 * data (pointer to double, input): location of first element of the dataset
 * gfs : (point to ngfs structure, input) Used to write the correct metadata
 *                                        for the HDF5 file
 *
 * returns (int): 0 on success, non-zero on failure
 *
 *****************************************************************************/

int BinaryWriteArray(const char *filename, const char *groupname,
                     const char *datasetname, const int ndim,
                     const size_t local_dim[], const double *data,
                     const struct ngfs *gfs)
{
    hid_t file_id;
    herr_t status;
    int file_exists = 0;
    int error_count = 0;

    H5E_BEGIN_TRY { file_exists = H5Fis_hdf5(filename) > 0; }
    H5E_END_TRY;

    if (file_exists)
    {
        file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
        if (file_id < 0)
        {
            fprintf(stderr, "Failed to open hdf5 file");
            return -1;
        }
    }
    else
    {
        file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0)
        {
            fprintf(stderr, "Failed to create hdf5 file");
            return -1;
        }
        write_metadata(file_id, gfs);
    }
#define BUFFSIZE 512
    char fullname[BUFFSIZE];

    snprintf(fullname, BUFFSIZE, "/%s", groupname);
    H5E_BEGIN_TRY
    {
        status = H5Lget_info(file_id, fullname, NULL, H5P_DEFAULT);
    }
    H5E_END_TRY;
    if (status < 0)
    {
        hid_t group_id;

        group_id =
            H5Gcreate(file_id, fullname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (group_id < 0)
        {
            fprintf(stderr, "Failed to create group");
            return -1;
        }
        H5Gclose(group_id);
    }

    assert(ndim < 4);

    hsize_t dim[4];
    for (int i = 0; i < ndim; i++)
    {
        dim[i] = local_dim[i];
    }

    snprintf(fullname, BUFFSIZE, "/%s/%s", groupname, datasetname);
    H5E_BEGIN_TRY { status = H5Ldelete(file_id, fullname, H5P_DEFAULT); }
    H5E_END_TRY;

    HDF5_ERROR(H5LTmake_dataset_double(file_id, fullname, ndim, dim, data));

    HDF5_ERROR(H5Fclose(file_id));

    if (error_count > 0)
    {
        fprintf(stderr, "Failed to write data to hdf5 file");
        return -1;
    }

    return 0;
}
