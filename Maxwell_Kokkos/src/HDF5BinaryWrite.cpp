#include "HDF5BinaryWrite.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>
#include <cassert>
#include <cstdio>
#include <cstring>

#include "hdf5_check.h"

static void create_string_attribute(hid_t id, const char *name, const char *val)
{
    hid_t dataspace_id, attribute_id, datatype;
    hsize_t oD1 = 1;

    HDF5_CHK(datatype = H5Tcopy(H5T_C_S1));
    HDF5_CHK(H5Tset_size(datatype, std::strlen(val) + 1));
    HDF5_CHK(dataspace_id = H5Screate_simple(1, &oD1, NULL));
    HDF5_CHK(attribute_id = H5Acreate(id, name, datatype, dataspace_id,
                                         H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CHK(H5Awrite(attribute_id, datatype, val));
    HDF5_CHK(H5Aclose(attribute_id));
    HDF5_CHK(H5Sclose(dataspace_id));
    HDF5_CHK(H5Tclose(datatype));
}

static void create_int_attribute(hid_t id, const char *name, int val)
{
    hid_t dataspace_id, attribute_id;
    hsize_t oD1 = 1;
    HDF5_CHK(dataspace_id = H5Screate_simple(1, &oD1, NULL));
    HDF5_CHK(attribute_id = H5Acreate(id, name, H5T_NATIVE_INT, dataspace_id,
                                         H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CHK(H5Awrite(attribute_id, H5T_NATIVE_INT, &val));
    HDF5_CHK(H5Aclose(attribute_id));
    HDF5_CHK(H5Sclose(dataspace_id));
}

static void create_double_attribute(hid_t id, const char *name, double val)
{
    hid_t dataspace_id, attribute_id;
    hsize_t oD1 = 1;
    HDF5_CHK(dataspace_id = H5Screate_simple(1, &oD1, NULL));
    HDF5_CHK(attribute_id = H5Acreate(id, name, H5T_NATIVE_DOUBLE,
                                         dataspace_id, H5P_DEFAULT,
                                         H5P_DEFAULT));
    HDF5_CHK(H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &val));
    HDF5_CHK(H5Aclose(attribute_id));
    HDF5_CHK(H5Sclose(dataspace_id));
}

static void write_metadata(hid_t file_id, const NGFS *gfs)
{
    hid_t group_id;
    HDF5_CHK(group_id = H5Gcreate(file_id, "/metadata", H5P_DEFAULT,
                                     H5P_DEFAULT, H5P_DEFAULT));

    create_int_attribute(group_id, "ghost_zones", gfs->gs);
    create_int_attribute(group_id, "mpi_size",    gfs->domain.mpi_size);
    create_int_attribute(group_id, "n_evol_vars", gfs->n_evol_vars);
    create_int_attribute(group_id, "n_aux_vars",  gfs->n_aux_vars);
    create_int_attribute(group_id, "local_ni",    (int)gfs->nx);
    create_int_attribute(group_id, "local_nj",    (int)gfs->ny);
    create_int_attribute(group_id, "local_nk",    (int)gfs->nz);
    create_int_attribute(group_id, "global_ni",   (int)gfs->domain.global_ni);
    create_int_attribute(group_id, "global_nj",   (int)gfs->domain.global_nj);
    create_int_attribute(group_id, "global_nk",   (int)gfs->domain.global_nk);
    create_int_attribute(group_id, "local_i0",    (int)gfs->domain.local_i0);
    create_int_attribute(group_id, "local_j0",    (int)gfs->domain.local_j0);
    create_int_attribute(group_id, "local_k0",    (int)gfs->domain.local_k0);
    create_double_attribute(group_id, "local_x0",  gfs->x0);
    create_double_attribute(group_id, "local_y0",  gfs->y0);
    create_double_attribute(group_id, "local_z0",  gfs->z0);
    create_double_attribute(group_id, "global_x0", gfs->domain.global_x0);
    create_double_attribute(group_id, "global_y0", gfs->domain.global_y0);
    create_double_attribute(group_id, "global_z0", gfs->domain.global_z0);
    create_double_attribute(group_id, "dx",        gfs->dx);
    create_double_attribute(group_id, "dy",        gfs->dy);
    create_double_attribute(group_id, "dz",        gfs->dz);

    char name[100];
    for (int i = 0; i < gfs->n_evol_vars; i++)
    {
        std::snprintf(name, sizeof name, "var_%d", i);
        create_string_attribute(group_id, name,
                                gfs->evol[i].vname.c_str());
    }
    for (int i = 0; i < gfs->n_aux_vars; i++)
    {
        std::snprintf(name, sizeof name, "aux_%d", i);
        create_string_attribute(group_id, name,
                                gfs->aux[i].vname.c_str());
    }
    HDF5_CHK(H5Gclose(group_id));
}

int BinaryWriteArray(const char *filename, const char *groupname,
                     const char *datasetname, int ndim,
                     const size_t local_dim[], const double *data,
                     const NGFS *gfs)
{
    hid_t file_id;
    herr_t status;
    int file_exists = 0;

    H5E_BEGIN_TRY { file_exists = H5Fis_hdf5(filename) > 0; }
    H5E_END_TRY;

    if (file_exists)
    {
        file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
        if (file_id < 0)
        {
            std::fprintf(stderr, "Failed to open hdf5 file");
            return -1;
        }
    }
    else
    {
        file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0)
        {
            std::fprintf(stderr, "Failed to create hdf5 file");
            return -1;
        }
        write_metadata(file_id, gfs);
    }

    char fullname[512];
    std::snprintf(fullname, sizeof fullname, "/%s", groupname);
    H5E_BEGIN_TRY {
        status = H5Lget_info(file_id, fullname, NULL, H5P_DEFAULT);
    } H5E_END_TRY;
    if (status < 0)
    {
        hid_t group_id = H5Gcreate(file_id, fullname, H5P_DEFAULT,
                                    H5P_DEFAULT, H5P_DEFAULT);
        if (group_id < 0)
        {
            std::fprintf(stderr, "Failed to create group");
            return -1;
        }
        H5Gclose(group_id);
    }

    assert(ndim < 4);
    hsize_t dim[4];
    for (int i = 0; i < ndim; i++) dim[i] = local_dim[i];

    std::snprintf(fullname, sizeof fullname, "/%s/%s", groupname, datasetname);
    H5E_BEGIN_TRY { status = H5Ldelete(file_id, fullname, H5P_DEFAULT); }
    H5E_END_TRY;

    HDF5_CHK(H5LTmake_dataset_double(file_id, fullname, ndim, dim, data));
    HDF5_CHK(H5Fclose(file_id));

    return 0;
}
