#include "HDF5BinaryWrite.h"

#include <hdf5.h>
#include <hdf5_hl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "domain.h"
#include "gf.h"

/* Lightweight error bookkeeping: every HDF5 call is wrapped so a
 * negative return increments a per-function error counter; the
 * top-level BinaryWriteArray returns -1 if any wrapped call failed.
 * This mirrors the pattern used in Maxwell_Penalty/src/HDF5BinaryWrite.c. */
#define HDF5_ERROR(fn_call)                                                    \
    do                                                                         \
    {                                                                          \
        int _ec = (int)(fn_call);                                              \
        if (_ec < 0)                                                           \
        {                                                                      \
            fprintf(stderr, "HDF5 call '%s' returned error code %d\n",         \
                    #fn_call, _ec);                                            \
            error_count++;                                                     \
        }                                                                      \
    } while (0)

/* ---- Static attribute writers (shared by 2D and 3D metadata) ----------- */

static void create_int_attribute(hid_t parent, const char *name, int val)
{
    int error_count = 0;
    hsize_t oD1 = 1;
    hid_t sp, attr;
    HDF5_ERROR(sp = H5Screate_simple(1, &oD1, NULL));
    HDF5_ERROR(attr = H5Acreate(parent, name, H5T_NATIVE_INT, sp,
                                H5P_DEFAULT, H5P_DEFAULT));
    HDF5_ERROR(H5Awrite(attr, H5T_NATIVE_INT, &val));
    HDF5_ERROR(H5Aclose(attr));
    HDF5_ERROR(H5Sclose(sp));
}

static void create_int64_attribute(hid_t parent, const char *name, int64_t val)
{
    int error_count = 0;
    hsize_t oD1 = 1;
    hid_t sp, attr;
    HDF5_ERROR(sp = H5Screate_simple(1, &oD1, NULL));
    HDF5_ERROR(attr = H5Acreate(parent, name, H5T_NATIVE_INT64, sp,
                                H5P_DEFAULT, H5P_DEFAULT));
    HDF5_ERROR(H5Awrite(attr, H5T_NATIVE_INT64, &val));
    HDF5_ERROR(H5Aclose(attr));
    HDF5_ERROR(H5Sclose(sp));
}

static void create_double_attribute(hid_t parent, const char *name, double val)
{
    int error_count = 0;
    hsize_t oD1 = 1;
    hid_t sp, attr;
    HDF5_ERROR(sp = H5Screate_simple(1, &oD1, NULL));
    HDF5_ERROR(attr = H5Acreate(parent, name, H5T_NATIVE_DOUBLE, sp,
                                H5P_DEFAULT, H5P_DEFAULT));
    HDF5_ERROR(H5Awrite(attr, H5T_NATIVE_DOUBLE, &val));
    HDF5_ERROR(H5Aclose(attr));
    HDF5_ERROR(H5Sclose(sp));
}

/* Boolean flag stored as an int (0/1) so h5py/h5dump display it without
 * boolean-type acrobatics. */
static void create_bool_attribute(hid_t parent, const char *name, int val)
{
    create_int_attribute(parent, name, val ? 1 : 0);
}

/* ---- 3D metadata ------------------------------------------------------- */

static void write_metadata_3d(hid_t file_id, const struct ngfs_3d *gfs)
{
    int error_count = 0;
    hid_t meta;
    HDF5_ERROR(meta = H5Gcreate(file_id, "/metadata", H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT));

    create_int_attribute(meta, "gs",       gfs->gs);
    create_int_attribute(meta, "rank",     gfs->domain.rank);
    create_int_attribute(meta, "mpi_size", gfs->domain.mpi_size);

    /* Local dims (full array, including ghost slots). */
    create_int64_attribute(meta, "nx", (int64_t)gfs->nx);
    create_int64_attribute(meta, "ny", (int64_t)gfs->ny);
    create_int64_attribute(meta, "nz", (int64_t)gfs->nz);

    /* This rank's offset in the global index space. */
    create_int64_attribute(meta, "local_i0", (int64_t)gfs->domain.local_i0);
    create_int64_attribute(meta, "local_j0", (int64_t)gfs->domain.local_j0);
    create_int64_attribute(meta, "local_k0", (int64_t)gfs->domain.local_k0);

    /* Global cell counts (used by verifiers to size the assembled array). */
    create_int64_attribute(meta, "global_cells_x",
                           (int64_t)gfs->domain.global_nx_cells);
    create_int64_attribute(meta, "global_cells_y",
                           (int64_t)gfs->domain.global_ny_cells);
    create_int64_attribute(meta, "global_cells_z",
                           (int64_t)gfs->domain.global_nz_cells);

    create_double_attribute(meta, "dx", gfs->dx);
    create_double_attribute(meta, "dy", gfs->dy);
    create_double_attribute(meta, "dz", gfs->dz);

    /* Local and global origins.  These are the *internally-shifted*
     * values that x = x0 + i*dx already produces in the C code -- the
     * verify scripts rely on that convention.  make_xdmf.py un-shifts
     * by +h/2 for any axis whose neumann_lower_* or neumann_upper_*
     * flag is set when computing user-frame coordinates. */
    create_double_attribute(meta, "x0", gfs->x0);
    create_double_attribute(meta, "y0", gfs->y0);
    create_double_attribute(meta, "z0", gfs->z0);
    create_double_attribute(meta, "global_x0", gfs->domain.global_x0);
    create_double_attribute(meta, "global_y0", gfs->domain.global_y0);
    create_double_attribute(meta, "global_z0", gfs->domain.global_z0);

    /* "_ghost" flags retained for backward parity with the previous JSON
     * layout: true means this face is an MPI ghost (neighbour rank
     * exists), false means it's a physical boundary owned by this rank. */
    create_bool_attribute(meta, "lower_x_ghost",
                          gfs->domain.lower_x_rank != INVALID_RANK);
    create_bool_attribute(meta, "upper_x_ghost",
                          gfs->domain.upper_x_rank != INVALID_RANK);
    create_bool_attribute(meta, "lower_y_ghost",
                          gfs->domain.lower_y_rank != INVALID_RANK);
    create_bool_attribute(meta, "upper_y_ghost",
                          gfs->domain.upper_y_rank != INVALID_RANK);
    create_bool_attribute(meta, "lower_z_ghost",
                          gfs->domain.lower_z_rank != INVALID_RANK);
    create_bool_attribute(meta, "upper_z_ghost",
                          gfs->domain.upper_z_rank != INVALID_RANK);

    /* Per-face boundary kind.  make_xdmf.py uses these to decide which
     * endpoint slots are physically meaningful (D vertex on hybrid axis
     * → include; pure-N ghost → exclude). */
    create_bool_attribute(meta, "neumann_lower_x",
                          gfs->domain.neumann_lower_x);
    create_bool_attribute(meta, "neumann_upper_x",
                          gfs->domain.neumann_upper_x);
    create_bool_attribute(meta, "neumann_lower_y",
                          gfs->domain.neumann_lower_y);
    create_bool_attribute(meta, "neumann_upper_y",
                          gfs->domain.neumann_upper_y);
    create_bool_attribute(meta, "neumann_lower_z",
                          gfs->domain.neumann_lower_z);
    create_bool_attribute(meta, "neumann_upper_z",
                          gfs->domain.neumann_upper_z);

    HDF5_ERROR(H5Gclose(meta));
    (void)error_count;
}

/* ---- 2D metadata (subset of the 3D fields) ----------------------------- */

static void write_metadata_2d(hid_t file_id, const struct ngfs_2d *gfs)
{
    int error_count = 0;
    hid_t meta;
    HDF5_ERROR(meta = H5Gcreate(file_id, "/metadata", H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT));

    create_int_attribute(meta, "gs",       gfs->gs);
    create_int_attribute(meta, "rank",     gfs->domain.rank);
    create_int_attribute(meta, "mpi_size", gfs->domain.mpi_size);

    create_int64_attribute(meta, "nx", (int64_t)gfs->nx);
    create_int64_attribute(meta, "ny", (int64_t)gfs->ny);

    create_int64_attribute(meta, "local_i0", (int64_t)gfs->domain.local_i0);
    create_int64_attribute(meta, "local_j0", (int64_t)gfs->domain.local_j0);

    create_int64_attribute(meta, "global_cells_x",
                           (int64_t)gfs->domain.global_nx_cells);
    create_int64_attribute(meta, "global_cells_y",
                           (int64_t)gfs->domain.global_ny_cells);

    create_double_attribute(meta, "dx", gfs->dx);
    create_double_attribute(meta, "dy", gfs->dy);
    create_double_attribute(meta, "x0", gfs->x0);
    create_double_attribute(meta, "y0", gfs->y0);
    create_double_attribute(meta, "global_x0", gfs->domain.global_x0);
    create_double_attribute(meta, "global_y0", gfs->domain.global_y0);

    create_bool_attribute(meta, "lower_x_ghost",
                          gfs->domain.lower_x_rank != INVALID_RANK);
    create_bool_attribute(meta, "upper_x_ghost",
                          gfs->domain.upper_x_rank != INVALID_RANK);
    create_bool_attribute(meta, "lower_y_ghost",
                          gfs->domain.lower_y_rank != INVALID_RANK);
    create_bool_attribute(meta, "upper_y_ghost",
                          gfs->domain.upper_y_rank != INVALID_RANK);

    /* 2D domain has no Neumann-flag fields (CellCentred_plan applies to
     * 3D only); 2D output is used solely by operator tests that don't
     * need the per-face boundary kind. */

    HDF5_ERROR(H5Gclose(meta));
    (void)error_count;
}

/* ---- Public writer entry points ---------------------------------------- */

static int write_dataset(hid_t file_id, const char *datasetname,
                         int ndim, const size_t local_dim[],
                         const double *data)
{
    int error_count = 0;

    char fullname[512];
    snprintf(fullname, sizeof(fullname), "/%s", datasetname);

    /* Overwrite any pre-existing dataset with the same name -- mirrors
     * the historical JSON behaviour where each driver invocation wrote
     * a fresh file. */
    H5E_BEGIN_TRY { H5Ldelete(file_id, fullname, H5P_DEFAULT); } H5E_END_TRY;

    hsize_t dim[4];
    for (int i = 0; i < ndim; i++) dim[i] = (hsize_t)local_dim[i];

    HDF5_ERROR(H5LTmake_dataset_double(file_id, fullname, ndim, dim, data));
    return error_count;
}

int BinaryWriteArray_3d(const char *filename, const char *datasetname,
                        const size_t local_dim[3], const double *data,
                        const struct ngfs_3d *gfs)
{
    int error_count = 0;
    int file_exists = 0;
    hid_t file_id;

    H5E_BEGIN_TRY { file_exists = H5Fis_hdf5(filename) > 0; } H5E_END_TRY;

    if (file_exists)
    {
        HDF5_ERROR(file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT));
    }
    else
    {
        HDF5_ERROR(file_id = H5Fcreate(filename, H5F_ACC_TRUNC,
                                       H5P_DEFAULT, H5P_DEFAULT));
        if (file_id >= 0)
            write_metadata_3d(file_id, gfs);
    }

    if (file_id < 0)
    {
        fprintf(stderr, "rank %d: cannot open HDF5 file '%s'\n",
                gfs->domain.rank, filename);
        return -1;
    }

    error_count += write_dataset(file_id, datasetname, 3, local_dim, data);
    HDF5_ERROR(H5Fclose(file_id));

    return (error_count > 0) ? -1 : 0;
}

int BinaryWriteArray_2d(const char *filename, const char *datasetname,
                        const size_t local_dim[2], const double *data,
                        const struct ngfs_2d *gfs)
{
    int error_count = 0;
    int file_exists = 0;
    hid_t file_id;

    H5E_BEGIN_TRY { file_exists = H5Fis_hdf5(filename) > 0; } H5E_END_TRY;

    if (file_exists)
    {
        HDF5_ERROR(file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT));
    }
    else
    {
        HDF5_ERROR(file_id = H5Fcreate(filename, H5F_ACC_TRUNC,
                                       H5P_DEFAULT, H5P_DEFAULT));
        if (file_id >= 0)
            write_metadata_2d(file_id, gfs);
    }

    if (file_id < 0)
    {
        fprintf(stderr, "rank %d: cannot open HDF5 file '%s'\n",
                gfs->domain.rank, filename);
        return -1;
    }

    error_count += write_dataset(file_id, datasetname, 2, local_dim, data);
    HDF5_ERROR(H5Fclose(file_id));

    return (error_count > 0) ? -1 : 0;
}
