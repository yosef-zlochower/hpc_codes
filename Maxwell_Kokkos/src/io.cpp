#include "io.hpp"
#include "HDF5BinaryWrite.hpp"
#include <Kokkos_Core.hpp>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <mpi.h>
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <vector>

#define BUFF_LEN 512

/* File-scope override for the output_gfs_3D_h5 static counter (used to
 * resume HDF5 group numbering after a checkpoint restart). */
static int output_3d_counter_override = -1;

/* Pull a 3D View into a freshly allocated host vector laid out in the
 * same i + j*nx + k*nx*ny offset scheme as the View's LayoutLeft. */
static std::vector<double> view_to_host_buffer(const Field3D &v)
{
    auto h = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h, v);
    const int64_t nx = v.extent(0);
    const int64_t ny = v.extent(1);
    const int64_t nz = v.extent(2);
    std::vector<double> out((size_t)nx * (size_t)ny * (size_t)nz);
    for (int64_t k = 0; k < nz; k++)
        for (int64_t j = 0; j < ny; j++)
            for (int64_t i = 0; i < nx; i++)
                out[(size_t)(i + j * nx + k * nx * ny)] = h(i, j, k);
    return out;
}

void output_gfs_metadata(NGFS *gfs)
{
    char name_buff[BUFF_LEN];
    std::snprintf(name_buff, BUFF_LEN, "plot_metadata.py");
    FILE *ofile = std::fopen(name_buff, "w");
    std::fprintf(ofile, "gf_names = [ ");
    for (int v = 0; v < gfs->n_evol_vars; v++)
    {
        std::fprintf(ofile, "\"%s\", ", gfs->evol[v].vname.c_str());
    }
    std::fprintf(ofile, "]\n");
    std::fprintf(ofile,
            "mpi_size = %d\n"
            "global_ni = %" PRId64 "\n"
            "global_nj = %" PRId64 "\n"
            "global_nk = %" PRId64 "\n"
            "dx = %20.16e\n"
            "dy = %20.16e\n"
            "dz = %20.16e\n"
            "global_x0 = %20.16e\n"
            "global_y0 = %20.16e\n"
            "global_z0 = %20.16e\n",
            gfs->domain.mpi_size, gfs->domain.global_ni, gfs->domain.global_nj,
            gfs->domain.global_nk, gfs->dx, gfs->dy, gfs->dz,
            gfs->domain.global_x0, gfs->domain.global_y0,
            gfs->domain.global_z0);
    std::fclose(ofile);
}

void output_gfs_3D_h5(NGFS *gfs)
{
    static int counter = 0;
    if (output_3d_counter_override >= 0)
    {
        counter = output_3d_counter_override;
        output_3d_counter_override = -1;
    }
    char filename[BUFF_LEN];
    const int64_t ni = gfs->nx;
    const int64_t nj = gfs->ny;
    const int64_t nk = gfs->nz;

    if (gfs->domain.rank == 0)
        std::fprintf(stderr, "Not safe to quit!\n");
    MPI_Barrier(MPI_COMM_WORLD);
    std::snprintf(filename, BUFF_LEN, "3D_rank_%d.h5", gfs->domain.rank);

    char groupname[BUFF_LEN];
    std::snprintf(groupname, BUFF_LEN, "%d", counter);

    size_t local_dim[3] = { (size_t)nk, (size_t)nj, (size_t)ni };

    for (int v = 0; v < gfs->n_evol_vars; v++)
    {
        auto buf = view_to_host_buffer(gfs->evol[v].state);
        BinaryWriteArray(filename, groupname,
                         gfs->evol[v].vname.c_str(),
                         3, local_dim, buf.data(), gfs);
    }
    for (int v = 0; v < gfs->n_aux_vars; v++)
    {
        auto buf = view_to_host_buffer(gfs->aux[v].state);
        BinaryWriteArray(filename, groupname,
                         gfs->aux[v].vname.c_str(),
                         3, local_dim, buf.data(), gfs);
    }
    ++counter;
    MPI_Barrier(MPI_COMM_WORLD);
    if (gfs->domain.rank == 0)
        std::fprintf(stderr, "Safe to quit\n");
}

void set_output_counter_3D(int value)
{
    output_3d_counter_override = value;
}

/* ── Checkpoint / Recovery ─────────────────────────────────────────────── */

#define HDF5_CHK(fn_call)                                                      \
    do {                                                                       \
        int _ec = fn_call;                                                     \
        if (_ec < 0) {                                                         \
            std::fprintf(stderr, "HDF5 error in '%s' (line %d)\n", #fn_call,   \
                         __LINE__);                                            \
            chk_errors++;                                                      \
        }                                                                      \
    } while (0)

#define MAX_TRACKED_CHECKPOINTS 4096
static int  tracked_iterations[MAX_TRACKED_CHECKPOINTS];
static int  n_tracked = 0;

static void delete_checkpoint_files(int old_it, int my_rank)
{
    char path[BUFF_LEN];
    std::snprintf(path, BUFF_LEN, "checkpoint_it_%d_rank_%d.h5",
                  old_it, my_rank);
    std::remove(path);
}

void write_checkpoint(NGFS *gfs, double t, int it, int max_checkpoints)
{
    int chk_errors = 0;
    char tmpname[BUFF_LEN], filename[BUFF_LEN];
    std::snprintf(tmpname, BUFF_LEN, "checkpoint_it_%d_rank_%d.h5.tmp",
                  it, gfs->domain.rank);
    std::snprintf(filename, BUFF_LEN, "checkpoint_it_%d_rank_%d.h5",
                  it, gfs->domain.rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (gfs->domain.rank == 0)
        std::fprintf(stderr, "Writing checkpoint at t = %g, iteration %d ...\n",
                     t, it);

    hid_t file_id;
    HDF5_CHK(file_id = H5Fcreate(tmpname, H5F_ACC_TRUNC, H5P_DEFAULT,
                                  H5P_DEFAULT));

    hid_t meta_id;
    HDF5_CHK(meta_id = H5Gcreate(file_id, "/metadata", H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT));
    {
        hsize_t one = 1;
        hid_t sp;
        HDF5_CHK(sp = H5Screate_simple(1, &one, NULL));

#define WRITE_INT_ATTR(name, val)                                              \
    do {                                                                       \
        int _v = (val);                                                        \
        hid_t a;                                                               \
        HDF5_CHK(a = H5Acreate(meta_id, name, H5T_NATIVE_INT, sp,             \
                                H5P_DEFAULT, H5P_DEFAULT));                    \
        HDF5_CHK(H5Awrite(a, H5T_NATIVE_INT, &_v));                           \
        HDF5_CHK(H5Aclose(a));                                                \
    } while (0)

#define WRITE_DBL_ATTR(name, val)                                              \
    do {                                                                       \
        double _v = (val);                                                     \
        hid_t a;                                                               \
        HDF5_CHK(a = H5Acreate(meta_id, name, H5T_NATIVE_DOUBLE, sp,          \
                                H5P_DEFAULT, H5P_DEFAULT));                    \
        HDF5_CHK(H5Awrite(a, H5T_NATIVE_DOUBLE, &_v));                        \
        HDF5_CHK(H5Aclose(a));                                                \
    } while (0)

        WRITE_INT_ATTR("ghost_zones", gfs->gs);
        WRITE_INT_ATTR("mpi_size",    gfs->domain.mpi_size);
        WRITE_INT_ATTR("global_ni",   (int)gfs->domain.global_ni);
        WRITE_INT_ATTR("global_nj",   (int)gfs->domain.global_nj);
        WRITE_INT_ATTR("global_nk",   (int)gfs->domain.global_nk);
        WRITE_INT_ATTR("n_evol_vars", gfs->n_evol_vars);
        WRITE_INT_ATTR("iteration",   it);
        WRITE_DBL_ATTR("time",        t);

#undef WRITE_INT_ATTR
#undef WRITE_DBL_ATTR

        HDF5_CHK(H5Sclose(sp));
    }
    HDF5_CHK(H5Gclose(meta_id));

    hid_t evol_id;
    HDF5_CHK(evol_id = H5Gcreate(file_id, "/evolved", H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT));
    const int64_t ni = gfs->nx;
    const int64_t nj = gfs->ny;
    const int64_t nk = gfs->nz;
    hsize_t dims[3] = { (hsize_t)nk, (hsize_t)nj, (hsize_t)ni };
    for (int v = 0; v < gfs->n_evol_vars; v++)
    {
        auto buf = view_to_host_buffer(gfs->evol[v].state);
        HDF5_CHK(H5LTmake_dataset_double(evol_id,
                                          gfs->evol[v].vname.c_str(),
                                          3, dims, buf.data()));
    }
    HDF5_CHK(H5Gclose(evol_id));

    hid_t mat_id;
    HDF5_CHK(mat_id = H5Gcreate(file_id, "/material", H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT));
    /* aux[0..2] are ieps, imu, sigma (the always-present material props). */
    for (int v = 0; v < 3; v++)
    {
        auto buf = view_to_host_buffer(gfs->aux[v].state);
        HDF5_CHK(H5LTmake_dataset_double(mat_id,
                                          gfs->aux[v].vname.c_str(),
                                          3, dims, buf.data()));
    }
    HDF5_CHK(H5Gclose(mat_id));
    HDF5_CHK(H5Fclose(file_id));

    MPI_Barrier(MPI_COMM_WORLD);
    std::rename(tmpname, filename);
    MPI_Barrier(MPI_COMM_WORLD);

    if (n_tracked < MAX_TRACKED_CHECKPOINTS)
        tracked_iterations[n_tracked++] = it;

    while (n_tracked > max_checkpoints)
    {
        int old_it = tracked_iterations[0];
        delete_checkpoint_files(old_it, gfs->domain.rank);
        for (int i = 1; i < n_tracked; i++)
            tracked_iterations[i - 1] = tracked_iterations[i];
        n_tracked--;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (gfs->domain.rank == 0)
        std::fprintf(stderr, "Checkpoint complete.\n");
}

static int find_latest_checkpoint_iteration(int my_rank)
{
    DIR *dir = opendir(".");
    if (!dir) return -1;
    struct dirent *entry;
    int latest_it = -1;
    while ((entry = readdir(dir)) != NULL)
    {
        int found_it, found_rank;
        if (std::sscanf(entry->d_name, "checkpoint_it_%d_rank_%d.h5",
                        &found_it, &found_rank) == 2 &&
            found_rank == my_rank && found_it > latest_it)
        {
            latest_it = found_it;
        }
    }
    closedir(dir);
    return latest_it;
}

/* Read a 3D dataset of dims (nz, ny, nx) into the given Field3D's host
 * mirror, reshape into LayoutLeft offsets, deep_copy back to the device. */
static int read_dataset_into(hid_t group_id, const char *dsname, Field3D &v)
{
    int chk_errors = 0;
    const int64_t nx = v.extent(0);
    const int64_t ny = v.extent(1);
    const int64_t nz = v.extent(2);
    std::vector<double> buf((size_t)nx * (size_t)ny * (size_t)nz);
    HDF5_CHK(H5LTread_dataset_double(group_id, dsname, buf.data()));
    auto h = Kokkos::create_mirror_view(v);
    for (int64_t k = 0; k < nz; k++)
        for (int64_t j = 0; j < ny; j++)
            for (int64_t i = 0; i < nx; i++)
                h(i, j, k) = buf[(size_t)(i + j * nx + k * nx * ny)];
    Kokkos::deep_copy(v, h);
    return chk_errors;
}

int read_checkpoint(NGFS *gfs, double *t, int *it)
{
    int chk_errors = 0;
    char filename[BUFF_LEN];
    int latest_it = find_latest_checkpoint_iteration(gfs->domain.rank);
    if (latest_it < 0)
    {
        std::fprintf(stderr, "rank %d: no checkpoint files found\n",
                     gfs->domain.rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    std::snprintf(filename, BUFF_LEN, "checkpoint_it_%d_rank_%d.h5",
                  latest_it, gfs->domain.rank);

    hid_t file_id;
    HDF5_CHK(file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT));
    if (file_id < 0)
    {
        std::fprintf(stderr, "rank %d: cannot open '%s'\n",
                     gfs->domain.rank, filename);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    hid_t meta_id;
    HDF5_CHK(meta_id = H5Gopen(file_id, "/metadata", H5P_DEFAULT));
#define READ_INT_ATTR(name, dst)                                               \
    do {                                                                       \
        hid_t a;                                                               \
        HDF5_CHK(a = H5Aopen(meta_id, name, H5P_DEFAULT));                    \
        HDF5_CHK(H5Aread(a, H5T_NATIVE_INT, dst));                            \
        HDF5_CHK(H5Aclose(a));                                                \
    } while (0)
#define READ_DBL_ATTR(name, dst)                                               \
    do {                                                                       \
        hid_t a;                                                               \
        HDF5_CHK(a = H5Aopen(meta_id, name, H5P_DEFAULT));                    \
        HDF5_CHK(H5Aread(a, H5T_NATIVE_DOUBLE, dst));                         \
        HDF5_CHK(H5Aclose(a));                                                \
    } while (0)

    int chk_gs, chk_mpi_size, chk_ni, chk_nj, chk_nk, chk_nvars, chk_it;
    double chk_t;
    READ_INT_ATTR("ghost_zones", &chk_gs);
    READ_INT_ATTR("mpi_size",    &chk_mpi_size);
    READ_INT_ATTR("global_ni",   &chk_ni);
    READ_INT_ATTR("global_nj",   &chk_nj);
    READ_INT_ATTR("global_nk",   &chk_nk);
    READ_INT_ATTR("n_evol_vars", &chk_nvars);
    READ_INT_ATTR("iteration",   &chk_it);
    READ_DBL_ATTR("time",        &chk_t);
#undef READ_INT_ATTR
#undef READ_DBL_ATTR
    HDF5_CHK(H5Gclose(meta_id));

    if (chk_gs != gfs->gs ||
        chk_mpi_size != gfs->domain.mpi_size ||
        chk_ni != (int)gfs->domain.global_ni ||
        chk_nj != (int)gfs->domain.global_nj ||
        chk_nk != (int)gfs->domain.global_nk ||
        chk_nvars != gfs->n_evol_vars)
    {
        std::fprintf(stderr,
                     "rank %d: checkpoint metadata mismatch\n",
                     gfs->domain.rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    *t  = chk_t;
    *it = chk_it;

    hid_t evol_id;
    HDF5_CHK(evol_id = H5Gopen(file_id, "/evolved", H5P_DEFAULT));
    for (int v = 0; v < gfs->n_evol_vars; v++)
        chk_errors += read_dataset_into(evol_id,
                                         gfs->evol[v].vname.c_str(),
                                         gfs->evol[v].state);
    HDF5_CHK(H5Gclose(evol_id));

    hid_t mat_id;
    HDF5_CHK(mat_id = H5Gopen(file_id, "/material", H5P_DEFAULT));
    for (int v = 0; v < 3; v++)
        chk_errors += read_dataset_into(mat_id,
                                         gfs->aux[v].vname.c_str(),
                                         gfs->aux[v].state);
    HDF5_CHK(H5Gclose(mat_id));
    HDF5_CHK(H5Fclose(file_id));

    if (gfs->domain.rank == 0)
        std::fprintf(stderr,
                     "Recovered from checkpoint: t = %g, iteration %d\n",
                     *t, *it);
    return chk_errors;
}
