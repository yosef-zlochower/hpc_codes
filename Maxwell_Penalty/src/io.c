#include <assert.h>
#include <dirent.h>
#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "gf.h"
#include "HDF5BinaryWrite.h"
#include "io.h"
#include "domain.h"
#include "parameter.h"

#define BUFF_LEN 512

/* File-scope override for the output_gfs_3D_h5 static counter.
 * A value of -1 means "no override pending". */
static int output_3d_counter_override = -1;

/* output_gf_metadata produces a metadata file. This function should
 * only be called once by a single rank.
 */
void output_gfs_metadata(struct ngfs *gfs)
{
    char name_buff[BUFF_LEN];

    snprintf(name_buff, BUFF_LEN, "plot_metadata.py");
    FILE *ofile = fopen(name_buff, "w");
    fprintf(ofile, "gf_names = [ ");
    for (int v = 0; v < gfs->n_evol_vars; v++)
    {
        char def_name[BUFF_LEN];
        snprintf(def_name, BUFF_LEN, "Var_%d", v);

        fprintf(ofile, "\"%s\", ",
                gfs->vars[v]->vname ? gfs->vars[v]->vname : def_name);
    }
    fprintf(ofile, "]\n");

    fprintf(ofile,
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
    fclose(ofile);
}

/* Outputs 2D arrays in gnuplot format. Should be called by all ranks*/
void output_gfs_2D_xy(struct ngfs *gfs, const int global_k_index)
{
    static int counter = 0;
    char name_buff[BUFF_LEN];
    const int gs = gfs->gs;
    const int64_t ni = gfs->nx;
    const int64_t nj = gfs->ny;
    const int64_t nk = gfs->nz;
    const int64_t local_k = global_k_index - gfs->domain.local_k0;

    if (local_k < gs || local_k >= nk - gs)
    {
        // This process does not own the plane
        return;
    }

    for (int v = 0; v < gfs->n_evol_vars; v++)
    {
        char def_name[BUFF_LEN];
        snprintf(def_name, BUFF_LEN, "Var_%d", v);
        snprintf(name_buff, BUFF_LEN, "%.450s_rank_%d_it_%07d.asc",
                 gfs->vars[v]->vname ? gfs->vars[v]->vname : def_name,
                 gfs->domain.rank, counter);
        FILE *ofile = fopen(name_buff, "w");

        for (int64_t j = gs; j < nj - gs; j++)
        {
            for (int64_t i = gs; i < ni - gs; i++)
            {
                const int64_t ijk = i + j * ni + local_k * ni * nj;
                const int64_t j_global = gfs->domain.local_j0 + j;
                const int64_t i_global = gfs->domain.local_i0 + i;
                fprintf(ofile, "%" PRId64 " %" PRId64 " %20.16e\n", i_global, j_global,
                        gfs->vars[v]->new[ijk]);
            }
        }

        fclose(ofile);
        ofile = NULL;
    }
    ++counter;
}

void output_gfs_2D_xy_h5(struct ngfs *gfs, const int global_k_index)
{
    static int counter = 0;
    char filename[BUFF_LEN];

    const int gs = gfs->gs;
    const int64_t ni = gfs->nx;
    const int64_t nj = gfs->ny;
    const int64_t nk = gfs->nz;
    const int64_t local_k = global_k_index - gfs->domain.local_k0;

    if (local_k < gs || local_k >= nk - gs)
    {
        // This process does not own the plane
        return;
    }

    snprintf(filename, BUFF_LEN, "rank_%d.h5", gfs->domain.rank);
    double *data_buffer = NULL;

    data_buffer = malloc(sizeof(double) * ni * nj);
    assert(data_buffer);

    for (int v = 0; v < gfs->n_evol_vars; v++)
    {
        char groupname[BUFF_LEN];
        snprintf(groupname, BUFF_LEN, "%d", counter);

        char datasetname[BUFF_LEN];
        snprintf(datasetname, BUFF_LEN, "%s", gfs->vars[v]->vname);

        for (int64_t j = 0; j < nj; j++)
        {
            for (int64_t i = 0; i < ni; i++)
            {
                const int64_t ijk = i + j * ni + local_k * ni * nj;
                const int64_t ijk_dst = i + j * ni;

                data_buffer[ijk_dst] = gfs->vars[v]->new[ijk];
            }
        }

        size_t local_dim[2] = { nj, ni };

        BinaryWriteArray(filename, groupname, datasetname, 2, local_dim,
                         data_buffer, gfs);
    }
    ++counter;
    free(data_buffer);
    data_buffer = NULL;
}

/* Output all evolved and auxiliary variables as 3D HDF5 datasets.
 * Each rank writes its own file (3D_rank_N.h5).  The "Not safe to quit"
 * / "Safe to quit" messages bracket the I/O phase so that an interactive
 * user can safely interrupt the run only between output calls. */
void output_gfs_3D_h5(struct ngfs *gfs)
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
        fprintf(stderr, "Not safe to quit!\n");
    MPI_Barrier(MPI_COMM_WORLD);
    snprintf(filename, BUFF_LEN, "3D_rank_%d.h5", gfs->domain.rank);

    for (int v = 0; v < gfs->n_evol_vars; v++)
    {
        char groupname[BUFF_LEN];
        snprintf(groupname, BUFF_LEN, "%d", counter);

        char datasetname[BUFF_LEN];
        snprintf(datasetname, BUFF_LEN, "%s", gfs->vars[v]->vname);

        size_t local_dim[3] = { nk, nj, ni };

        BinaryWriteArray(filename, groupname, datasetname, 3, local_dim,
                         gfs->vars[v]->new, gfs);
    }
    for (int v = 0; v < gfs->n_aux_vars; v++)
    {
        char groupname[BUFF_LEN];
        snprintf(groupname, BUFF_LEN, "%d", counter);

        char datasetname[BUFF_LEN];
        snprintf(datasetname, BUFF_LEN, "%s", gfs->auxvars[v]->vname);

        size_t local_dim[3] = { nk, nj, ni };

        BinaryWriteArray(filename, groupname, datasetname, 3, local_dim,
                         gfs->auxvars[v]->new, gfs);
    }
    ++counter;
    MPI_Barrier(MPI_COMM_WORLD);
    if (gfs->domain.rank == 0)
    {
        fprintf(stderr, "Safe to quit\n");
    }
}

void set_output_counter_3D(int value) {
    /* Force the next output_gfs_3D_h5 call to use this counter value.
     * We achieve this by calling output_gfs_3D_h5's static counter through
     * a file-scope variable that the function checks on entry. */
    output_3d_counter_override = value;
}

/* ── Checkpoint / Recovery ─────────────────────────────────────────────── */

#define HDF5_CHK(fn_call)                                                      \
    do                                                                         \
    {                                                                          \
        int _ec = fn_call;                                                     \
        if (_ec < 0)                                                           \
        {                                                                      \
            fprintf(stderr, "HDF5 error in '%s' (line %d)\n", #fn_call,        \
                    __LINE__);                                                 \
            chk_errors++;                                                      \
        }                                                                      \
    } while (0)

/* ── Per-execution checkpoint tracking ──────────────────────────────── */
/* We only ever delete checkpoints written during THIS execution,
 * so files from a previous run are never removed.                      */
#define MAX_TRACKED_CHECKPOINTS 4096
static int  tracked_iterations[MAX_TRACKED_CHECKPOINTS];
static int  n_tracked = 0;

static void delete_checkpoint_files(int old_it, int mpi_size, int my_rank)
{
    /* Each rank deletes its own file for the given iteration */
    (void)mpi_size;
    char path[BUFF_LEN];
    snprintf(path, BUFF_LEN, "checkpoint_it_%d_rank_%d.h5", old_it, my_rank);
    remove(path);
}

void write_checkpoint(struct ngfs *gfs, double t, int it, int max_checkpoints)
{
    int chk_errors = 0;
    char tmpname[BUFF_LEN], filename[BUFF_LEN];
    snprintf(tmpname,  BUFF_LEN, "checkpoint_it_%d_rank_%d.h5.tmp",
             it, gfs->domain.rank);
    snprintf(filename, BUFF_LEN, "checkpoint_it_%d_rank_%d.h5",
             it, gfs->domain.rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (gfs->domain.rank == 0)
        fprintf(stderr, "Writing checkpoint at t = %g, iteration %d ...\n",
                t, it);

    hid_t file_id;
    HDF5_CHK(file_id = H5Fcreate(tmpname, H5F_ACC_TRUNC,
                                  H5P_DEFAULT, H5P_DEFAULT));

    /* ── /metadata group ────────────────────────────────────────────── */
    hid_t meta_id;
    HDF5_CHK(meta_id = H5Gcreate(file_id, "/metadata", H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT));
    {
        hsize_t one = 1;
        hid_t sp;
        HDF5_CHK(sp = H5Screate_simple(1, &one, NULL));

#define WRITE_INT_ATTR(name, val)                                              \
    do                                                                         \
    {                                                                          \
        int _v = (val);                                                        \
        hid_t a;                                                               \
        HDF5_CHK(a = H5Acreate(meta_id, name, H5T_NATIVE_INT, sp,             \
                                H5P_DEFAULT, H5P_DEFAULT));                    \
        HDF5_CHK(H5Awrite(a, H5T_NATIVE_INT, &_v));                           \
        HDF5_CHK(H5Aclose(a));                                                \
    } while (0)

#define WRITE_DBL_ATTR(name, val)                                              \
    do                                                                         \
    {                                                                          \
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

    /* ── /evolved group with one dataset per variable ───────────────── */
    hid_t evol_id;
    HDF5_CHK(evol_id = H5Gcreate(file_id, "/evolved", H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT));

    const int64_t ni = gfs->nx;
    const int64_t nj = gfs->ny;
    const int64_t nk = gfs->nz;
    hsize_t dims[3] = { (hsize_t)nk, (hsize_t)nj, (hsize_t)ni };

    for (int v = 0; v < gfs->n_evol_vars; v++)
    {
        char dsname[BUFF_LEN];
        snprintf(dsname, BUFF_LEN, "%s", gfs->vars[v]->vname);
        HDF5_CHK(H5LTmake_dataset_double(evol_id, dsname, 3, dims,
                                          gfs->vars[v]->new));
    }

    HDF5_CHK(H5Gclose(evol_id));

    /* ── /material group: spatially varying material properties ─────── */
    hid_t mat_id;
    HDF5_CHK(mat_id = H5Gcreate(file_id, "/material", H5P_DEFAULT,
                                  H5P_DEFAULT, H5P_DEFAULT));

    /* auxvars[0] = ieps, auxvars[1] = imu, auxvars[2] = sigma */
    for (int v = 0; v < 3; v++)
    {
        char dsname[BUFF_LEN];
        snprintf(dsname, BUFF_LEN, "%s", gfs->auxvars[v]->vname);
        HDF5_CHK(H5LTmake_dataset_double(mat_id, dsname, 3, dims,
                                          gfs->auxvars[v]->new));
    }

    HDF5_CHK(H5Gclose(mat_id));
    HDF5_CHK(H5Fclose(file_id));

    /* Atomic rename: all ranks finish writing, then rename */
    MPI_Barrier(MPI_COMM_WORLD);
    rename(tmpname, filename);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Track this checkpoint and delete old ones from this execution */
    if (n_tracked < MAX_TRACKED_CHECKPOINTS)
        tracked_iterations[n_tracked++] = it;

    while (n_tracked > max_checkpoints)
    {
        int old_it = tracked_iterations[0];
        delete_checkpoint_files(old_it, gfs->domain.mpi_size,
                                gfs->domain.rank);
        /* Shift the tracking array */
        for (int i = 1; i < n_tracked; i++)
            tracked_iterations[i - 1] = tracked_iterations[i];
        n_tracked--;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (gfs->domain.rank == 0)
        fprintf(stderr, "Checkpoint complete.\n");
}

/* Find the highest iteration number among checkpoint files for this rank. */
static int find_latest_checkpoint_iteration(int my_rank)
{
    DIR *dir = opendir(".");
    if (!dir)
        return -1;

    struct dirent *entry;
    int latest_it = -1;

    while ((entry = readdir(dir)) != NULL)
    {
        int found_it, found_rank;
        if (sscanf(entry->d_name, "checkpoint_it_%d_rank_%d.h5",
                   &found_it, &found_rank) == 2 &&
            found_rank == my_rank &&
            found_it > latest_it)
        {
            latest_it = found_it;
        }
    }
    closedir(dir);
    return latest_it;
}

int read_checkpoint(struct ngfs *gfs, double *t, int *it)
{
    int chk_errors = 0;
    char filename[BUFF_LEN];

    int latest_it = find_latest_checkpoint_iteration(gfs->domain.rank);
    if (latest_it < 0)
    {
        fprintf(stderr, "rank %d: no checkpoint files found\n",
                gfs->domain.rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    snprintf(filename, BUFF_LEN, "checkpoint_it_%d_rank_%d.h5",
             latest_it, gfs->domain.rank);

    hid_t file_id;
    HDF5_CHK(file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT));
    if (file_id < 0)
    {
        fprintf(stderr, "rank %d: cannot open checkpoint file '%s'\n",
                gfs->domain.rank, filename);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* ── Read and validate metadata ─────────────────────────────────── */
    hid_t meta_id;
    HDF5_CHK(meta_id = H5Gopen(file_id, "/metadata", H5P_DEFAULT));

#define READ_INT_ATTR(name, dst)                                               \
    do                                                                         \
    {                                                                          \
        hid_t a;                                                               \
        HDF5_CHK(a = H5Aopen(meta_id, name, H5P_DEFAULT));                    \
        HDF5_CHK(H5Aread(a, H5T_NATIVE_INT, dst));                            \
        HDF5_CHK(H5Aclose(a));                                                \
    } while (0)

#define READ_DBL_ATTR(name, dst)                                               \
    do                                                                         \
    {                                                                          \
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

    /* Validate against current run */
    if (chk_gs != gfs->gs ||
        chk_mpi_size != gfs->domain.mpi_size ||
        chk_ni != (int)gfs->domain.global_ni ||
        chk_nj != (int)gfs->domain.global_nj ||
        chk_nk != (int)gfs->domain.global_nk ||
        chk_nvars != gfs->n_evol_vars)
    {
        fprintf(stderr,
                "rank %d: checkpoint metadata mismatch\n"
                "  checkpoint: gs=%d mpi=%d ni=%d nj=%d nk=%d nvars=%d\n"
                "  current:    gs=%d mpi=%d ni=%" PRId64 " nj=%" PRId64
                " nk=%" PRId64 " nvars=%d\n",
                gfs->domain.rank,
                chk_gs, chk_mpi_size, chk_ni, chk_nj, chk_nk, chk_nvars,
                gfs->gs, gfs->domain.mpi_size, gfs->domain.global_ni,
                gfs->domain.global_nj, gfs->domain.global_nk,
                gfs->n_evol_vars);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    *t  = chk_t;
    *it = chk_it;

    /* ── Read evolved variable data ─────────────────────────────────── */
    hid_t evol_id;
    HDF5_CHK(evol_id = H5Gopen(file_id, "/evolved", H5P_DEFAULT));

    for (int v = 0; v < gfs->n_evol_vars; v++)
    {
        char dsname[BUFF_LEN];
        snprintf(dsname, BUFF_LEN, "%s", gfs->vars[v]->vname);
        HDF5_CHK(H5LTread_dataset_double(evol_id, dsname,
                                          gfs->vars[v]->new));
    }

    HDF5_CHK(H5Gclose(evol_id));

    /* ── Read material property arrays (ieps, imu, sigma) ──────────── */
    hid_t mat_id;
    HDF5_CHK(mat_id = H5Gopen(file_id, "/material", H5P_DEFAULT));

    for (int v = 0; v < 3; v++)
    {
        char dsname[BUFF_LEN];
        snprintf(dsname, BUFF_LEN, "%s", gfs->auxvars[v]->vname);
        HDF5_CHK(H5LTread_dataset_double(mat_id, dsname,
                                          gfs->auxvars[v]->new));
    }

    HDF5_CHK(H5Gclose(mat_id));
    HDF5_CHK(H5Fclose(file_id));

    if (gfs->domain.rank == 0)
        fprintf(stderr, "Recovered from checkpoint: t = %g, iteration %d\n",
                *t, *it);

    return chk_errors;
}
