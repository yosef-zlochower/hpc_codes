#include "gf.hpp"
#include <cassert>
#include <cstdio>
#include <exception>
#include <string>
#include <mpi.h>

/* Set up CommAxis device buffers + their host mirrors for one axis.
 * Both faces (lower/upper) share the same shape; both ends pack/unpack
 * `maxvars` variables back-to-back, so the total allocation per face is
 *   face_size * maxvars   doubles
 * with the per-variable face_size stored on the struct so the
 * pack/unpack offset arithmetic in comm.cpp stays per-variable. */
static void comm_axis_alloc(CommAxis *ax, size_t face_size, size_t maxvars,
                            const char *axname)
{
    ax->face_size = face_size;
    const size_t total = face_size * maxvars;
    char buf[64];
    snprintf(buf, sizeof buf, "comm_%s_src_lo", axname);
    ax->src_lo_dev = Field1D(buf, total);
    snprintf(buf, sizeof buf, "comm_%s_dst_lo", axname);
    ax->dst_lo_dev = Field1D(buf, total);
    snprintf(buf, sizeof buf, "comm_%s_src_up", axname);
    ax->src_up_dev = Field1D(buf, total);
    snprintf(buf, sizeof buf, "comm_%s_dst_up", axname);
    ax->dst_up_dev = Field1D(buf, total);

    ax->src_lo_host = Kokkos::create_mirror_view(ax->src_lo_dev);
    ax->dst_lo_host = Kokkos::create_mirror_view(ax->dst_lo_dev);
    ax->src_up_host = Kokkos::create_mirror_view(ax->src_up_dev);
    ax->dst_up_host = Kokkos::create_mirror_view(ax->dst_up_dev);
}

static void comm_axis_free(CommAxis *ax)
{
    /* Reset to default-constructed (zero-extent) Views; the underlying
     * allocations release when the last handle goes out of scope. */
    *ax = CommAxis{};
}

int ngfs_allocate(int n_evol, int n_aux, NGFS *gfs)
{
    assert(gfs->evol == nullptr);
    assert(gfs->aux  == nullptr);

    gfs->n_evol_vars = n_evol;
    gfs->n_aux_vars  = n_aux;
    gfs->nx    = gfs->domain.nx;
    gfs->ny    = gfs->domain.ny;
    gfs->nz    = gfs->domain.nz;
    gfs->n_tot = gfs->nx * gfs->ny * gfs->nz;
    gfs->gs    = gfs->domain.gs;
    gfs->dx    = gfs->domain.dx;
    gfs->dy    = gfs->domain.dy;
    gfs->dz    = gfs->domain.dz;
    gfs->x0    = gfs->domain.global_x0 + gfs->domain.local_i0 * gfs->domain.dx;
    gfs->y0    = gfs->domain.global_y0 + gfs->domain.local_j0 * gfs->domain.dy;
    gfs->z0    = gfs->domain.global_z0 + gfs->domain.local_k0 * gfs->domain.dz;

    /* All allocations below can throw: operator new[] raises
     * std::bad_alloc and a Kokkos::View constructor raises on a failed
     * device/host allocation.  An uncaught throw here would unwind into
     * std::terminate(), producing an opaque crash with no indication of
     * which rank ran out of memory.  Catch and abort the whole job with
     * a rank-tagged diagnostic instead (the C-version equivalent of the
     * CHECK_ALLOC macro). */
    try
    {
        /* Per-face buffer holds doubles for max(n_evol, n_aux) variables.
         * face_size on the struct is the per-variable face size (in
         * doubles); the actual allocation is face_size * maxvars. */
        const size_t maxvars = (size_t)((n_evol > n_aux) ? n_evol : n_aux);
        comm_axis_alloc(&gfs->comm_x, gfs->gs * gfs->ny * gfs->nz, maxvars, "x");
        comm_axis_alloc(&gfs->comm_y, gfs->nx * gfs->gs * gfs->nz, maxvars, "y");
        comm_axis_alloc(&gfs->comm_z, gfs->nx * gfs->ny * gfs->gs, maxvars, "z");

        gfs->evol = new EvolField[n_evol]();
        gfs->aux  = new AuxField [n_aux]();

        const int64_t nx = gfs->nx, ny = gfs->ny, nz = gfs->nz;

        char buf[128];
        for (int i = 0; i < n_evol; i++)
        {
            snprintf(buf, sizeof buf, "Var%d", i);
            gfs->evol[i].vname = buf;
            snprintf(buf, sizeof buf, "evol_%d_state", i);
            gfs->evol[i].state = Field3D(buf, nx, ny, nz);
            snprintf(buf, sizeof buf, "evol_%d_old",   i);
            gfs->evol[i].old_  = Field3D(buf, nx, ny, nz);
            for (int s = 0; s < 4; s++)
            {
                snprintf(buf, sizeof buf, "evol_%d_K%d", i, s + 1);
                gfs->evol[i].K[s] = Field3D(buf, nx, ny, nz);
            }
        }

        for (int i = 0; i < n_aux; i++)
        {
            snprintf(buf, sizeof buf, "Aux%d", i);
            gfs->aux[i].vname = buf;
            snprintf(buf, sizeof buf, "aux_%d_state", i);
            gfs->aux[i].state = Field3D(buf, nx, ny, nz);
        }
    }
    catch (const std::exception &e)
    {
        int rk = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rk);
        std::fprintf(stderr,
                     "rank %d: ngfs_allocate failed (n_evol=%d n_aux=%d): %s\n",
                     rk, n_evol, n_aux, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return 0;
}

int ngfs_deallocate(NGFS *gfs)
{
    if (gfs->evol)
    {
        delete[] gfs->evol;
        gfs->evol = nullptr;
    }
    if (gfs->aux)
    {
        delete[] gfs->aux;
        gfs->aux = nullptr;
    }
    gfs->n_evol_vars = 0;
    gfs->n_aux_vars  = 0;
    gfs->n_tot       = 0;
    gfs->gs          = 0;

    comm_axis_free(&gfs->comm_x);
    comm_axis_free(&gfs->comm_y);
    comm_axis_free(&gfs->comm_z);
    return 0;
}

void gf_rename_evol(NGFS *gfs, int slot, const char *name)
{
    gfs->evol[slot].vname = name;
}

void gf_rename_aux(NGFS *gfs, int slot, const char *name)
{
    gfs->aux[slot].vname = name;
}
