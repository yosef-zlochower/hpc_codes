#include "comm.hpp"
#include "timer.h"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cassert>
#include <cstdio>

/* ── Pack / unpack via a Kokkos kernel over the IndexBox ───────────────
 *
 * The buffer is laid out as
 *     [v=0 face data][v=1 face data]...[v=nvars-1 face data]
 * with each per-variable face stored in (k, j, i) iteration order.
 * `face_size` is the per-variable size in doubles.
 *
 * On a single GPU build with MAXWELL_CUDA_AWARE_MPI=ON the device
 * pointer is handed straight to MPI; otherwise the device buffer is
 * deep-copied to its host mirror (and back, after recv) before/after
 * the MPI call.
 */

using Pack3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

static void pack_box_into(NGFS *gfs, int vstart, int nvars, int kidx,
                          var_type type, IndexBox box,
                          Field1D dst, size_t face_size)
{
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t is = box.is, js = box.js, ks = box.ks;
    Pack3D pol({box.is, box.js, box.ks}, {box.ie, box.je, box.ke});

    for (int v = 0; v < nvars; v++)
    {
        const double *src;
        if (type == AUX)
            src = gfs->aux[v + vstart].state.data();
        else if (kidx < 0)
            src = gfs->evol[v + vstart].state.data();
        else
            src = gfs->evol[v + vstart].K[kidx].data();
        const size_t offset = (size_t)v * face_size;
        Kokkos::parallel_for("comm_pack", pol,
            KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                const int64_t grid_idx = i + j * nx + k * nx * ny;
                const int64_t buf_idx  = (i - is)
                                       + (j - js) * (box.ie - box.is)
                                       + (k - ks) * (box.ie - box.is)
                                                  * (box.je - box.js);
                dst(offset + buf_idx) = src[grid_idx];
            });
    }
}

static void unpack_box_from(NGFS *gfs, int vstart, int nvars, int kidx,
                            var_type type, IndexBox box,
                            Field1D src, size_t face_size)
{
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t is = box.is, js = box.js, ks = box.ks;
    Pack3D pol({box.is, box.js, box.ks}, {box.ie, box.je, box.ke});

    for (int v = 0; v < nvars; v++)
    {
        double *dst;
        if (type == AUX)
            dst = gfs->aux[v + vstart].state.data();
        else if (kidx < 0)
            dst = gfs->evol[v + vstart].state.data();
        else
            dst = gfs->evol[v + vstart].K[kidx].data();
        const size_t offset = (size_t)v * face_size;
        Kokkos::parallel_for("comm_unpack", pol,
            KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                const int64_t grid_idx = i + j * nx + k * nx * ny;
                const int64_t buf_idx  = (i - is)
                                       + (j - js) * (box.ie - box.is)
                                       + (k - ks) * (box.ie - box.is)
                                                  * (box.je - box.js);
                dst[grid_idx] = src(offset + buf_idx);
            });
    }
}

/* Per-axis state used to drive non-blocking exchanges in parallel. */
struct AxisOp
{
    CommAxis *ax;
    IndexBox  send_lo, recv_lo, send_up, recv_up;
    int       rank_lo, rank_up;
    bool      active;
};

/* Returns the byte/double pointer that should be handed to MPI for a
 * given device buffer.  Without CUDA-aware MPI this triggers a host
 * mirror deep_copy and returns the host pointer; with CUDA-aware MPI
 * (built with -DMAXWELL_CUDA_AWARE_MPI=ON) it returns the device
 * pointer directly. */
static double* mpi_send_ptr(Field1D dev, Field1D::host_mirror_type host)
{
#ifdef MAXWELL_CUDA_AWARE_MPI
    (void)host;
    return dev.data();
#else
    Kokkos::deep_copy(host, dev);
    return host.data();
#endif
}

static double* mpi_recv_ptr(Field1D dev, Field1D::host_mirror_type host)
{
#ifdef MAXWELL_CUDA_AWARE_MPI
    (void)host;
    return dev.data();
#else
    (void)dev;
    return host.data();
#endif
}

static void post_recvs_for_axis(AxisOp &op, int total_buf_size,
                                MPI_Comm cart_comm,
                                MPI_Request reqs[], int &n_recv)
{
    if (!op.active) return;
    if (op.rank_lo > INVALID_RANK)
    {
        MPI_ERROR(MPI_Irecv(mpi_recv_ptr(op.ax->dst_lo_dev,
                                         op.ax->dst_lo_host),
                            total_buf_size, MPI_DOUBLE, op.rank_lo, 1,
                            cart_comm, &reqs[n_recv++]));
    }
    if (op.rank_up > INVALID_RANK)
    {
        MPI_ERROR(MPI_Irecv(mpi_recv_ptr(op.ax->dst_up_dev,
                                         op.ax->dst_up_host),
                            total_buf_size, MPI_DOUBLE, op.rank_up, 0,
                            cart_comm, &reqs[n_recv++]));
    }
}

static void post_sends_for_axis(AxisOp &op, int total_buf_size,
                                MPI_Comm cart_comm,
                                MPI_Request reqs[], int &n_send)
{
    if (!op.active) return;
    if (op.rank_lo > INVALID_RANK)
    {
        MPI_ERROR(MPI_Isend(mpi_send_ptr(op.ax->src_lo_dev,
                                         op.ax->src_lo_host),
                            total_buf_size, MPI_DOUBLE, op.rank_lo, 0,
                            cart_comm, &reqs[n_send++]));
    }
    if (op.rank_up > INVALID_RANK)
    {
        MPI_ERROR(MPI_Isend(mpi_send_ptr(op.ax->src_up_dev,
                                         op.ax->src_up_host),
                            total_buf_size, MPI_DOUBLE, op.rank_up, 1,
                            cart_comm, &reqs[n_send++]));
    }
}

int sync_vars(NGFS *gfs, var_type type, int kidx)
{
    static int t_total = -1, t_pack = -1, t_unpack = -1, t_wait = -1;
    if (t_total < 0)
    {
        t_total  = register_timer("/sync/total");
        t_pack   = register_timer("/sync/pack");
        t_unpack = register_timer("/sync/unpack");
        t_wait   = register_timer("/sync/wait");
    }

    const int nvars  = (type == EVOLVED) ? gfs->n_evol_vars : gfs->n_aux_vars;
    const int vstart = 0;
    const int64_t nx = gfs->nx, ny = gfs->ny, nz = gfs->nz;
    const int gs = gfs->gs;
    MPI_Comm cart_comm = gfs->domain.cart_comm;

    BEGIN_TIMER(t_total)
    {

    /* Per-axis box geometry. */
    AxisOp ax_x{ &gfs->comm_x,
        IndexBox{gs,        2*gs, 0,  ny,    0,  nz},
        IndexBox{0,         gs,   0,  ny,    0,  nz},
        IndexBox{nx-2*gs,   nx-gs,0,  ny,    0,  nz},
        IndexBox{nx-gs,     nx,   0,  ny,    0,  nz},
        gfs->domain.lower_x_rank, gfs->domain.upper_x_rank, true };

    AxisOp ax_y{ &gfs->comm_y,
        IndexBox{0, nx,  gs,        2*gs,  0,  nz},
        IndexBox{0, nx,  0,         gs,    0,  nz},
        IndexBox{0, nx,  ny-2*gs,   ny-gs, 0,  nz},
        IndexBox{0, nx,  ny-gs,     ny,    0,  nz},
        gfs->domain.lower_y_rank, gfs->domain.upper_y_rank, true };

    AxisOp ax_z{ &gfs->comm_z,
        IndexBox{0, nx, 0, ny, gs,        2*gs},
        IndexBox{0, nx, 0, ny, 0,         gs},
        IndexBox{0, nx, 0, ny, nz-2*gs,   nz-gs},
        IndexBox{0, nx, 0, ny, nz-gs,     nz},
        gfs->domain.lower_z_rank, gfs->domain.upper_z_rank, true };

    AxisOp axes[3] = { ax_x, ax_y, ax_z };

    /* Sequential x → y → z sync.
     *
     * Corner ghosts are filled by reading x-ghost columns into the
     * y-send box (and similarly z-send reads x- and y-ghost data),
     * which only works if x is synced before y, and y before z. The
     * Maxwell production stencils are axis-aligned and never read
     * corner ghosts, so a fully parallel three-axis sync would be
     * functionally correct for the PDE solve — but the test_sync
     * harness corrupts every ghost cell (including corners) and
     * verifies all of them, and any hypothetical mixed-derivative
     * stencil would also need them. Stay sequential to match the C
     * version's semantics. */
    for (int a = 0; a < 3; a++)
    {
        AxisOp &op = axes[a];
        CommAxis *ax = op.ax;
        const size_t total_buf_size = ax->face_size * (size_t)nvars;

        BEGIN_TIMER(t_pack)
        {
            if (op.rank_lo > INVALID_RANK)
                pack_box_into(gfs, vstart, nvars, kidx, type, op.send_lo,
                              ax->src_lo_dev, ax->face_size);
            if (op.rank_up > INVALID_RANK)
                pack_box_into(gfs, vstart, nvars, kidx, type, op.send_up,
                              ax->src_up_dev, ax->face_size);
            Kokkos::fence();
        }
        END_TIMER(t_pack)

        MPI_Request recv_reqs[2];
        MPI_Request send_reqs[2];
        int n_recv = 0, n_send = 0;
        post_recvs_for_axis(op, (int)total_buf_size, cart_comm,
                            recv_reqs, n_recv);
        post_sends_for_axis(op, (int)total_buf_size, cart_comm,
                            send_reqs, n_send);

        BEGIN_TIMER(t_wait)
        {
            MPI_ERROR(MPI_Waitall(n_recv, recv_reqs, MPI_STATUSES_IGNORE));
        }
        END_TIMER(t_wait)

        BEGIN_TIMER(t_unpack)
        {
#ifndef MAXWELL_CUDA_AWARE_MPI
            if (op.rank_lo > INVALID_RANK)
                Kokkos::deep_copy(ax->dst_lo_dev, ax->dst_lo_host);
            if (op.rank_up > INVALID_RANK)
                Kokkos::deep_copy(ax->dst_up_dev, ax->dst_up_host);
#endif
            if (op.rank_lo > INVALID_RANK)
                unpack_box_from(gfs, vstart, nvars, kidx, type, op.recv_lo,
                                ax->dst_lo_dev, ax->face_size);
            if (op.rank_up > INVALID_RANK)
                unpack_box_from(gfs, vstart, nvars, kidx, type, op.recv_up,
                                ax->dst_up_dev, ax->face_size);
            Kokkos::fence();
        }
        END_TIMER(t_unpack)

        MPI_ERROR(MPI_Waitall(n_send, send_reqs, MPI_STATUSES_IGNORE));
    }

    }
    END_TIMER(t_total)
    return 0;
}
