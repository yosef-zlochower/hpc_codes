#include "comm.h"
#include "gf.h"
#include "timer.h"
#include <assert.h>
#include <mpi.h>
#include <stdio.h>

/* Pack data from a grid array into a linear buffer (MODE_PACK)
 * or unpack from a linear buffer into the grid array (MODE_UNPACK).
 * Iterates over the 3D index box [ks,ke) x [js,je) x [is,ie). */
static void transfer_data(double *val, double *buffer, IndexBox box,
                           int64_t nx, int64_t ny, TransferMode mode)
{
    int64_t buf_idx = 0;
    for (int64_t k = box.ks; k < box.ke; k++)
    {
        for (int64_t j = box.js; j < box.je; j++)
        {
            for (int64_t i = box.is; i < box.ie; i++)
            {
                int64_t grid_idx = i + j * nx + k * nx * ny;
                if (mode == MODE_PACK)
                {
                    buffer[buf_idx++] = val[grid_idx];
                }
                else
                {
                    val[grid_idx] = buffer[buf_idx++];
                }
            }
        }
    }
}

/* Size (in doubles) of one IndexBox slab for a single variable */
static size_t box_size(IndexBox box)
{
    return (box.ie - box.is) * (box.je - box.js) * (box.ke - box.ks);
}

/* Exchange ghost zones along a single axis for nvars variables.
 * Packs all variables into the axis's send buffers (ax->lower.src and
 * ax->upper.src), performs non-blocking MPI communication, then unpacks
 * received data from ax->lower.dst / ax->upper.dst. */
static void exchange_direction(struct ngfs *gfs, int nvars, int vstart,
                                int rank_lower, int rank_upper,
                                struct comm_axis *ax,
                                IndexBox box_send_lower, IndexBox box_recv_lower,
                                IndexBox box_send_upper, IndexBox box_recv_upper,
                                int copy_timer, int send_timer,
                                int recv_timer, int wait_timer)
{
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const size_t face_size = box_size(box_send_lower); /* same for all boxes */
    const size_t total_buff_size = face_size * nvars;
    MPI_Comm cart_comm = gfs->domain.cart_comm;

    MPI_Request send_reqs[2];
    MPI_Request recv_reqs[2];
    int n_send = 0;
    int n_recv = 0;

    /* Pack data for all variables into send buffers */
    BEGIN_TIMER(copy_timer)
    {
        if (rank_lower > INVALID_RANK)
        {
            for (int v = 0; v < nvars; v++)
            {
                transfer_data(gfs->vars[v + vstart]->dot,
                              ax->lower.src + v * face_size,
                              box_send_lower, nx, ny, MODE_PACK);
            }
        }

        if (rank_upper > INVALID_RANK)
        {
            for (int v = 0; v < nvars; v++)
            {
                transfer_data(gfs->vars[v + vstart]->dot,
                              ax->upper.src + v * face_size,
                              box_send_upper, nx, ny, MODE_PACK);
            }
        }
    }
    END_TIMER(copy_timer)

    /* Non-blocking sends and receives.
     * Tags distinguish direction so that self-communication (periodic BC
     * with 1 process per axis) doesn't swap the two messages.
     *   Tag 0: data sent toward the lower neighbour
     *   Tag 1: data sent toward the upper neighbour
     * A recv from the lower neighbour expects tag 1 (the lower neighbour's
     * upper-direction send), and vice versa. */
    BEGIN_TIMER(send_timer)
    {
        if (rank_lower > INVALID_RANK)
        {
            MPI_ERROR(MPI_Isend(ax->lower.src, total_buff_size, MPI_DOUBLE,
                                rank_lower, 0, cart_comm,
                                &send_reqs[n_send++]));
        }
        if (rank_upper > INVALID_RANK)
        {
            MPI_ERROR(MPI_Isend(ax->upper.src, total_buff_size, MPI_DOUBLE,
                                rank_upper, 1, cart_comm,
                                &send_reqs[n_send++]));
        }
    }
    END_TIMER(send_timer)

    BEGIN_TIMER(recv_timer)
    {
        if (rank_lower > INVALID_RANK)
        {
            MPI_ERROR(MPI_Irecv(ax->lower.dst, total_buff_size, MPI_DOUBLE,
                                rank_lower, 1, cart_comm,
                                &recv_reqs[n_recv++]));
        }
        if (rank_upper > INVALID_RANK)
        {
            MPI_ERROR(MPI_Irecv(ax->upper.dst, total_buff_size, MPI_DOUBLE,
                                rank_upper, 0, cart_comm,
                                &recv_reqs[n_recv++]));
        }

        /* Wait for all receives to complete before unpacking */
        MPI_ERROR(MPI_Waitall(n_recv, recv_reqs, MPI_STATUSES_IGNORE));
    }
    END_TIMER(recv_timer)

    /* Unpack received data for all variables */
    BEGIN_TIMER(copy_timer)
    {
        if (rank_lower > INVALID_RANK)
        {
            for (int v = 0; v < nvars; v++)
            {
                transfer_data(gfs->vars[v + vstart]->dot,
                              ax->lower.dst + v * face_size,
                              box_recv_lower, nx, ny, MODE_UNPACK);
            }
        }

        if (rank_upper > INVALID_RANK)
        {
            for (int v = 0; v < nvars; v++)
            {
                transfer_data(gfs->vars[v + vstart]->dot,
                              ax->upper.dst + v * face_size,
                              box_recv_upper, nx, ny, MODE_UNPACK);
            }
        }
    }
    END_TIMER(copy_timer)

    /* Wait for sends to complete before buffers can be reused */
    BEGIN_TIMER(wait_timer)
    {
        MPI_ERROR(MPI_Waitall(n_send, send_reqs, MPI_STATUSES_IGNORE));
    }
    END_TIMER(wait_timer)
}

int sync_vars(struct ngfs *gfs, enum var_type type)
{
    static int timer_x = -1;
    static int timer_y = -1;
    static int timer_z = -1;
    static int copy_timer = -1;
    static int recv_timer = -1;
    static int send_timer = -1;
    static int wait_timer = -1;
    if (timer_x < 0)
    {
        timer_x = register_timer("/sync/x");
        timer_y = register_timer("/sync/y");
        timer_z = register_timer("/sync/z");
        copy_timer = register_timer("/sync/data_copy");
        recv_timer = register_timer("/sync/recv");
        send_timer = register_timer("/sync/send");
        wait_timer = register_timer("/sync/wait");
    }

    const int nvars = type == EVOLVED ? gfs->n_evol_vars : gfs->n_aux_vars;
    const int vstart = type == EVOLVED ? 0 : gfs->n_evol_vars;
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t nz = gfs->nz;
    const int gs = gfs->gs;

    /* ── X direction ────────────────────────────────────────────── */
    IndexBox x_send_lo = { .is = gs,         .ie = 2*gs, .js = 0, .je = ny, .ks = 0, .ke = nz };
    IndexBox x_recv_lo = { .is = 0,          .ie = gs,   .js = 0, .je = ny, .ks = 0, .ke = nz };
    IndexBox x_send_up = { .is = nx - 2*gs,  .ie = nx-gs,.js = 0, .je = ny, .ks = 0, .ke = nz };
    IndexBox x_recv_up = { .is = nx - gs,    .ie = nx,   .js = 0, .je = ny, .ks = 0, .ke = nz };

    BEGIN_TIMER(timer_x)
    {
        exchange_direction(gfs, nvars, vstart,
                           gfs->domain.lower_x_rank, gfs->domain.upper_x_rank,
                           &gfs->comm_x,
                           x_send_lo, x_recv_lo, x_send_up, x_recv_up,
                           copy_timer, send_timer, recv_timer, wait_timer);
    }
    END_TIMER(timer_x);

    /* ── Y direction ────────────────────────────────────────────── */
    IndexBox y_send_lo = { .is = 0, .ie = nx, .js = gs,        .je = 2*gs,  .ks = 0, .ke = nz };
    IndexBox y_recv_lo = { .is = 0, .ie = nx, .js = 0,         .je = gs,    .ks = 0, .ke = nz };
    IndexBox y_send_up = { .is = 0, .ie = nx, .js = ny - 2*gs, .je = ny-gs, .ks = 0, .ke = nz };
    IndexBox y_recv_up = { .is = 0, .ie = nx, .js = ny - gs,   .je = ny,    .ks = 0, .ke = nz };

    BEGIN_TIMER(timer_y)
    {
        exchange_direction(gfs, nvars, vstart,
                           gfs->domain.lower_y_rank, gfs->domain.upper_y_rank,
                           &gfs->comm_y,
                           y_send_lo, y_recv_lo, y_send_up, y_recv_up,
                           copy_timer, send_timer, recv_timer, wait_timer);
    }
    END_TIMER(timer_y)

    /* ── Z direction ────────────────────────────────────────────── */
    IndexBox z_send_lo = { .is = 0, .ie = nx, .js = 0, .je = ny, .ks = gs,        .ke = 2*gs };
    IndexBox z_recv_lo = { .is = 0, .ie = nx, .js = 0, .je = ny, .ks = 0,         .ke = gs };
    IndexBox z_send_up = { .is = 0, .ie = nx, .js = 0, .je = ny, .ks = nz - 2*gs, .ke = nz-gs };
    IndexBox z_recv_up = { .is = 0, .ie = nx, .js = 0, .je = ny, .ks = nz - gs,   .ke = nz };

    BEGIN_TIMER(timer_z)
    {
        exchange_direction(gfs, nvars, vstart,
                           gfs->domain.lower_z_rank, gfs->domain.upper_z_rank,
                           &gfs->comm_z,
                           z_send_lo, z_recv_lo, z_send_up, z_recv_up,
                           copy_timer, send_timer, recv_timer, wait_timer);
    }
    END_TIMER(timer_z)
    return 0;
}
