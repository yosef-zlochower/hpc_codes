#include "comm.h"
#include "domain.h"
#include "gf.h"
#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/******************************************************************
* Purpose: Synchronise the ghost zones of a 2D grid variable with
*     all MPI neighbours. Packs interior-edge data into send buffers,
*     performs non-blocking sends and receives in the x-direction
*     then y-direction, and unpacks received data into the local
*     ghost-zone regions. After this call all ghost zones adjacent to
*     MPI neighbours hold current values from those neighbours.
* Input Variables:
*     gfs: struct ngfs_2d*, grid function container; communication
*         buffers and domain neighbour ranks must already be set up
*     var: int, index of the variable in gfs->vars[] to synchronise
* Output Variables:
*     gfs->vars[var]->val: double*, ghost-zone regions updated with
*         data from MPI neighbours
* Return Values and indicators of success / failure
*     void. Side effects: two rounds of MPI_Isend/MPI_Irecv/
*     MPI_Waitall (x then y).
*******************************************************************/
void sync_var_2d(struct ngfs_2d *gfs, int var)
{
    /* --- 1. Setup and Dimensions --- */
    const int gs = gfs->gs;
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    double *val = gfs->vars[var]->val;

    const int lower_x_rank = gfs->domain.lower_x_rank;
    const int upper_x_rank = gfs->domain.upper_x_rank;
    const int lower_y_rank = gfs->domain.lower_y_rank;
    const int upper_y_rank = gfs->domain.upper_y_rank;

    /* Calculate buffer sizes */
    const size_t buff_x_size = gfs->buff_x_size;
    const size_t buff_y_size = gfs->buff_y_size;

    double *lower_x_src = gfs->lower_x_src;
    double *lower_x_dst = gfs->lower_x_dst;
    double *upper_x_src = gfs->upper_x_src;
    double *upper_x_dst = gfs->upper_x_dst;

    double *lower_y_src = gfs->lower_y_src;
    double *lower_y_dst = gfs->lower_y_dst;
    double *upper_y_src = gfs->upper_y_src;
    double *upper_y_dst = gfs->upper_y_dst;
    MPI_Comm cart_comm = gfs->domain.cart_comm;

    MPI_Request send_requests[2];
    MPI_Request recv_requests[2];
    int num_requests = 0;

    /* =========================================================
     * X-DIRECTION SYNCHRONIZATION
     * ========================================================= */

    if (lower_x_rank != INVALID_RANK)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int i = 0; i < gs; i++)
            {
                const int64_t ij = gf_indx_2d(gfs, gs + i, j);
                lower_x_src[i + j * gs] = val[ij];
            }
        }
        MPI_Isend(lower_x_src, buff_x_size, MPI_DOUBLE, lower_x_rank, 0,
                  cart_comm, send_requests + num_requests);
        MPI_Irecv(lower_x_dst, buff_x_size, MPI_DOUBLE, lower_x_rank, 0,
                  cart_comm, recv_requests + num_requests);
        num_requests++;
    }

    if (upper_x_rank != INVALID_RANK)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int i = 0; i < gs; i++)
            {
                const int64_t ij = gf_indx_2d(gfs, i + nx - 2 * gs, j);
                upper_x_src[i + j * gs] = val[ij];
            }
        }
        MPI_Isend(upper_x_src, buff_x_size, MPI_DOUBLE, upper_x_rank, 0,
                  cart_comm, send_requests + num_requests);
        MPI_Irecv(upper_x_dst, buff_x_size, MPI_DOUBLE, upper_x_rank, 0,
                  cart_comm, recv_requests + num_requests);
        num_requests++;
    }

    assert(num_requests <= 2);
    MPI_Waitall(num_requests, recv_requests, MPI_STATUSES_IGNORE);

    if (lower_x_rank != INVALID_RANK)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int i = 0; i < gs; i++)
            {
                const int64_t ij = gf_indx_2d(gfs, i, j);
                val[ij] = lower_x_dst[i + j * gs];
            }
        }
    }

    if (upper_x_rank != INVALID_RANK)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int i = 0; i < gs; i++)
            {
                const int64_t ij = gf_indx_2d(gfs, i + nx - gs, j);
                val[ij] = upper_x_dst[i + j * gs];
            }
        }
    }

    MPI_Waitall(num_requests, send_requests, MPI_STATUSES_IGNORE);
    num_requests = 0;

    /* =========================================================
     * Y-DIRECTION SYNCHRONIZATION (UPDATED)
     * ========================================================= */

    /* Pack and Send to Lower Y (Bottom) */
    if (lower_y_rank != INVALID_RANK)
    {
        for (int j = 0; j < gs; j++)
        {
            for (int i = 0; i < nx; i++)
            {
                const int64_t ij = gf_indx_2d(gfs, i, gs + j);

                lower_y_src[i + j * nx] = val[ij];
            }
        }
        MPI_Isend(lower_y_src, buff_y_size, MPI_DOUBLE, lower_y_rank, 0,
                  cart_comm, send_requests + num_requests);
        MPI_Irecv(lower_y_dst, buff_y_size, MPI_DOUBLE, lower_y_rank, 0,
                  cart_comm, recv_requests + num_requests);
        num_requests++;
    }

    /* Pack and Send to Upper Y (Top) */
    if (upper_y_rank != INVALID_RANK)
    {
        for (int j = 0; j < gs; j++)
        {
            for (int i = 0; i < nx; i++)
            {
                const int64_t ij = gf_indx_2d(gfs, i, j + ny - 2 * gs);

                upper_y_src[i + j * nx] = val[ij];
            }
        }
        MPI_Isend(upper_y_src, buff_y_size, MPI_DOUBLE, upper_y_rank, 0,
                  cart_comm, send_requests + num_requests);
        MPI_Irecv(upper_y_dst, buff_y_size, MPI_DOUBLE, upper_y_rank, 0,
                  cart_comm, recv_requests + num_requests);
        num_requests++;
    }

    assert(num_requests <= 2);
    MPI_Waitall(num_requests, recv_requests, MPI_STATUSES_IGNORE);

    /* Unpack Lower Y */
    if (lower_y_rank != INVALID_RANK)
    {
        for (int j = 0; j < gs; j++)
        {
            for (int i = 0; i < nx; i++)
            {
                const int64_t ij = gf_indx_2d(gfs, i, j);

                // UPDATED: Row-major indexing [i + j * nx]
                val[ij] = lower_y_dst[i + j * nx];
            }
        }
    }

    /* Unpack Upper Y */
    if (upper_y_rank != INVALID_RANK)
    {
        for (int j = 0; j < gs; j++)
        {
            for (int i = 0; i < nx; i++)
            {
                const int64_t ij = gf_indx_2d(gfs, i, j + ny - gs);
                val[ij] = upper_y_dst[i + j * nx];
            }
        }
    }

    MPI_Waitall(num_requests, send_requests, MPI_STATUSES_IGNORE);
}


/* --- Core Logic --- */

/******************************************************************
* Purpose: Pack data from a grid array into a linear buffer
*     (MODE_PACK) or unpack from a linear buffer into the grid array
*     (MODE_UNPACK). Iterates over the 3D index box
*     [ks,ke) x [js,je) x [is,ie) in k-j-i order (k outermost) for
*     optimal cache locality given row-major layout (i is
*     fastest-varying index).
* Input Variables:
*     val: double*, flat grid array, length nx*ny*nz
*     buffer: double*, linear communication buffer
*     box: IndexBox, iteration bounds [ks,ke)x[js,je)x[is,ie)
*     nx: int64_t, local x extent
*     ny: int64_t, local y extent
*     mode: TransferMode, MODE_PACK reads val->buffer;
*         MODE_UNPACK writes buffer->val
* Output Variables:
*     In MODE_PACK: buffer is written.
*     In MODE_UNPACK: val is written at the grid points covered by
*         box.
* Return Values and indicators of success / failure
*     void
*******************************************************************/
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

                // Calculate grid index assuming standard 3D mapping
                // For 2D, k is 0, so this collapses cleanly.
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

/******************************************************************
* Purpose: Execute one full halo exchange along a single axis
*     direction. Packs and sends to the lower and upper neighbours
*     simultaneously using non-blocking MPI, waits for receives to
*     complete, then unpacks received data.
* Input Variables:
*     val: double*, grid data array
*     nx: int64_t, local x extent
*     ny: int64_t, local y extent
*     rank_lower: int, lower neighbour rank; INVALID_RANK(-1) means
*         no neighbour
*     rank_upper: int, upper neighbour rank; INVALID_RANK(-1) means
*         no neighbour
*     buf_send_lower: double*, pre-allocated send buffer for lower
*         neighbour
*     buf_recv_lower: double*, pre-allocated receive buffer for lower
*         neighbour
*     buf_send_upper: double*, pre-allocated send buffer for upper
*         neighbour
*     buf_recv_upper: double*, pre-allocated receive buffer for upper
*         neighbour
*     buff_size: size_t, number of doubles in each buffer
*     box_send_lower: IndexBox, iteration box for packing data to
*         send to lower neighbour
*     box_recv_lower: IndexBox, iteration box for unpacking data
*         received from lower neighbour
*     box_send_upper: IndexBox, iteration box for packing data to
*         send to upper neighbour
*     box_recv_upper: IndexBox, iteration box for unpacking data
*         received from upper neighbour
*     cart_comm: MPI_Comm, Cartesian communicator
* Output Variables:
*     val: double*, updated at the receive-box indices with data from
*         the corresponding neighbours
* Return Values and indicators of success / failure
*     void. Side effects: MPI_Isend, MPI_Irecv, MPI_Waitall calls.
*******************************************************************/
static void exchange_direction(double *val, int64_t nx, int64_t ny,
                               int rank_lower, int rank_upper,
                               double *buf_send_lower, double *buf_recv_lower,
                               double *buf_send_upper, double *buf_recv_upper,
                               size_t buff_size, IndexBox box_send_lower,
                               IndexBox box_recv_lower, IndexBox box_send_upper,
                               IndexBox box_recv_upper, MPI_Comm cart_comm)
{
    MPI_Request send_reqs[2];
    MPI_Request recv_reqs[2];
    int n_req = 0;

    // 1. Pack and Start Sends/Recvs
    if (rank_lower != INVALID_RANK)
    {
        transfer_data(val, buf_send_lower, box_send_lower, nx, ny, MODE_PACK);
        MPI_Isend(buf_send_lower, buff_size, MPI_DOUBLE, rank_lower, 0,
                  cart_comm, &send_reqs[n_req]);
        MPI_Irecv(buf_recv_lower, buff_size, MPI_DOUBLE, rank_lower, 0,
                  cart_comm, &recv_reqs[n_req]);
        n_req++;
    }

    if (rank_upper != INVALID_RANK)
    {
        transfer_data(val, buf_send_upper, box_send_upper, nx, ny, MODE_PACK);
        MPI_Isend(buf_send_upper, buff_size, MPI_DOUBLE, rank_upper, 0,
                  cart_comm, &send_reqs[n_req]);
        MPI_Irecv(buf_recv_upper, buff_size, MPI_DOUBLE, rank_upper, 0,
                  cart_comm, &recv_reqs[n_req]);
        n_req++;
    }

    // 2. Wait for data to arrive
    MPI_Waitall(n_req, recv_reqs, MPI_STATUSES_IGNORE);

    // 3. Unpack received data
    if (rank_lower != INVALID_RANK)
    {
        transfer_data(val, buf_recv_lower, box_recv_lower, nx, ny, MODE_UNPACK);
    }

    if (rank_upper != INVALID_RANK)
    {
        transfer_data(val, buf_recv_upper, box_recv_upper, nx, ny, MODE_UNPACK);
    }
    MPI_Waitall(n_req, send_reqs, MPI_STATUSES_IGNORE);
}

/* --- Main Interface --- */

/******************************************************************
* Purpose: Synchronise the ghost zones of a 3D grid variable with
*     all MPI neighbours. Uses the generic transfer_data kernel and
*     exchange_direction driver to handle x, y, and z axes in
*     sequence with non-blocking MPI communication.
* Input Variables:
*     gfs: struct ngfs_3d*, grid function container; communication
*         buffers and domain neighbour ranks must already be set up
*     var: int, index of the variable in gfs->vars[] to synchronise
* Output Variables:
*     gfs->vars[var]->val: double*, ghost-zone regions updated with
*         data from MPI neighbours
* Return Values and indicators of success / failure
*     void. Side effects: three rounds of non-blocking MPI
*     communication (one per axis).
*******************************************************************/
void sync_var_3d(struct ngfs_3d *gfs, int var)
{
    const int gs = gfs->gs;
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t nz = gfs->nz; // nz is 1 for 2D logic
    double *val = gfs->vars[var]->val;
    MPI_Comm cart_comm = gfs->domain.cart_comm;

    /* --- X Direction --- */
    /* Send Left: Read from [gs, 2*gs) -> Send to Lower */
    /* Recv Left: Recv from Lower -> Write to [0, gs)   */
    IndexBox x_send_lo = {
        .is = gs, .ie = 2 * gs, .js = 0, .je = ny, .ks = 0, .ke = nz
    };
    IndexBox x_recv_lo = {
        .is = 0, .ie = gs, .js = 0, .je = ny, .ks = 0, .ke = nz
    };
    /* Send Right: Read from [nx-2*gs, nx-gs) -> Send to Upper */
    /* Recv Right: Recv from Upper -> Write to [nx-gs, nx)     */
    IndexBox x_send_up = {
        .is = nx - 2 * gs, .ie = nx - gs, .js = 0, .je = ny, .ks = 0, .ke = nz
    };
    IndexBox x_recv_up = {
        .is = nx - gs, .ie = nx, .js = 0, .je = ny, .ks = 0, .ke = nz
    };

    exchange_direction(
        val, nx, ny, gfs->domain.lower_x_rank, gfs->domain.upper_x_rank,
        gfs->lower_x_src, gfs->lower_x_dst, gfs->upper_x_src, gfs->upper_x_dst,
        gfs->buff_x_size, x_send_lo, x_recv_lo, x_send_up, x_recv_up, cart_comm);

    /* --- Y Direction --- */
    IndexBox y_send_lo = {
        .is = 0, .ie = nx, .js = gs, .je = 2 * gs, .ks = 0, .ke = nz
    };
    IndexBox y_recv_lo = {
        .is = 0, .ie = nx, .js = 0, .je = gs, .ks = 0, .ke = nz
    };
    IndexBox y_send_up = {
        .is = 0, .ie = nx, .js = ny - 2 * gs, .je = ny - gs, .ks = 0, .ke = nz
    };
    IndexBox y_recv_up = {
        .is = 0, .ie = nx, .js = ny - gs, .je = ny, .ks = 0, .ke = nz
    };

    exchange_direction(
        val, nx, ny, gfs->domain.lower_y_rank, gfs->domain.upper_y_rank,
        gfs->lower_y_src, gfs->lower_y_dst, gfs->upper_y_src, gfs->upper_y_dst,
        gfs->buff_y_size, y_send_lo, y_recv_lo, y_send_up, y_recv_up, cart_comm);

    /* --- Z Direction (Completed!) --- */
    IndexBox z_send_lo = {
        .is = 0, .ie = nx, .js = 0, .je = ny, .ks = gs, .ke = 2 * gs
    };
    IndexBox z_recv_lo = {
        .is = 0, .ie = nx, .js = 0, .je = ny, .ks = 0, .ke = gs
    };
    IndexBox z_send_up = {
        .is = 0, .ie = nx, .js = 0, .je = ny, .ks = nz - 2 * gs, .ke = nz - gs
    };
    IndexBox z_recv_up = {
        .is = 0, .ie = nx, .js = 0, .je = ny, .ks = nz - gs, .ke = nz
    };

    exchange_direction(
        val, nx, ny, gfs->domain.lower_z_rank, gfs->domain.upper_z_rank,
        gfs->lower_z_src, gfs->lower_z_dst, gfs->upper_z_src, gfs->upper_z_dst,
        gfs->buff_z_size, z_send_lo, z_recv_lo, z_send_up, z_recv_up, cart_comm);
}
