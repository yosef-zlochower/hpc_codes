#ifndef COMM_HPP
#define COMM_HPP
/* Ghost-zone exchange.  The pack/unpack kernels run on the device; MPI
 * sends/recvs run on either device pointers (CUDA-aware MPI build) or
 * a host mirror copy (default).  See gf.hpp for the CommAxis layout. */

#include "gf.hpp"

enum var_type
{
    EVOLVED,
    AUX
};

/* 3D iteration space [is,ie) x [js,je) x [ks,ke). */
struct IndexBox
{
    int64_t is, ie;
    int64_t js, je;
    int64_t ks, ke;
};

/* Synchronise ghost zones for all variables of the given type.
 *
 * For EVOLVED:
 *   kidx in 0..3 selects which RK4 stage buffer K1..K4 to sync.
 *   kidx == -1 syncs the `state` buffer instead (used by the startup
 *              sync test in the driver and by maxwell_constraints'
 *              sync of the cD/cB outputs in AUX).
 * For AUX kidx is ignored (aux fields have only `state`).
 *
 * Posts the x, y, z exchanges all at once and waits with a single
 * MPI_Waitall — corner ghosts are never read by the axis-aligned
 * Maxwell stencils, so per-axis serialisation is unnecessary. */
int sync_vars(NGFS *gfs, enum var_type type, int kidx = 0);

#endif /* COMM_HPP */
