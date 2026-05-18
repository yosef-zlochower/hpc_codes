#ifndef COMM_H
#define COMM_H
#include "gf.h"
#include "mpi_check.h"

enum var_type
{
    EVOLVED,
    AUX
};

/* Synchronise ghost zones of all variables of the given type (EVOLVED or AUX).
 * Packs the ->dot arrays of all variables into per-face buffers, exchanges
 * them with MPI neighbours using non-blocking communication on the Cartesian
 * communicator, and unpacks the received data into the ghost zone regions.
 * Communication proceeds axis by axis: x, then y, then z. */
int sync_vars(struct ngfs *gfs, enum var_type type);

/* MPI_ERROR (abort-on-error wrapper) now lives in mpi_check.h, included
 * above, so domain.c and the Kokkos comm layer can share it. */

/* Direction of data transfer between grid arrays and communication buffers */
typedef enum
{
    MODE_PACK,   /* grid → buffer */
    MODE_UNPACK  /* buffer → grid */
} TransferMode;

/* 3D iteration space [is,ie) x [js,je) x [ks,ke) describing which grid
 * points to pack into or unpack from a communication buffer */
typedef struct
{
    int64_t is, ie;
    int64_t js, je;
    int64_t ks, ke;
} IndexBox;

#endif
