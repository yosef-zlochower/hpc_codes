#ifndef COMM_H
#define COMM_H
#include "gf.h"

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

/* Abort on MPI error */
#define MPI_ERROR(command)                                                     \
    {                                                                          \
        int mpires = command;                                                  \
        if (mpires != MPI_SUCCESS)                                             \
        {                                                                      \
            fprintf(stderr, "MPI ERROR DETECTED in FILE %s, LINE %d\n",        \
                    __FILE__, __LINE__);                                       \
            MPI_Abort(MPI_COMM_WORLD, -1);                                     \
        }                                                                      \
    }

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
