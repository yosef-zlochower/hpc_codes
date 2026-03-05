#ifndef COMM_H
#define COMM_H

#include "gf.h"
#include <stdint.h>

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
void sync_var_2d(struct ngfs_2d *gfs, int var );

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
void sync_var_3d(struct ngfs_3d *gfs, int var );

typedef enum
{
    MODE_PACK,
    MODE_UNPACK
} TransferMode;

/* Defines a 3D iteration space (inclusive start, exclusive end) */
typedef struct
{
    int64_t is, ie;
    int64_t js, je;
    int64_t ks, ke;
} IndexBox;


#endif
