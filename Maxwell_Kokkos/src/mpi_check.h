#ifndef MPI_CHECK_H
#define MPI_CHECK_H

#include <mpi.h>
#include <stdio.h>

/* Abort the whole MPI job with a rank-tagged, located diagnostic when an
 * MPI call returns anything other than MPI_SUCCESS.  An unchecked MPI
 * return code otherwise lets a failed ghost exchange (or topology setup)
 * corrupt the solution silently — there is no recovery from a botched
 * collective, so fail loud and fast.
 *
 * Only meaningful when the communicator's error handler is
 * MPI_ERRORS_RETURN; under the default MPI_ERRORS_ARE_FATAL the call
 * aborts internally before returning.  Wrapping is harmless either way. */
#define MPI_ERROR(command)                                                     \
    do {                                                                       \
        int _mpires_ = (command);                                              \
        if (_mpires_ != MPI_SUCCESS) {                                         \
            int _rk_ = -1;                                                     \
            char _es_[MPI_MAX_ERROR_STRING];                                   \
            int _el_ = 0;                                                      \
            MPI_Comm_rank(MPI_COMM_WORLD, &_rk_);                              \
            MPI_Error_string(_mpires_, _es_, &_el_);                           \
            fprintf(stderr, "rank %d: MPI error at %s:%d — %s\n",              \
                    _rk_, __FILE__, __LINE__, _es_);                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

#endif /* MPI_CHECK_H */
