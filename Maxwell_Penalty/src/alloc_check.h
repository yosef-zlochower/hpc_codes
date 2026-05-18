#ifndef ALLOC_CHECK_H
#define ALLOC_CHECK_H

#include <mpi.h>
#include <stdio.h>

/* Abort the whole MPI job with a rank-tagged diagnostic when an
 * allocation returns NULL.  Use this instead of bare assert() — assert()
 * compiles to a no-op under -DNDEBUG (the default in Release builds), so
 * a failed malloc/calloc would silently fall through and segfault on the
 * next deref, with no indication of which rank ran out of memory. */
#define CHECK_ALLOC(ptr, what)                                                 \
    do {                                                                       \
        if ((ptr) == NULL) {                                                   \
            int _rk_ = -1;                                                     \
            MPI_Comm_rank(MPI_COMM_WORLD, &_rk_);                              \
            fprintf(stderr, "rank %d: allocation failed in %s:%d (%s)\n",      \
                    _rk_, __FILE__, __LINE__, (what));                         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

#endif /* ALLOC_CHECK_H */
