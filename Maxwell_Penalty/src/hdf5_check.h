#ifndef HDF5_CHECK_H
#define HDF5_CHECK_H

#include <mpi.h>
#include <stdio.h>

/* Abort the whole job on the first HDF5 error instead of accumulating a
 * counter and ploughing on with an invalid handle.  The old behaviour
 * let a failed H5Fcreate/H5Gcreate cascade into a flood of follow-on
 * errors and still report success, silently producing a corrupt
 * checkpoint / output file.
 *
 * The result is captured into a wide signed type: HDF5 mixes herr_t
 * (int) and hid_t (int64) return values, and a valid hid_t handle can
 * exceed INT_MAX — narrowing it to int could wrap negative and trip a
 * false error.  Only a genuine error (< 0) aborts. */
#define HDF5_CHK(fn_call)                                                      \
    do {                                                                       \
        long long _ec = (long long)(fn_call);                                  \
        if (_ec < 0) {                                                          \
            int _rk_ = -1;                                                      \
            MPI_Comm_rank(MPI_COMM_WORLD, &_rk_);                               \
            fprintf(stderr,                                                     \
                    "rank %d: HDF5 error in '%s' (%s:%d) — aborting\n",         \
                    _rk_, #fn_call, __FILE__, __LINE__);                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

#endif /* HDF5_CHECK_H */
