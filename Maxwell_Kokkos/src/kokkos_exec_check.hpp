#ifndef KOKKOS_EXEC_CHECK_HPP
#define KOKKOS_EXEC_CHECK_HPP

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <type_traits>

/* Report (and, on a CUDA build, compile-time enforce) the Kokkos
 * default execution space the tests actually run on.
 *
 * Without this, a build configured with -DKokkos_ENABLE_CUDA=ON but
 * whose default space silently fell back to a host backend would pass
 * every test while never exercising the device — the infra tests would
 * be validating the wrong code path.  The static_assert turns that
 * misconfiguration into a build failure; the print gives visibility on
 * any backend.  Call once, just after Kokkos::initialize(). */
static inline void report_kokkos_exec_space(int rank)
{
    if (rank == 0)
        std::fprintf(stderr, "Kokkos default execution space: %s\n",
                     Kokkos::DefaultExecutionSpace::name());

#ifdef KOKKOS_ENABLE_CUDA
    static_assert(
        std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>::value,
        "CUDA-enabled build but the Kokkos default execution space is "
        "not Kokkos::Cuda — kernels would silently run on the host and "
        "the device path would go untested.");
#endif
}

#endif /* KOKKOS_EXEC_CHECK_HPP */
