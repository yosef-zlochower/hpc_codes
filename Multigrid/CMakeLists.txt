# =============================================================================
# Multigrid Poisson solver — top-level CMake configuration
# =============================================================================
#
# Out-of-tree builds.  Typical workflow from the project root:
#
#     # production configuration (default: -ffast-math on, no tests)
#     cmake -B build
#     cmake --build build -j
#     mpirun -np <N> build/src/driver_multigrid <params.toml>
#
#     # test configuration (BUILD_TESTING must be explicitly enabled;
#     # this also flips MULTIGRID_FAST_MATH off so the convergence test
#     # measures the true discretisation error)
#     cmake -B build-test -DBUILD_TESTING=ON
#     cmake --build build-test -j
#     ctest --test-dir build-test --output-on-failure
#
# Object files, executables, and test artefacts all land under the
# chosen build directory.  No build state ever lands in the source tree.
#
# Options:
#   BUILD_TESTING        OFF by default — production builds do not
#                        compile the test harness.  Set
#                          -DBUILD_TESTING=ON
#                        when configuring a tree to run the test suite;
#                        this is the build configuration ctest expects.
#   MULTIGRID_FAST_MATH  Per-language `-ffast-math` toggle.  Defaults are
#                        chosen automatically from BUILD_TESTING:
#                          BUILD_TESTING=ON  -> default OFF (strict IEEE so
#                                              tests measure what the code
#                                              actually does, not what the
#                                              compiler optimised it into);
#                          BUILD_TESTING=OFF -> default ON  (production speed).
#                        Either default can be overridden explicitly:
#                          cmake ... -DMULTIGRID_FAST_MATH=ON
#                          cmake ... -DMULTIGRID_FAST_MATH=OFF
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(multigrid LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

# BUILD_TESTING is the standard CTest umbrella option.  We default it
# OFF: the common build is a production driver, not the test harness.
# Set -DBUILD_TESTING=ON when configuring a tree intended for ctest --
# this is the configuration the test suite expects, and it implicitly
# flips MULTIGRID_FAST_MATH off (see below) so the convergence test
# measures the discretisation error rather than -ffast-math
# reassociations.
option(BUILD_TESTING "Build tests and enable CTest" OFF)

# Determine the default for MULTIGRID_FAST_MATH from BUILD_TESTING, but
# only if the user didn't already pass -DMULTIGRID_FAST_MATH=... on the
# command line.  Once cached, the value sticks for subsequent invocations.
if(NOT DEFINED CACHE{MULTIGRID_FAST_MATH})
    if(BUILD_TESTING)
        set(_fast_math_default OFF)
    else()
        set(_fast_math_default ON)
    endif()
    set(MULTIGRID_FAST_MATH ${_fast_math_default} CACHE BOOL
        "Enable -ffast-math (default: OFF when BUILD_TESTING, ON otherwise)")
    unset(_fast_math_default)
endif()

message(STATUS "BUILD_TESTING       = ${BUILD_TESTING}")
message(STATUS "MULTIGRID_FAST_MATH = ${MULTIGRID_FAST_MATH}")

# -----------------------------------------------------------------------------
# Compiler flags (apply to every subdirectory)
# -----------------------------------------------------------------------------

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Mirror the historical Makefile flags.  Optimisation level and -g apply
# unconditionally; -Wall is informational; -ffast-math is gated by the
# option above.
add_compile_options(-Wall -O3 -g)
if(MULTIGRID_FAST_MATH)
    add_compile_options(-ffast-math)
endif()

# -----------------------------------------------------------------------------
# MPI (used by every translation unit that talks to the network)
# -----------------------------------------------------------------------------

# We use both MPI::MPI_C (for solver/driver) and MPI::MPI_CXX (for the
# parser, which is C++).  find_package puts the right include paths and
# wrapper flags on every target that links to the imported targets.
find_package(MPI REQUIRED COMPONENTS C CXX)

# -----------------------------------------------------------------------------
# HDF5 (per-rank output via the serial C interface + High-Level helpers)
# -----------------------------------------------------------------------------

# Each rank writes its own rank_<R>.h5 file (no parallel HDF5 dependency;
# the serial library is enough because there is no shared-file I/O).  The
# post-processing script scripts/make_xdmf.py stitches the per-rank files
# into a single XDMF view that ParaView / VisIt open directly.
find_package(HDF5 REQUIRED COMPONENTS C HL)
message(STATUS "HDF5 version        = ${HDF5_VERSION}")
message(STATUS "HDF5 include dirs   = ${HDF5_INCLUDE_DIRS}")

# -----------------------------------------------------------------------------
# CTest hookup (must be at the top level so add_test() in subdirectories
# is registered)
# -----------------------------------------------------------------------------

if(BUILD_TESTING)
    enable_testing()
endif()

# -----------------------------------------------------------------------------
# Source tree
# -----------------------------------------------------------------------------

add_subdirectory(src)
