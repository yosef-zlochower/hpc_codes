# =============================================================================
# Multigrid solver targets
# =============================================================================
#
# Project-level configuration (options, compile flags, MPI discovery,
# enable_testing()) lives in the top-level CMakeLists.txt.  This file
# only defines targets.
# =============================================================================

# -----------------------------------------------------------------------------
# Solver library — C
# -----------------------------------------------------------------------------
#
# Bundle the shared C objects into a single static library so the driver
# and every test binary can link against one target rather than each
# re-listing the source files.

add_library(multigrid_solver STATIC
    domain.c
    gf.c
    timer.c
    comm.c
    multigrid.c
    gauss_seidel.c
    io.c
)

target_link_libraries(multigrid_solver PUBLIC MPI::MPI_C m)
target_include_directories(multigrid_solver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

# -----------------------------------------------------------------------------
# Parameter parser — C++ (toml++ wrapper)
# -----------------------------------------------------------------------------

add_library(multigrid_parser STATIC
    parameter.cc
    multigrid_parameters.cc
)

target_link_libraries(multigrid_parser PUBLIC MPI::MPI_CXX)
target_include_directories(multigrid_parser PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

add_executable(driver_multigrid driver_multigrid.c)
target_link_libraries(driver_multigrid PRIVATE
    multigrid_solver
    multigrid_parser)

# driver_multigrid.c is C, but it transitively links to a C++ library
# (multigrid_parser via parameter.cc/multigrid_parameters.cc).  The
# final link step needs the C++ runtime; LINKER_LANGUAGE=CXX makes
# CMake pick mpicxx for the link.
set_target_properties(driver_multigrid PROPERTIES LINKER_LANGUAGE CXX)

# -----------------------------------------------------------------------------
# Tests (only when BUILD_TESTING is ON, set at the top level)
# -----------------------------------------------------------------------------

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
