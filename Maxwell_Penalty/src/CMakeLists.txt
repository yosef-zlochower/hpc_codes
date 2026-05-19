cmake_minimum_required(VERSION 3.12)
project(maxwell_system LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Mirror the Makefile's baseline flags (-Wall -ffast-math -g).
# -O3 comes from the Release build type.
add_compile_options(-Wall -ffast-math -g)

find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(OpenMP)

# ── Core library: sources shared by the main driver and the test binaries ────
set(CORE_SOURCES
    comm.c
    domain.c
    gf.c
    rk4.c
    io.c
    HDF5BinaryWrite.c
    timer.c
    numerical.c
    analytic_solutions.c
)

add_library(maxwell_core OBJECT ${CORE_SOURCES})
target_include_directories(maxwell_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(maxwell_core PUBLIC MPI::MPI_C)
# Link OpenMP into the core object library so its sources compile with
# -fopenmp (matching the documented toolchain) and so every binary that
# embeds these objects links libgomp consistently — previously OpenMP
# was attached only to maxwell_system, leaving the core sources and the
# test binaries with an inconsistent OpenMP state.
if(OpenMP_C_FOUND)
    target_link_libraries(maxwell_core PUBLIC OpenMP::OpenMP_C)
endif()

# ── TOML parameter parsing (C++) ─────────────────────────────────────────────
add_library(maxwell_params OBJECT parameter.cc maxwell_parameters.cc)
target_include_directories(maxwell_params PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ── Main executable ──────────────────────────────────────────────────────────
add_executable(maxwell_system
    driver.c
    maxwell_eqs.c
    $<TARGET_OBJECTS:maxwell_core>
    $<TARGET_OBJECTS:maxwell_params>
)
target_include_directories(maxwell_system PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(maxwell_system PRIVATE
    MPI::MPI_C
    MPI::MPI_CXX
    ${HDF5_C_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
    m
)
if(OpenMP_C_FOUND)
    target_link_libraries(maxwell_system PRIVATE OpenMP::OpenMP_C)
endif()

# ── Test executables ─────────────────────────────────────────────────────────
set(TESTS_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tests)

function(add_maxwell_test name)
    set(sources ${ARGN})
    add_executable(${name} ${sources})
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HDF5_INCLUDE_DIRS}
    )
    target_link_libraries(${name} PRIVATE
        MPI::MPI_C
        ${HDF5_C_LIBRARIES}
        ${HDF5_HL_LIBRARIES}
        m
    )
    # $<TARGET_OBJECTS:maxwell_core> contributes only the compiled
    # objects, not maxwell_core's usage requirements, so the OpenMP
    # runtime must be linked here explicitly too.
    if(OpenMP_C_FOUND)
        target_link_libraries(${name} PRIVATE OpenMP::OpenMP_C)
    endif()
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR}
    )
endfunction()

add_maxwell_test(test_sync tests/test_sync.c $<TARGET_OBJECTS:maxwell_core>)
add_maxwell_test(test_rk4  tests/test_rk4.c  $<TARGET_OBJECTS:maxwell_core>)

# test_topology only needs domain.c and timer.c (no HDF5).
add_executable(test_topology tests/test_topology.c domain.c timer.c)
target_include_directories(test_topology PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_topology PRIVATE MPI::MPI_C m)
set_target_properties(test_topology PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR}
)

# Stage the test-driver shell script and its Python verifier next to the
# test binaries so tests/run_tests.sh runs unchanged from the build dir.
configure_file(tests/run_tests.sh ${TESTS_OUTPUT_DIR}/run_tests.sh COPYONLY)
configure_file(tests/verify.py    ${TESTS_OUTPUT_DIR}/verify.py    COPYONLY)

# ── CTest integration ────────────────────────────────────────────────────────
enable_testing()
add_test(
    NAME maxwell_tests
    COMMAND bash ${TESTS_OUTPUT_DIR}/run_tests.sh
    WORKING_DIRECTORY ${TESTS_OUTPUT_DIR}
)

# Catch stale generated code: re-run generate_ccode.py and diff against
# the committed simple_maxwell.h.  Skips cleanly if SymPy or clang-format
# are unavailable; fails loudly on a real mismatch.
add_test(
    NAME generated_simple_maxwell_in_sync
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_generated_simple_maxwell.sh
            ${CMAKE_CURRENT_SOURCE_DIR}/generate_ccode.py
            ${CMAKE_CURRENT_SOURCE_DIR}/simple_maxwell.h
)
