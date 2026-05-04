cmake_minimum_required(VERSION 3.18)
project(maxwell_system_kokkos LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
# Kokkos 5.x requires C++20 or newer.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Mirror the C-version flags except for warnings; -O3 comes from Release.
add_compile_options(-Wall -ffast-math -g)

# Suppress the deprecated MPI-2 C++ bindings (mpi/cxx/*.h).  We only use
# the C MPI API; the C++ bindings declare MPI::Init etc. with linkage
# attributes that nvcc rejects ("more than one instance of overloaded
# function MPI::Init has C linkage").  Defining these macros before any
# <mpi.h> include is the standard workaround for both OpenMPI and MPICH.
add_compile_definitions(OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Default Kokkos backend: OpenMP (override on the command line:
#   cmake -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_CUDA=ON \
#         -DKokkos_ARCH_AMPERE80=ON ..
# ).
set(Kokkos_ENABLE_OPENMP ON  CACHE BOOL "Enable Kokkos OpenMP backend")

# When set, the comm layer hands device-resident View pointers directly to
# MPI (skipping the host hop).  Off by default since CUDA-aware MPI is not
# universally available; turn on with -DMAXWELL_CUDA_AWARE_MPI=ON.
option(MAXWELL_CUDA_AWARE_MPI "Pass device pointers directly to MPI" OFF)

add_subdirectory(kokkos)

find_package(MPI  REQUIRED COMPONENTS C CXX)
find_package(HDF5 REQUIRED COMPONENTS C HL)

# ── CPU bookkeeping kept verbatim from Maxwell_Penalty (C-language) ──
set(C_BOOKKEEPING_SOURCES
    domain.c
    timer.c
)

# ── New C++ Kokkos sources ──
# Mirrors the C-version split: maxwell_eqs.cpp references the file-scope
# globals maxwell_params / analytic_params (defined in driver.cpp), so it
# is linked only into the main executable, not into the test binaries.
set(CXX_KOKKOS_SOURCES
    gf.cpp
    rk4.cpp
    numerical.cpp
    comm.cpp
    io.cpp
    HDF5BinaryWrite.cpp
)

set(CXX_PARAM_SOURCES
    parameter.cpp
    maxwell_parameters.cpp
)

add_library(maxwell_core OBJECT
    ${C_BOOKKEEPING_SOURCES}
    ${CXX_KOKKOS_SOURCES}
)
target_include_directories(maxwell_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(maxwell_core PUBLIC
    Kokkos::kokkos
    MPI::MPI_C
    MPI::MPI_CXX
)
if(MAXWELL_CUDA_AWARE_MPI)
    target_compile_definitions(maxwell_core PUBLIC MAXWELL_CUDA_AWARE_MPI=1)
endif()

add_library(maxwell_params OBJECT ${CXX_PARAM_SOURCES})
target_include_directories(maxwell_params PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(maxwell_system
    driver.cpp
    maxwell_eqs.cpp
    $<TARGET_OBJECTS:maxwell_core>
    $<TARGET_OBJECTS:maxwell_params>
)
target_include_directories(maxwell_system PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(maxwell_system PRIVATE
    Kokkos::kokkos
    MPI::MPI_C
    MPI::MPI_CXX
    ${HDF5_C_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
    m
)

# Bake CUDA runtime path into RPATH so the binary finds libcudart.so without
# the user having to load a CUDA module.
if(Kokkos_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set_target_properties(maxwell_system PROPERTIES
        BUILD_RPATH   "${CUDAToolkit_LIBRARY_DIR}"
        INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")
endif()

# ── Tests ──
set(TESTS_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tests)

function(add_maxwell_test name)
    set(sources ${ARGN})
    add_executable(${name} ${sources})
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HDF5_INCLUDE_DIRS}
    )
    target_link_libraries(${name} PRIVATE
        Kokkos::kokkos
        MPI::MPI_C
        MPI::MPI_CXX
        ${HDF5_C_LIBRARIES}
        ${HDF5_HL_LIBRARIES}
        m
    )
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR}
    )
endfunction()

add_maxwell_test(test_sync tests/test_sync.cpp $<TARGET_OBJECTS:maxwell_core>)
add_maxwell_test(test_rk4  tests/test_rk4.cpp  $<TARGET_OBJECTS:maxwell_core>)

add_executable(test_topology tests/test_topology.c domain.c timer.c)
target_include_directories(test_topology PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_topology PRIVATE MPI::MPI_C m)
set_target_properties(test_topology PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${TESTS_OUTPUT_DIR}
)

configure_file(tests/run_tests.sh ${TESTS_OUTPUT_DIR}/run_tests.sh COPYONLY)
configure_file(tests/verify.py    ${TESTS_OUTPUT_DIR}/verify.py    COPYONLY)

enable_testing()
add_test(
    NAME maxwell_tests
    COMMAND bash ${TESTS_OUTPUT_DIR}/run_tests.sh
    WORKING_DIRECTORY ${TESTS_OUTPUT_DIR}
)
