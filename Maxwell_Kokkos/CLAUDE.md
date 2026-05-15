# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`Report.md` (at this directory) is the design/build report — read it first for the rationale behind the Kokkos port and how it diverges from the original `../Maxwell_Penalty/` C version.

## Build Commands

Either from the project root (out-of-source build alongside `src/`):

```bash
cmake -S . -B build && cmake --build build -j     # canonical
ctest --test-dir build                            # runs tests/run_tests.sh
```

…or from inside `src/` (the top-level `CMakeLists.txt` is just a thin
wrapper that calls `add_subdirectory(src)`):

```bash
cd src
cmake -S . -B build && cmake --build build -j
ctest --test-dir build
```

Top-level builds put the main executable at `build/src/maxwell_system`
and test binaries at `build/tests/`; `src/`-local builds put them at
`src/build/src/maxwell_system` and `src/build/tests/`.

Toolchain: `mpicc` / `mpicxx`, C11 / C++20 (Kokkos 5.x requirement),
`-Wall -ffast-math -g` plus `-O3` from `Release`, HDF5 (C+HL),
Kokkos (bundled under `src/kokkos/`).

### Kokkos backends

OpenMP is the default. Switch backends at configure time:

```bash
# CUDA (e.g. Ampere SM 80):
cmake -S . -B build \
      -DKokkos_ENABLE_OPENMP=OFF \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE80=ON
```

`-DMAXWELL_CUDA_AWARE_MPI=ON` hands device-resident `View` pointers
directly to MPI (skipping the host hop). Off by default; only turn on
where CUDA-aware MPI is known to be available.

## Running

```bash
mpirun -np <N> ./maxwell_system maxwell.toml
```

The default `maxwell.toml` and `maxwell_waveguide.toml` mirror the
configurations from `../Maxwell_Penalty/`; the binary, TOML schema, and
HDF5 output format are intentionally identical.

## Running the infrastructure tests

`ctest --test-dir build` invokes `tests/run_tests.sh`, which runs all
14 infrastructure tests (topology, rk4, sync across multiple periodic
and process-count configurations). To invoke them directly:

```bash
cd build/tests
mpirun --map-by :OVERSUBSCRIBE -np 4 ./test_sync 32 32 32 0 0 0
mpirun -np 1 ./test_topology
mpirun -np 1 ./test_rk4
```

`test_sync` writes per-rank `Var0_rank_*.json`; `verify.py` (copied
into the test output dir at configure time) reads them and checks
ghost-zone correctness against a known periodic test function.
