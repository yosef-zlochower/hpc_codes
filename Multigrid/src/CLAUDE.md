# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

The build is CMake-driven and out-of-tree — every artefact (object
files, libraries, executables, test outputs) lands under the chosen
build directory and never in `src/`.  The top-level `CMakeLists.txt`
lives in the project root and pulls in `src/` via `add_subdirectory`,
so the typical commands are run from there:

```bash
# Default configuration: production build.
#   BUILD_TESTING       = OFF (no tests compiled; ctest will be a no-op)
#   MULTIGRID_FAST_MATH = ON  (auto-defaulted from BUILD_TESTING)
cmake -B build
cmake --build build -j
mpirun -np <N> build/src/driver_multigrid <params.toml>

# Test configuration: BUILD_TESTING must be enabled explicitly.
# Doing so also flips MULTIGRID_FAST_MATH off (auto-default) so the
# convergence test measures the discretisation error rather than
# -ffast-math reassociations.
cmake -B build-test -DBUILD_TESTING=ON
cmake --build build-test -j
ctest --test-dir build-test --output-on-failure
```

**Note:** running `ctest --test-dir build` against the default
production build will report "No tests were found" — the test
binaries and CTest entries are only generated when
`BUILD_TESTING=ON` is set at configure time. Re-configure with
`-DBUILD_TESTING=ON` (or use a separate build directory as shown
above) to run the suite.

The build-tree layout mirrors the source tree, so the driver lands at
`build/src/driver_multigrid` and test binaries at
`build-test/src/tests/`.

### CMake options

| Option | Default | Effect |
|--------|---------|--------|
| `BUILD_TESTING` | `ON` | Build the operator-level test binaries, copy bash/Python verifiers into the build tree, and register CTest entries. |
| `MULTIGRID_FAST_MATH` | auto: `OFF` when `BUILD_TESTING=ON`, `ON` otherwise | Adds `-ffast-math` to the compile flags. The auto-default keeps tests strict and production fast; either default can be overridden explicitly with `-DMULTIGRID_FAST_MATH=ON/OFF` on the cmake command line. |

The auto-policy gives you what you almost always want without having
to remember to set anything.  Explicit overrides are still supported
for advanced workflows (e.g. running the test suite under `-ffast-math`
to spot any numerics-sensitive assertion drift).

### Test labels

CTest tests are tagged so you can run a subset:

```bash
ctest --test-dir build -L operator    # operator-level unit tests (6 suites)
ctest --test-dir build -L end_to_end  # parser + convergence end-to-end tests
```

Compilers: CMake picks up MPI's `mpicc` / `mpicxx` wrappers via
`find_package(MPI REQUIRED COMPONENTS C CXX)`. The driver is C but
links against the C++ parser (`multigrid_parameters.cc`,
`parameter.cc`); the link step uses the C++ frontend so the C++
runtime is pulled in. Tests use Python (`verify.py`,
`verify_nl_prolong.py`, `verify_convergence.py`, etc.) to check
correctness either by reconstructing the global field from per-rank
HDF5 output (via `h5read.load_rank`) or by parsing the driver's
printed error.

### Dependencies

- **MPI** — required (the solver is parallel from the bottom up).
- **HDF5** (serial C + High-Level helpers) — required for per-rank
  output.  `find_package(HDF5 REQUIRED COMPONENTS C HL)` picks up the
  standard system install: `dnf install hdf5-devel` on Fedora/RHEL,
  `apt install libhdf5-dev` on Debian/Ubuntu.  No parallel HDF5 build
  is needed; each rank writes its own file.
- **Python 3** with `h5py` and `numpy` — required to run the test
  suite's verifiers and `scripts/make_xdmf.py`.  `pip install h5py`
  if your distro doesn't ship a package.

### Cleaning up

```bash
cmake --build build --target clean    # remove generated objects/binaries
rm -rf build                          # full reset including test outputs
```

## Running the Solver

The driver takes a single argument: the path to a TOML parameter file.

```bash
mpirun -np <N> ./driver_multigrid <params.toml>
```

A sample `multigrid.toml` is shipped in this directory.  The single
binary supports both the multigrid V-cycle (`multigrid = true`) and a
plain red-black Gauss-Seidel SOR (`multigrid = false`); the choice is
made at runtime.

Required TOML keys:

| Section / key      | Type   | Meaning                                                |
|--------------------|--------|--------------------------------------------------------|
| `[grid] nx_cells`  | int    | cells in x; grid points = `nx_cells + 1`               |
| `[grid] ny_cells`  | int    | cells in y                                             |
| `[grid] nz_cells`  | int    | cells in z                                             |
| `[solver] multigrid` | bool | `true` → V-cycle, `false` → single-grid GS             |
| `[solver] omega`   | float  | SOR relaxation parameter (`0 < omega < 2`)             |
| `[solver] n_smooth`| int    | red-black GS iterations per smoothing call             |
| `[solver] n_iters` | int    | maximum outer iterations                               |
| `[solver] tol`     | float  | convergence threshold on `|defect|_inf`                |
| `[solver] subcycles` | int  | (multigrid only) max coarse-grid visits per level      |
| `[solver] min_cells` | int  | (multigrid only) min interior cells per rank per axis  |

The driver solves the 3D Poisson equation with manufactured solution
`u = sin(πx) sin(πy) sin(πz)` on `[0,1]^3` with homogeneous Dirichlet
BCs, and prints `|u - u_exact|_inf` at exit.  See `../doc/documentation.tex`
for the full algorithmic description and guidance on extending the code
to other source terms and boundary conditions.

## Architecture

### Layer structure (bottom to top)

1. **`domain.{c,h}`** — MPI Cartesian decomposition. `setup_3d_domain` / `setup_2d_domain` create a Cartesian communicator and distribute grid points across ranks. `automatic_topology` uses greedy prime factorisation to assign processes per dimension.

2. **`gf.{c,h}`** — Grid function containers. `struct ngfs_3d` / `struct ngfs_2d` hold local dimensions, origin/spacing, an array of `struct gf *` variable slots, pre-allocated MPI communication buffers, and `parent`/`child` pointers for the multigrid hierarchy. Index macro: `gf_indx_3d(gfs, i, j, k) = i + (j + k*ny)*nx` (i is fastest-varying).

3. **`comm.{c,h}`** — Ghost-zone synchronisation. `sync_var_3d` / `sync_var_2d` pack/unpack face buffers and perform non-blocking MPI sends/receives one axis at a time.

4. **`gauss_seidel.{c,h}`** — Smoother and defect. Implements red-black Gauss-Seidel SOR for the 7-point Laplacian (`gauss_seidel_3d`), defect calculation (`calc_defect_3d`), and Dirichlet BC enforcement (`apply_bc_3d`). Red-black colouring is based on the global index parity `(global_i + global_j + global_k) % 2`.

5. **`multigrid.{c,h}`** — Multigrid hierarchy and V-cycle. `ngfs_3d_create_hierarchy` builds the coarse-grid chain via repeated halving. Transfer operators: `inject_var_3d` (injection), `restrict_var_3d` (full-weighting, 27-point stencil, weight 8/4/2/1 for centre/face/edge/corner), `prolong_var_3d` (trilinear interpolation, subtracts correction). `vcycle_3d` performs pre-smooth → defect → restrict → recurse → post-smooth → prolong.

6. **`io.{c,h}`** + **`HDF5BinaryWrite.{c,h}`** — Per-rank HDF5 output. `output_3d_gf(gfs, var, dir)` writes the variable as a dataset named `/<vname>` into `<dir>/rank_<R>.h5`; multiple calls from the same rank append additional datasets to the same file. The first call from a rank also writes a `/metadata` group with grid dimensions, spacings, origins, ghost-zone count, per-face Neumann flags, and per-face has-neighbour flags. The post-processing helper `scripts/make_xdmf.py` reads all `rank_*.h5` files in a directory and writes a `multigrid.xmf` sidecar that ParaView / VisIt / PyVista open directly as a single assembled grid (no data copying needed). The `xmf` file references the HDF5 datasets via HyperSlabs and includes the per-axis vertex coordinates, with special handling for the cell-centred layout: hybrid Dirichlet vertices are rendered at the physical boundary; pure-Neumann ghost slots are skipped.

7. **`timer.{c,h}`** — Wall-clock timing utility.

8. **`parameter.{cc,h}` + `toml.hpp`** — TOML parameter file parser (C++ with C linkage), wrapping the `toml.hpp` header-only library.

### Variable indices (defined in `gauss_seidel.h`)

| Index | Meaning |
|-------|---------|
| `VAR_SOL` (0) | Solution field `u` |
| `VAR_RHS` (1) | Right-hand side `f` |
| `VAR_DEF` (2) | Defect `r = Lu - f` |

### PythonReference/

Contains standalone Python/Numba implementations (1D, 2D, 3D) used as reference solvers during development. `Report.md` documents a code review of those implementations.

### Tests

Each operator-level test binary exercises a specific operation (domain
decomposition, hierarchy creation, injection, restriction, prolongation)
in both 2D and 3D. Test scripts invoke `mpirun --map-by :OVERSUBSCRIBE`
so tests run on any machine regardless of core count. Most use Python
verifiers (`verify.py`, `verify_zeros.py`, etc.) that reconstruct the
global field from per-rank HDF5 files via the `h5read.load_rank`
helper; the cell-centred operator tests (`test_restrict_cc_3d`,
`test_prolong_cc_3d`) check pass/fail in C directly and don't need a
Python helper.

`run_test_make_xdmf.sh` is a smoke test for `scripts/make_xdmf.py`:
it runs the driver at np=2 (so the z-axis is split across ranks),
invokes the XMF generator, validates that the resulting `.xmf` is
well-formed XML, that every referenced HDF5 dataset exists, and that
the per-rank slabs assemble into a global grid with the expected
vertex count (after de-duplicating the shared boundary points).

In addition to the operator-level tests, `run_test_convergence.sh` runs
the full driver against an auto-generated TOML at three grid resolutions
(32³, 64³, 128³) on both `np=1` and `np=8`, parses the printed
`|u - u_exact|_inf`, and asserts that the empirical convergence rate on
the finest pair lies in `[1.8, 2.3]`. The expected rate for the
seven-point Laplacian on this manufactured solution is exactly 2.

Two presets in `problem_registry.c` are intentionally excluded from
the convergence test list (see `Plan.md` "Known limitations"):
`manufactured_neumann_inhomog` and `manufactured_mixed_inhomog`.  Both
are blocked by the same O(h)-truncation issue at boundary cells when
the exact solution has non-zero higher derivatives there; the fix is
the 4-point higher-order ghost extrapolation, scheduled for a
follow-up phase.
