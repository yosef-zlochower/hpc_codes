# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build the driver (default target)
make

# Equivalent: build the multigrid driver explicitly
make driver_multigrid

# Build and run the full test suite (operator unit tests + end-to-end
# convergence test).  Depends on driver_multigrid being built.
make tests

# Run tests individually (must be run from tests/ directory)
cd tests && bash run_test.sh                # MPI domain decomposition
cd tests && bash run_test_child.sh          # multigrid hierarchy
cd tests && bash run_test_project.sh        # injection + restriction
cd tests && bash run_test_prolong.sh        # linear prolongation
cd tests && bash run_test_restrict_nl.sh    # full-weighting restriction
cd tests && bash run_test_prolong_nl.sh     # trilinear prolongation
cd tests && bash run_test_convergence.sh    # end-to-end second-order rate

# Clean (also removes tests/convergence_run/)
make clean

# Clean + remove all per-rank JSON output
make distclean
```

Compilers: `mpicc` (C, `-Wall -O3 -ffast-math -g`) and `mpicxx` (C++17 for the
TOML parameter reader).  The final link uses `mpicxx` so the C++ standard
library runtime is included alongside the C objects.  Tests use Python
(`verify.py`, `verify_nl_prolong.py`, `verify_convergence.py`, etc.) to
check correctness by reconstructing the global solution from per-rank
JSON output, or by parsing the driver's printed error.

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

6. **`io.{c,h}`** — JSON output. Each rank writes `<vname>_rank_<rank>.json` with grid metadata and the local data array.

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
so tests run on any machine regardless of core count. Python verification
scripts reconstruct the global field from per-rank JSON files and check
correctness.

In addition to the operator-level tests, `run_test_convergence.sh` runs
the full driver against an auto-generated TOML at three grid resolutions
(32³, 64³, 128³) on both `np=1` and `np=8`, parses the printed
`|u - u_exact|_inf`, and asserts that the empirical convergence rate on
the finest pair lies in `[1.8, 2.3]`. The expected rate for the
seven-point Laplacian on this manufactured solution is exactly 2.
