# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build all object files
make all

# Build the plain Gauss-Seidel driver
make driver

# Build the multigrid V-cycle driver
make driver_multigrid

# Build all test binaries
make tests

# Run all tests (must be run from tests/ directory)
cd tests && bash run_test.sh
cd tests && bash run_test_child.sh
cd tests && bash run_test_project.sh
cd tests && bash run_test_prolong.sh
cd tests && bash run_test_restrict_nl.sh
cd tests && bash run_test_prolong_nl.sh

# Clean
make clean
```

Compiler: `mpicc` with `-Wall -O3 -ffast-math -g`. Tests use Python (`verify.py`, `verify_nl_prolong.py`, etc.) to check correctness by reconstructing the global solution from per-rank JSON output.

## Running the Solvers

```bash
# Plain Gauss-Seidel solver
mpirun -np <N> ./driver NX NY NZ [omega] [n_smooth] [n_iters]

# Multigrid V-cycle solver
mpirun -np <N> ./driver_multigrid NX NY NZ [omega] [n_smooth] [n_iters] [subcycles] [min_cells] [tol]
```

Both solve the 3D Poisson equation with manufactured solution `u = sin(πx)sin(πy)sin(πz)` on `[0,1]^3` with homogeneous Dirichlet BCs.

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

Each test binary exercises a specific operator (domain decomposition, hierarchy creation, projection, prolongation, restriction) in both 2D and 3D. Test scripts invoke `mpirun --map-by :OVERSUBSCRIBE` so tests run on any machine regardless of core count. Python verification scripts reconstruct the global field from per-rank JSON files and check correctness.
