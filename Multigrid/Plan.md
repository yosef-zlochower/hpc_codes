# Plan: Outstanding issues

## Already landed

| # | Description | Files touched |
|---|-------------|---------------|
| 1.1 | `assert(ptr->nvars)` after `calloc` replaced with `assert(ptr->vars)` (calloc-failure check was misdirected) | `src/gf.c` |
| 1.2 | `fopen` return checked in both `output_2d_gf` and `output_3d_gf`; abort with diagnostic on failure | `src/io.c` |
| 1.3 | `*_deallocate` and `*_free` lifecycle refactor: single source of truth for chain walk; idempotent deallocate; NULL'ed buffer pointers prevent double-free under future refactors | `src/gf.c` |
| 1.4 | Stale `driver_multigrid_parameters.o` reference removed from `make clean`; `$(RM)` replaces `-rm`; `distclean` target wipes JSON output | `src/Makefile` |
| 5.2 | End-to-end convergence test added: runs driver at 32³, 64³, 128³ on np=1 and np=8, parses `\|u - u_exact\|_inf`, asserts rate ∈ [1.8, 2.3].  Currently observed rate: 2.000 on both serial and parallel paths | `src/tests/run_test_convergence.sh`, `src/tests/verify_convergence.py`, `src/Makefile` |
| 2.1 | `src/CLAUDE.md` rewritten to document the TOML-driven driver and the `make`/`make tests`/`make distclean` targets; obsolete `make driver` and positional-args invocation removed; convergence test mentioned. | `src/CLAUDE.md` |
| 2.2 | Path-like source references (e.g. `gauss_seidel.c:196`) prefixed with `src/` in `Documentation.md` and `doc/documentation.tex`; PDF rebuilt cleanly. | `Documentation.md`, `doc/documentation.tex` |
| 3.1 | Two-pass post-smooth (SOR with user `omega`, then plain GS with `omega = 1.0`) is deliberate: SOR accelerates the smooth-component error, plain GS provides the $h$-independent smoothing factor that the multigrid coarse-grid step requires regardless of the user's `omega`. Rationale documented as a code comment in `multigrid.c` and as new prose paragraphs in both `Documentation.md` (V-cycle section) and `doc/documentation.tex` (§6.1, "Why two post-smoothing passes?"). Convergence test still passes at rate 2.000. | `src/multigrid.c`, `Documentation.md`, `doc/documentation.tex` |
| 4.3 | Parser refactored to the Maxwell_Kokkos two-width helper API: 64-bit pass-through (`get_*_integer64_value`) for `int64_t` fields, range-checked narrowing (`get_*_integer32_value`) for `int` fields. Public ABI split out into a new `multigrid_parameters.h` (POD struct + `extern "C"` entry, no toml++); internal helpers now live in `parameter.hpp`. All four `(int)` narrowing casts in `multigrid_parameters.cc` removed; an out-of-range value (e.g. `n_smooth = 9999999999`) now produces a precise parser error and the driver refuses to start instead of silently truncating. | `src/parameter.hpp` (new), `src/parameter.cc` (rewritten), `src/multigrid_parameters.h` (new), `src/multigrid_parameters.cc`, `src/driver_multigrid.c`, `src/Makefile` |
| 4.2 | Two generic allowlist validators added to `parameter.{hpp,cc}` -- `check_known_sections` (top-level table) and `check_known_keys` (one section). Each unknown name increments the parser error counter, so the driver refuses to start instead of silently using a missing-key default.  Wired into `multigrid_parameters.cc` with explicit allowlists for `[grid]` and `[solver]`.  New `tests/run_test_parser.sh` exercises three malformed-key scenarios (typo'd key in `[solver]`, typo'd key in `[grid]`, typo'd section name) plus the §4.3 range-check regression; all four scenarios produce the expected diagnostic and a non-zero exit. | `src/parameter.hpp`, `src/parameter.cc`, `src/multigrid_parameters.cc`, `src/tests/run_test_parser.sh` (new), `src/Makefile` |
| Build | Build system migrated from Make to CMake; strictly out-of-tree (no artefacts in `src/`); CTest replaces the hand-rolled test orchestration. Two-level layout: top-level `CMakeLists.txt` at the project root holds project metadata, options, compile flags, MPI discovery, and `enable_testing()`; `src/CMakeLists.txt` contains the target definitions; `src/tests/CMakeLists.txt` declares the test binaries and their CTest entries. CTest tests are labelled `operator` vs `end_to_end` for selective runs. `BUILD_TESTING` defaults **OFF** (production-first build); `-DBUILD_TESTING=ON` is required for `ctest`. Build-tree layout mirrors the source layout: driver lands at `build/src/driver_multigrid`, test binaries at `build-test/src/tests/`. `Makefile` removed. | `CMakeLists.txt` (new top-level), `src/CMakeLists.txt` (new), `src/tests/CMakeLists.txt` (new), `src/Makefile` (removed), `src/CLAUDE.md`, `Documentation.md` (new §10 "Building and running"), `doc/documentation.tex` (new §7 "Building and running the code") |
| 4.4 | `-ffast-math` is now an explicit CMake option (`MULTIGRID_FAST_MATH`) with an auto-default tied to `BUILD_TESTING`: OFF when `BUILD_TESTING=ON` (strict IEEE so the convergence test measures the discretisation error rather than `-ffast-math` reassociations), ON when `BUILD_TESTING=OFF`. Combined with the `BUILD_TESTING=OFF` default this means the out-of-the-box configuration is a fast production build; the test suite needs `-DBUILD_TESTING=ON` and inherits strict numerics for free. Either default can be overridden explicitly. The configure-time status print (`-- MULTIGRID_FAST_MATH = ON/OFF`) makes the resolved choice unmissable. | `CMakeLists.txt`, `src/CLAUDE.md` |

The earlier TOML-reader refactor described in the original plan
(`struct param_st` in `parameter.h`, `parameters::` namespace,
`multigrid_parameters.cc`, `driver.c` removal, C/C++ Makefile split,
return-type fixes in `parameter.cc`) was already complete before the
items above were tackled.

---

## Remaining issues

Severity-ordered.  Tags: **doc**, **decision**, **robustness**,
**ergonomics**, **extension**.

### 3. Behaviour worth deciding

#### 3.2 Unconditional V-cycle chatter  *(ergonomics)*

`vcycle_3d` always prints `Starting Vcycle` plus a per-level
`defect = ...` trace via `vcycle_debug`.  Only rank 0 prints, but there
is no quiet mode for benchmark or production runs.

**Fix.** Add a `verbose` field to `param_st` (default false), gate the
prints on it.

---

### 4. Robustness

#### 4.1 `comm.c` rank comparison style  *(robustness, cosmetic)*

`comm.c` uses `if (lower_x_rank > -1)` throughout (`sync_var_2d`,
`exchange_direction`, etc.).  `domain.c` already converts `MPI_PROC_NULL`
to the symbolic `INVALID_RANK = -1`, so the comparison happens to work,
but every other file uses `!= INVALID_RANK`.

**Fix.** Replace each `*_rank > -1` with `*_rank != INVALID_RANK` in
`comm.c`.  No behaviour change.

### 5. Output management

#### 5.1 Per-rank JSON spam in `cwd`  *(ergonomics)*

`output_3d_gf` writes `<vname>_rank_<rank>.json` to the current
directory.  At 1024 ranks this means 1024 files in one directory and
no run identifier.

**Fix (small).** Read an output directory from the TOML
(`[output] dir = "out_001"`); rank 0 calls `mkdir -p` and the rest wait
on `MPI_Barrier`.

**Fix (larger, optional).** Replace per-rank JSON with a single HDF5
or MPI-IO file.  Useful student exercise; not a must-do.

---

### 6. Extensibility scaffolding

The convergence test (§5.2) already proves that the solver as shipped
is second-order on the manufactured Poisson problem.  These items make
*using* the solver for new problems easier without touching hot-path
code.

#### 6.1 RHS / exact-solution callbacks  *(extension)*

The RHS and exact-solution loops are hard-coded in
`src/driver_multigrid.c:125-185`.  Replace with function pointers:

```c
typedef double (*scalar_field_fn)(double x, double y, double z);

void initialise_rhs(struct ngfs_3d *gfs, scalar_field_fn f);
double compute_max_error(struct ngfs_3d *gfs, scalar_field_fn u_exact);
```

Default the function pointers to the manufactured-solution choices.
A student wanting to try a different $f$ writes one C function and
re-runs.

#### 6.2 BC dispatch  *(extension)*

`apply_bc_3d` writes 0 on every physical-boundary face.  The
documentation (`doc/documentation.tex` §8) explains how to extend to
inhomogeneous Dirichlet / Neumann / Robin conditions but the code has
no scaffolding.  First step: factor `apply_bc_3d` per face, taking a
`bc_kind` and a callback per face.  More invasive than 6.1; defer
until needed.

---

## Suggested order of attack

1. **6.1** — natural enabler if the code is going to be used for
   student exercises.
2. **3.2**, **5.1** — quality-of-life on big runs.
3. **4.1**, **6.2** — defer until they bite.
