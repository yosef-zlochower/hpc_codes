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
| 4.1 | Replaced 12 instances of `if (*_rank > -1)` in `src/comm.c` with the symbolic `if (*_rank != INVALID_RANK)` for consistency with the rest of the codebase (`gauss_seidel.c`, `multigrid.c`). Behaviour-preserving — `domain.c` already converts `MPI_PROC_NULL` to `INVALID_RANK = -1` before storing in the rank fields. | `src/comm.c` |
| 6.1, 6.2 | Boundary-plan implemented in four phases (see `Boundary_plan.md`). New `bc_spec_t` per-face BC machinery in `src/bc.{h,c}`; a `problem_t` registry in `src/problem.h` + `src/problem_registry.c` unifies BC kinds, RHS, exact solution, and singular-flag for each preset; `src/problem_helpers.c` provides RHS init, mean-zero projection, and max-error reduction. `apply_bc_3d` writes Dirichlet face values (homogeneous or callback-supplied); the smoother and defect kernels widen their sweep on Neumann faces and substitute a ghost mirror `u_int + 2 h q` (homogeneous when `q_cb == NULL`); `prolong_var_3d` skips only Dirichlet faces. Hierarchy constructor now homogenises the parent's BC spec onto each child via `bc_spec_homogenize`, so coarse levels see the correction problem (homogeneous BCs of the same kind). Driver applies mean-zero projection on `singular` problems. Six presets registered: `manufactured_dirichlet_homog` (default; original behaviour), `manufactured_dirichlet_inhomog` (cos³), `manufactured_neumann_homog` (cos³, singular), `manufactured_neumann_inhomog` (sin³_half, singular — registered but not in the convergence test suite, see "known limitations"), `manufactured_mixed` (x²cos(πy)cos(πz); homogeneous Dirichlet on lower-x, Neumann on the other 5 faces with non-zero q only on upper-x), `manufactured_mixed_inhomog` (e^(x+y+z); 3 inhomogeneous Dirichlet on lower faces + 3 inhomogeneous Neumann on upper faces — every face has non-zero data, exercising the full BC machinery simultaneously; rate 2.000). All 12 CTest entries pass: 6 operator suites + parser hardening + 5 convergence presets at three resolutions × np=1 and np=8, asserting rate ≈ 2 on the finest pair. | `src/bc.{h,c}` (new), `src/problem.h` (new), `src/problem_registry.c` (new), `src/problem_helpers.c` (new), `src/multigrid_parameters.{h,cc}`, `src/driver_multigrid.c`, `src/gauss_seidel.c`, `src/multigrid.c`, `src/gf.{h,c}`, `src/CMakeLists.txt`, `src/tests/{CMakeLists.txt,run_test_convergence.sh}`, `Boundary_plan.md` |

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

### Known limitations of the boundary-plan implementation

* **Singular all-Neumann inhomogeneous case.**  The
  `manufactured_neumann_inhomog` preset is registered in
  `problem_registry.c` but **excluded** from the CTest convergence
  loop.  My ghost-mirror Neumann discretisation on a node-centred
  grid implies a discrete compatibility condition $\sum f_h = 2
  \sum_{boundary} q/h_n$, whereas the continuous compatibility is
  $\int f = \int q$ (a factor-of-two discrepancy from the boundary
  nodes carrying full-cell weight in the row sum).  The mean-zero
  projection of $f$ at startup makes the discrete linear system
  inconsistent for the inhomogeneous case, and Gauss-Seidel
  stagnates instead of converging.  The fix is a boundary-aware
  RHS shift: subtract a constant $c = (\sum f - 2 \sum q/h_n) / N$
  from $f$ at startup so $\sum (f - 2 q/h_n) = 0$.  Deferred — the
  inhomogeneous Neumann *code path* is still exercised by
  `manufactured_mixed` (which is non-singular thanks to a Dirichlet
  face), so the test suite covers the kernel logic.

* **Sub-optimal SOR on Neumann boundaries.**  With $\omega = 1.5$ the
  V-cycle rate is $\sim 0.001$ for Dirichlet-only problems (textbook
  multigrid behaviour) but degrades to $\sim 0.6$ at $h = 1/128$ for
  Neumann or mixed BC.  With $\omega = 1.0$ the rate stabilises at
  $\sim 0.34$ for Neumann/mixed.  The convergence test therefore uses
  $\omega = 1.0$ and $n_{\text{iters}} = 60$ to guarantee convergence
  within the tolerance.  A proper fix would investigate why the
  smoother/coarse-correction pair is sub-optimal at Neumann
  boundaries — possibly the ghost-mirror stencil interacts badly with
  red-black SOR ordering at the boundary, or the prolongation needs
  adjustment at Neumann face nodes.  Deferred.

### 7. Remaining cleanup items

#### 3.2 Unconditional V-cycle chatter  *(ergonomics)*

(Same as before — adding a `verbose` flag to `param_st`.)

#### 5.1 Per-rank JSON spam in `cwd`  *(ergonomics)*

(Same as before — adding `[output] dir = "..."` config.)

#### Singular-Neumann compatibility fix *(extension)*

(Documented above; would re-enable the
`convergence_manufactured_neumann_inhomog` test.)

---

## Suggested order of attack

1. **3.2**, **5.1** — quality-of-life on big runs.
2. Singular-Neumann compatibility fix — closes the deferred
   `manufactured_neumann_inhomog` test.
3. Investigate sub-optimal SOR rate on Neumann boundaries —
   diagnose whether the smoother kernel, prolongation, or transfer
   operators degrade at Neumann faces.
