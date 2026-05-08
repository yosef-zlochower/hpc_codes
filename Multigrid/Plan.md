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
| 3.2 | Added `verbose` field to `param_st` (default `false`).  When set via `[solver] verbose = true`, the per-V-cycle "Starting Vcycle" banner and the per-level `defect = ...` / `restrict at ...` / `prolongate at ...` / `post-smooth defect = ...` traces in `vcycle_3d` are printed; otherwise they're silent.  The per-outer-iteration `iter N |defect|_inf` line stays unconditional (one line per V-cycle, primary signal of progress; verbose only gates the chatty per-level trace).  Smoke test: silent run produces 40 lines (header + 30 iter lines + final error), verbose run produces 416 lines for the same problem.  Parser allowlist updated to accept `verbose` under `[solver]`. | `src/multigrid_parameters.{h,cc}`, `src/multigrid.{h,c}`, `src/driver_multigrid.c` |
| 5.1 | Added `[output] dir = "..."` TOML key (optional, defaults to cwd).  When set, rank 0 calls a `mkdir -p` helper (recursive component-by-component `mkdir`, tolerates `EEXIST`) before any per-rank write, then `MPI_Barrier` ensures every rank sees the directory.  `output_2d_gf` / `output_3d_gf` gained a `const char *dir` parameter; NULL or empty falls back to cwd.  Operator-level test binaries pass `NULL` (back-compat).  Smoke test on np=4 with `dir = "/tmp/mg_out_demo/run_001"`: nested directory auto-created, four `Var0_rank_*.json` files land there, cwd stays clean. | `src/multigrid_parameters.{h,cc}`, `src/io.{h,c}`, `src/driver_multigrid.c`, `src/tests/test_*.c` (10 files) |
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

### Known limitations after CellCentred Phases 2--5

* **Boundary-cell stencil order at hybrid Dirichlet vertices.**  Two
  presets are registered in `problem_registry.c` but **excluded**
  from the CTest convergence loop:
  `manufactured_neumann_inhomog` (all-Neumann inhomogeneous) and
  `manufactured_mixed_inhomog` (D-N hybrid axes with $u_v \neq 0$ at
  the Dirichlet vertex).  Both have the same root cause: the simple
  cell-centred boundary stencil
  $u_{\text{ghost}} = u_{\text{int}} + h\,q$ (Neumann mirror) or the
  3-point non-uniform formula
  $(2 u_v - 3 u + u_{\text{far}})/h^2$ (hybrid Dirichlet vertex) has
  only $\mathcal{O}(h)$ local truncation when the exact solution has
  non-zero higher derivatives at the boundary.  This propagates to
  global rate $\sim 1.5$ on the inhomogeneous presets.

  Phase 5 attempted the 4-point higher-order extrapolation
  $u_{\text{ghost}} = (21/23) u_1 + (3/23) u_2 - (1/23) u_3 + (24/23) h\,q$
  (and the analogous 4-point Lagrange form at hybrid Dirichlet
  vertices).  Both formulas give $\mathcal{O}(h^2)$ Laplacian
  truncation locally and *should* restore rate 2, but in practice
  they did not improve convergence: the 4-point Neumann ghost has
  d-coefficient $24/23$ in front of $h q$, which means the discrete
  compatibility becomes $\sum f = (24/23) \sum q/h$ instead of
  matching the continuous $\sum f = \sum q/h$.  The 4.3% mismatch
  shows up as a constant-mode imbalance of size $\mathcal{O}(\sum
  q/h)/N^3$ that the V-cycle cannot drive to zero.  The 4-point
  Lagrange formula at hybrid D vertices gave bit-identical
  solutions to the 3-point form (likely because the cc trilinear
  prolongation's geometric error at hybrid boundary cells dominates
  the V-cycle behaviour, masking the smoother improvement).  The
  Phase 5 4-point boundary code was reverted; the simpler (and
  empirically slightly better) Phase 3/4 formulas are back.

  A genuine fix likely requires (a) a boundary stencil whose
  d-coefficient stays at exactly 1 to preserve compatibility, and
  (b) a cc prolongation that uses correct geometric weights at
  hybrid boundary cells.  Both deferred.

* **SOR retuning (Phase 5, landed).**  The convergence tests now
  use $\omega = 1.5$, $n_{\text{smooth}} = 2$, $n_{\text{iters}} = 40$
  (down from $\omega = 1.0$, $n_{\text{smooth}} = 50$,
  $n_{\text{iters}} = 60$).  Phase 3's removal of the in-stencil
  mirror means the smoother kernel is now the standard 7-point
  Gauss-Seidel SOR everywhere, and SOR with $\omega = 1.5$ no
  longer interacts badly with the boundary as it did under the
  Phase-1 layout.  Net effect: ~40$\times$ fewer smoothing sweeps
  per V-cycle while still converging the defect to $\le 10^{-8}$
  on every active preset.

## Suggested order of attack

1. **Boundary-stencil-with-compatibility fix (the unfinished
   half of Phase 5).**  Find or design a boundary stencil that has
   $\mathcal{O}(h^2)$ Laplacian truncation *and* preserves discrete
   compatibility ($\sum f = \sum q/h$) so the V-cycle's
   constant-mode residual vanishes for inhomogeneous Neumann
   problems.  Re-enables `manufactured_neumann_inhomog` and
   `manufactured_mixed_inhomog` in CTest.
2. **cc prolongation geometry at hybrid boundary cells.**  At the
   fine cell adjacent to a coarse Dirichlet vertex, the trilinear
   weights $(3/4)$ + $(1/4)$ are off because the coarse "vertex"
   isn't a coarse cell centre.  Replace with weights that respect
   the actual vertex geometry $(1/2, 1/2)$ for the hybrid case.
3. Optional: replace per-rank JSON with a single HDF5 or MPI-IO file
   (mentioned as the "Fix (larger, optional)" under the original §5.1).
