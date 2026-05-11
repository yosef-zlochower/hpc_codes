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

### Known limitations after CellCentred Phases 2--7

* **`manufactured_neumann_inhomog` excluded from CTest.**  This
  is the singular all-Neumann inhomogeneous preset
  ($u = \sin(\pi x/2) \sin(\pi y/2) \sin(\pi z/2)$, with non-zero
  $\partial_n u$ on the three lower faces).  The V-cycle's defect
  plateaus at $\sim 10^{-2}$ instead of driving to $10^{-9}$:
  the singular system's constant null space is repeatedly
  re-injected by the V-cycle and immediately stripped by the
  per-iteration mean-zero projection, so the defect oscillates
  around a non-zero stationary value.  This caps the empirical
  rate at the finest pair to $\sim 1.7$ at `min_cells = 2` or
  $\sim 1.88$ at `min_cells = 4` -- both below the test's
  $[1.8, 2.3]$ threshold (the latter is borderline-passing but
  not consistently so).

  The discretisation itself is rate-2 (consistent with the other
  Neumann-bearing presets that do pass).  The missing rate is on
  the iterative side -- a Krylov-accelerated coarse solve or an
  explicit null-space-orthogonal V-cycle iteration would close
  the gap.  Deferred.

* **`run_test_convergence.sh` uses `min_cells = 2` (was 4).**
  At np=8, the auto-topology decomposes the global grid into
  $2 \times 2 \times 2$ per-rank tiles, so $N = 32$ leaves each
  rank with only 16 cells/dir.  `min_cells = 4` caps the
  hierarchy at 3 levels (coarsest $4 \times 4 \times 4$
  per rank), at which point the V-cycle plateaus at a defect
  reduction factor of $\sim 0.8$ per cycle.  `min_cells = 2`
  unlocks 5+ levels on the same $N$ and recovers the
  $h$-independent convergence rate that the discretisation
  permits.  All currently-active presets (including the
  re-enabled `manufactured_mixed_inhomog`) pass at both np=1 and
  np=8 with `min_cells = 2`.

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

## Phase 6 outcome (partial)

Phase 6 was meant to land two coupled fixes that together would
re-enable `manufactured_neumann_inhomog` and
`manufactured_mixed_inhomog` in CTest.  In practice only the
prolongation fix landed cleanly; the boundary-stencil fix made
SOR with $\omega = 1.5$ unstable on Neumann faces (the
4-point ghost has a negative off-diagonal $-1/23$ on $u_3$,
breaking the M-matrix property the SOR convergence theorem
relies on) and was reverted.  The deferred tests remain deferred.

### 6.1 Position-aware cc prolongation (landed)

`prolong_var_cc_3d` now branches per-axis on whether the fine
cell is adjacent to a hybrid Dirichlet vertex (using the same
`x_lower_d_v` / `x_upper_d_v` predicates that
`gauss_seidel_3d` uses).  When the predicate fires, the 1D
weights for that axis become $(1/2, 1/2)$ instead of $(3/4, 1/4)$,
correcting the geometric error: at the fine cell adjacent to a
coarse Dirichlet vertex, the fine cell is equidistant from the
coarse vertex (at the box edge) and the coarse cell 1 (at
$h_c/2$ inside the box), so the linear weights should be equal,
not the $(3/4, 1/4)$ that the cc-trilinear formula assumes.

**Effect.**  No change in the converged discrete solution
(verified bit-for-bit on `manufactured_mixed_inhomog`), so the
deferred presets do not become eligible for re-enabling on this
change alone.  Big change in V-cycle convergence *rate*: the
previously-slightly-wrong prolongation forced 30+ V-cycles to
drive the boundary error down; the corrected version converges
in well under 20.  `tol = 1.0e-12` early-exit in
`run_test_convergence.sh` makes the convergence test 10x faster
end-to-end (75 s vs 760 s for the full ctest convergence pass).

### 6.2 Hybrid Dirichlet-vertex 4-point Lagrange (landed)

`gauss_seidel_3d`'s slow path at cells adjacent to a hybrid
Dirichlet vertex now uses the 4-point Lagrange-extrapolation
form
$$u_{xx} \approx \frac{1}{5 h^2}\bigl(16 u_v - 25 u_i + 10 u_{\text{far}} - u_{\text{far,far}}\bigr).$$
This is $\mathcal{O}(h^2)$ truncation locally vs. $\mathcal{O}(h)$
for the 3-point form.  No q-coefficient, so no compatibility
issue.  `calc_defect_3d` matches.

**Effect.**  No change in the converged discrete L_$\infty$
error on `manufactured_mixed_inhomog` (the L_$\infty$ in that
problem turns out to be at the upper-N ghost row, where the
Neumann mirror's $\mathcal{O}(h^3)$ ghost truncation dominates --
so improving the lower-D-vertex stencil doesn't move the needle).
Kept in the code anyway because it improves the boundary-cell
discretisation order in principle and the per-cell cost is
negligible.

### 6.2B 4-point Neumann ghost (attempted, reverted)

Tried the higher-order Neumann ghost
$u_{\text{ghost}} = (21/23) u_1 + (3/23) u_2 - (1/23) u_3 + (24/23) h\,q$
in `apply_bc_3d`.  At $\omega = 1.0$ this restores rate $\sim 1.86$
on `manufactured_neumann_inhomog` (up from $1.65$ at the
finest pair).  At $\omega = 1.5$ (the current convergence-test
setting) the iteration **diverges** at $N = 128$ because the
$-1/23$ off-diagonal coefficient on $u_3$ makes the system
matrix non-M, and the SOR convergence theorem breaks.

The plan's proposed compensating $f$-shift restores
$1^\top b = 0$ but doesn't fix the M-matrix violation, so
$\omega = 1.5$ would still diverge.  The cleanest path to closing
the deferred tests therefore needs **either** a different
boundary stencil whose off-diagonals are non-negative, **or** a
relaxation method that doesn't require an M-matrix (e.g.,
Krylov-accelerated GS, or unpreconditioned Krylov on top of the
V-cycle).

The 4-point Neumann ghost code was reverted on every face;
`apply_bc_3d` is back to $u_{\text{ghost}} = u_{\text{int}} + h q$
on every Neumann boundary.

## Phase 7 (deferred correction): investigated, abandoned

A deferred-correction "Phase B" stage was prototyped to attack
what looked like an $\mathcal{O}(h)$ Neumann-boundary truncation
on `manufactured_neumann_inhomog` and `manufactured_mixed_inhomog`.
The infrastructure (`boundary_truncation_3d` helper, an operator
test, the driver-level Phase A/B split) all worked as designed:
the helper computed $\tau_h$ exactly on a cubic test, the V-cycle
solved $A\,v = \tau_h$ to $10^{-13}$, and the magnitudes of $v$
were consistent with the local-truncation prediction.

But on `manufactured_mixed_inhomog` the correction did *not* move
the reported $L_\infty$ error: $|v|_\infty \approx 8 \times
10^{-4}$ while the metric reported $|u_h - u_{\text{exact}}|_\infty
= 0.118$.  Tracking this back, the 0.118 was at cell
$(i, j, k) = (32, 32, 0)$ on a $34^3$ array -- a corner slot
where (i, j) sit on the upper-x and upper-y interior edges and
$k = 0$ is the hybrid-D vertex on z, whose stored value (set by
`apply_bc` at $z = 0$) disagrees with what the metric evaluated
at the formula coordinate $z = -h/2$.  The "rate-1" reading was
an **`owned_bounds_3d` artefact**, not a discretisation issue.
The 4-point Lagrange formula at hybrid D-vertex cells (Phase 6.2)
and the simple Neumann ghost both genuinely give the $L_\infty$
rate that asymptotic theory predicts (rate 2 for smooth
solutions; the $\mathcal{O}(h)$ ghost truncation is absorbed by
the discrete Green's function).

With `owned_bounds_3d` corrected to skip endpoint slots on any
axis with at least one Neumann face, single-rank runs of both
excluded presets show clean rate-2 convergence on the finest
pair, **without** any deferred-correction infrastructure.
Phase 7's helper, operator test, and driver Phase B were
removed; the simple solver is sufficient on np=1.

The remaining blocker for re-enabling the two presets is the
parallel V-cycle bug documented under "Known limitations"
above, not the discretisation order.

## Reference: original Phase 7 sketch (kept for context)

The original Phase 7 plan, before the metric artefact was
identified, follows.  Useful only as background; the helper and
infrastructure it describes have been removed.

### 7.1 Why a stencil-only fix is impossible

A higher-order ghost extrapolation
$$u_{\text{ghost}} = a\,u_1 + b\,u_2 + c\,u_3 + d\,h\,q$$
is, geometrically, polynomial extrapolation from the interior
cells $u_1, u_2, u_3$ at $x = h/2, 3h/2, 5h/2$ to the ghost
location $x = -h/2$, with a Neumann correction proportional to
$q$.  The Lagrange basis values at $x = -h/2$ for points at
$\{h/2, 3h/2, 5h/2, 7h/2\}$ are $(4, -6, 4, -1)$ -- alternating
signs are inherent.  Solving for the coefficients with the
constraint $d = 1$ (compatibility-preserving) and
$\mathcal{O}(h^4)$ truncation gives 4-point coefficients
$(a, b, c, e) = (25/24, -1/8, 1/8, -1/24)$ -- same alternating
signs, regardless of stencil width.  No
compatibility-preserving high-order ghost stencil has all
non-negative coefficients; the matrix's boundary row will
violate the M-matrix property.  The fix has to live somewhere
other than the ghost coefficient set.

### 7.2 Deferred correction

Solve the low-order system *twice*.  The first solve produces
$u_h$ with rate-1 boundary truncation; the second solve corrects
it.

**Mathematical setup.**  Let $A$ be the simple-ghost discrete
Laplacian (M-matrix preserved), $u_h$ its converged solution,
and $u^*$ the grid-restricted exact solution.  The local
truncation error
$$\tau_h \;=\; A\,u^* - f_h$$
is $\mathcal{O}(h^2)$ at every interior cell and
$\mathcal{O}(h)$ at cells adjacent to a Neumann face, where the
leading boundary contribution is
$$\tau_h(\text{boundary cell}) \;=\; \frac{h}{24}\,\partial_n^3
u^*(\text{boundary}) \;+\; \mathcal{O}(h^2).$$

The discrete error $e_h = u^* - u_h$ satisfies $A\,e_h = \tau_h$,
so $e_h = A^{-1}\tau_h = \mathcal{O}(h)$ at boundary cells.  The
deferred-correction step is: estimate $\tau_h$ from the
low-order solution $u_h$, solve $A v = \hat{\tau}_h$ as a
correction problem with homogeneous BCs, and report
$$u_{\text{corrected}} \;=\; u_h + v.$$
If $\hat{\tau}_h$ matches $\tau_h$ to $\mathcal{O}(h^2)$, then
$u_{\text{corrected}} - u^* = \mathcal{O}(h^2)$ globally,
including in $L_\infty$.  This is the standard Pereyra (1968)
deferred-correction argument.

**Estimating $\partial_n^3 u^*$.**  For a Neumann face on a
cell-centred axis, a one-sided fourth-order finite difference
using the four interior cells nearest the boundary plus the
ghost row gives $\partial_n^3 u^*$ to $\mathcal{O}(h)$, which is
enough to make the correction $(h/24)\,\hat{\tau}$ accurate to
$\mathcal{O}(h^2)$.  For the lower-x face,
$$\partial_x^3 u(\text{boundary}) \;\approx\;
\frac{u_3 - 3 u_2 + 3 u_1 - u_{\text{ghost}}}{h^3}
\;=\; \frac{u_3 - 3 u_2 + 4 u_1 - h\,q}{h^3},$$
substituting $u_{\text{ghost}} = u_1 + h\,q$.  No extra
dependencies, no special-casing the corner.

### 7.3 Implementation sub-steps

**Step 1.**  Add a helper
`boundary_truncation_3d(struct ngfs_3d *gfs, double *tau_buf)`
in `problem_helpers.c` that:
* Zeros `tau_buf` over all cells.
* For each face the rank owns and that is Neumann, computes
  $\partial_n^3 u_h$ at every cell on the face's interior row
  using the 4-point one-sided formula above, scales by $h/24$,
  and writes that into the corresponding entry of `tau_buf`.
* Handles the per-axis spacing $h_a$ (each face has its own
  normal direction and own $h$).
* At a corner cell where two Neumann faces meet, the two
  $\partial_n^3$ contributions are along independent normals
  and add directly.

**Step 2.**  Modify `driver_multigrid.c`'s outer loop into two
phases:

* *Phase A* -- the current low-order V-cycle loop, unchanged.
* *Phase B* -- a single defect-correction solve.  Compute
  $\hat{\tau}_h$ from the converged $u_h$; set the correction
  variable's RHS to $-\hat{\tau}_h$; flip the BC spec to its
  homogeneous variant via `bc_spec_homogenize` (already exists
  for the hierarchy constructor); run the V-cycle on the
  correction system; fold the result into $u_h$ with
  `u_h += v`.  For singular all-Neumann problems, mean-zero-
  project $v$ before folding (otherwise the correction adds an
  arbitrary constant to $u_h$).  No further iteration needed --
  one pass of Phase B is enough to bump rate 1 → rate 2 for any
  smooth solution.

**Step 3.**  Update `CONVERGENCE_PRESETS` in
`src/tests/CMakeLists.txt`: re-enable
`manufactured_neumann_inhomog` and
`manufactured_mixed_inhomog`.  Verify the rate is in
$[1.8, 2.3]$ at the verifier's finest pair.

**Step 4.**  Add an operator-level test
`test_deferred_correction_3d` that builds a known-cubic-RHS
problem (where $\partial_n^3 u^*$ is constant so $\tau_h$ is
exact), runs Phases A+B, and verifies the corrected solution
matches the analytic answer to round-off.  This is a stronger
check than a convergence rate -- it exercises the
$\hat{\tau}_h$ formula in isolation.

### 7.4 What can go wrong, and how to catch it

* **`tau_buf` wrong at corners.**  Cross-check: the sum of
  $\tau$ over all interior cells should equal the sum of
  boundary $\partial_n^3 u_h$ contributions scaled by
  $h_a/24$, summed over each face independently.  Add an
  assertion in development.

* **Correction V-cycle slow.**  Same operator as Phase A, so
  same rate constant.  Should converge in 20–40 V-cycles with
  $\omega = 1.5$.  If we want to bound total cost at $\sim$1.5x
  the current ctest time, drop `n_iters_corr` to 20.

* **Singular Neumann.**  Both phases' $A$ is singular with the
  same null space (constants).  Apply mean-zero projection of
  $v$ at the end of Phase B before folding it in.

* **Truncation estimate noisy at coarse $h$.**  At $N = 32$ the
  4-point one-sided $\partial_n^3$ formula has $\mathcal{O}(h)$
  accuracy, fine; at the coarsest CTest level (smaller-N
  hierarchy levels during the V-cycle) the noise might
  dominate.  Phase B is a one-shot correction so worst case the
  corrected solution is no worse than $u_h$ -- verify
  empirically.

### 7.5 Why this preserves the M-matrix property

Both phases solve $A u = b$ with the *same* $A$ -- the
simple-ghost matrix that's already an M-matrix.  Phase B
differs only in $b$ ($-\hat{\tau}_h$ instead of $f$) and BCs
(homogeneous instead of inhomogeneous).  All of SOR's
convergence guarantees carry through unchanged; $\omega = 1.5$
stays stable.

### 7.6 Verification plan

| Preset | Pre-fix rate | Expected post-fix rate | Test reactivation |
|---|---|---|---|
| `manufactured_dirichlet_homog`/`inhomog` | 2.00 | 2.00 (Phase B contributes zero $\tau$ on pure D-D problems) | unchanged |
| `manufactured_neumann_homog` | 2.00 | 2.00 (no change; $u''' = 0$ by symmetry, $\tau_h \equiv 0$) | already in CTest |
| `manufactured_neumann_inhomog` | $\sim 1.65$ (excluded) | $\ge 1.8$ | **re-enable** in `CONVERGENCE_PRESETS` |
| `manufactured_mixed` | 2.00 | 2.00 (same symmetry reason) | already in CTest |
| `manufactured_mixed_inhomog` | $\sim 1.0$ (excluded) | $\ge 1.8$ | **re-enable** |

### 7.7 Estimated cost and ordering

| Item | Lines of code | Risk |
|---|---|---|
| `boundary_truncation_3d` helper | $\sim$80 | low |
| Driver Phase A/B split | $\sim$30 | low |
| Correction-solve BC swap | $\sim$20 | low (re-uses `bc_spec_homogenize`) |
| New CTest entries | $\sim$5 | low |
| Operator test for cubic RHS | $\sim$120 | low |

Total: well under a day of focused work.

Order: write the helper and the operator test first (so we have
a known-good $\tau_h$ to validate against).  Then wire up Phase
B in the driver.  Then run the singular-Neumann presets and
tune `n_iters_corr`.  Finally re-enable the deferred CTest
entries.

### 7.8 Fallback if deferred correction underperforms

If empirically the corrected rate sits at $\sim 1.5$ instead of
$\sim 2$ -- most likely cause is the $\partial_n^3$ estimate
being noisy at coarse $h$ -- escalate to **Krylov-wrapped
V-cycle**.  Wrap the V-cycle in a Krylov outer iteration (CG for
the singular-but-symmetric problem after deflating constants,
or BiCGStab for the general case) and use the
M-matrix-violating 4-point ghost as the discrete operator.  CG
doesn't require the M-matrix property.  Implementation cost is
roughly 2x the deferred-correction route -- a real Krylov outer
loop, plus careful handling of the constant null space when the
system is singular.

## Other deferred work

* **Replace per-rank JSON with a single HDF5 or MPI-IO file**
  (mentioned as the "Fix (larger, optional)" under the original
  §5.1).  Independent of the boundary-stencil work.
