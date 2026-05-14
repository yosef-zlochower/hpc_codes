# Code Review: Multigrid Poisson Solver

**Scope.**  A from-scratch review of the codebase on `main` at commit
`a09677a` (the merge that brought the cell-centred work in).  Source
inspected: every `.c`, `.h`, `.cc`, `.hpp`, `.cmake`, `.sh`, and `.py`
file under `src/`, plus the top-level documentation
(`Documentation.md`, `doc/documentation.tex`, `doc/tutorial.tex`,
`Plan.md`, `CELL_CENTERED.md`).  Build system and test infrastructure
exercised: a clean `cmake -B build-test -DBUILD_TESTING=ON && ctest`
pass (15/15) ran on the same commit immediately before writing this
review.

The review is organised in three parts: a one-paragraph **overall
assessment**, an inventory of **strengths**, and a severity-ordered
list of **issues with recommended fixes**.  A short list of
**recommendations** at the end picks out the few items that should
land soon.

---

## Overall assessment

This is a well-engineered teaching/research code that punches well
above its size.  In ~6.5 kLOC of production C (excluding the
~17 kLOC of vendored `toml.hpp`) it implements:

* A 3D Poisson solver on a Cartesian box with arbitrary per-face
  Dirichlet / Neumann boundary conditions.
* Three layouts -- vertex-centred (pure D--D), cell-centred (pure
  N--N), and hybrid -- chosen per axis based on the BC pair, with
  per-cell stencil dispatch (4-point Lagrange at hybrid Dirichlet
  vertices).
* MPI parallelisation via a Cartesian topology and ghost-zone
  exchange.
* Geometric multigrid with two pairs of transfer operators
  (vertex-centred full-weighting / trilinear; cell-centred
  box-average / position-aware cc-trilinear) dispatched per level.
* HDF5 output with an XDMF post-processor producing single-file
  ParaView views.
* A TOML parameter parser with allowlist validation and 64-bit /
  range-checked 32-bit integer helpers.

The codebase shows real care: nearly every public function carries a
structured docstring; algorithmic decisions are explained in the
comments (often citing the section of a textbook); a thorough test
suite covers both operator-level invariants and end-to-end
convergence rate; documentation is split into a formal reference
manual (LaTeX, 21 pages) and a follow-along tutorial (LaTeX, 8
pages).  The recent (cell-centred) work made the solver 2--18×
faster than the previous version while *improving* accuracy on the
hardest preset (mixed inhomogeneous BCs); the benchmark and report
are checked in (`bench_results.csv`, `CELL_CENTERED.md`).  No real
correctness bugs were found in the production code paths; the
issues below are tech debt, polish, and one known-but-not-yet-fixed
iterative-side limitation on singular Neumann problems.

---

## Strengths

### Architecture

The code is cleanly layered, and each layer has one job.  Reading
top-to-bottom:

1. **`domain.{c,h}`** -- MPI Cartesian decomposition.
   `automatic_topology` picks per-axis rank counts by greedy prime
   factorisation; `setup_3d_domain` builds the Cartesian
   communicator, computes per-rank local extents via the 1D helper,
   and stores per-face Neumann flags.
2. **`gf.{c,h}`** -- Grid-function containers (`ngfs_3d`) and
   per-variable storage (`gf`), plus pre-allocated face buffers for
   sync, parent/child pointers for the MG hierarchy, and a per-level
   `bc_spec_t *`.
3. **`comm.{c,h}`** -- Ghost-zone sync.  The 3D version uses a
   clean `IndexBox` abstraction and an `exchange_direction` helper
   that handles one axis at a time with non-blocking MPI.
4. **`bc.{c,h}` + `problem.h` + `problem_registry.c`** --
   per-face BC kinds, callbacks, and a registry of named problem
   presets.  `bc_spec_homogenize` produces the homogeneous variant
   that coarse levels carry.
5. **`gauss_seidel.{c,h}`** -- Red-black GS SOR smoother, defect
   operator, and `apply_bc_3d` for face writes.  Each is self-
   contained; the smoother and defect share the same stencil
   dispatch (uniform centred 7-point everywhere except at cells
   adjacent to a hybrid Dirichlet vertex, where a 4-point
   one-sided Lagrange formula is substituted on the affected axis).
6. **`multigrid.{c,h}`** -- Hierarchy construction (`create_child` /
   `create_hierarchy`), two pairs of transfer operators (vertex-
   centred + cell-centred), and `vcycle_3d` itself.  The
   `all_axes_cc(level)` helper picks the right transfer pair at
   each level.
7. **`HDF5BinaryWrite.{c,h}` + `io.{c,h}`** -- Per-rank HDF5 output
   with metadata, file-overwrite refusal, and a thin layer on top
   for the `output_3d_gf` / `output_2d_gf` API.
8. **`parameter.{cc,hpp}` + `multigrid_parameters.{cc,h}`** --
   TOML parser, split into a C++-internal helper namespace and a
   C-callable POD entry point so the C driver can include only the
   POD struct.
9. **`driver_multigrid.c`** -- Top-level program: parse TOML, build
   domain + hierarchy, run V-cycle loop until tol or n_iters,
   compute exact-solution error if available, write HDF5.

Module dependencies form a clean DAG: each layer depends only on
layers below it.  The public ABI of each layer is small and well
documented.

### Algorithm correctness

* **Red-black ordering uses global indices** (`gauss_seidel.c:551`),
  so the partition into red / black cells is consistent across MPI
  rank boundaries regardless of decomposition.
* **Per-axis layout dispatch.**  `axis_x_neumann()` and friends
  (`gauss_seidel.c:27`) decide layout per axis from the per-face
  Neumann flags.  Hybrid axes get the 4-point Lagrange stencil at
  cell 1 (or `nx-2`) on the affected axis; the rest of the grid
  uses the standard 7-point centred form.  The smoother and the
  defect operator are consistent: both apply the same stencil
  substitution (`gauss_seidel.c:478` and `gauss_seidel.c:646`).
* **Position-aware cc prolongation.**  `prolong_var_cc_3d`
  (`multigrid.c:748`) flips the per-axis interpolation weights from
  the standard $(3/4, 1/4)$ to $(1/2, 1/2)$ at the fine cell
  adjacent to a hybrid Dirichlet vertex -- the geometrically
  correct choice when the coarse "far" slot is the D vertex at the
  boundary rather than an interior cell.
* **Two-pass post-smoother.**  `vcycle_3d` runs an SOR sweep with
  the user's `omega` followed by a plain GS sweep (`omega = 1.0`);
  the comment (`multigrid.c:1012-1031`) explains why -- SOR
  accelerates the low-frequency error, plain GS guarantees the
  $h$-independent high-frequency smoothing factor that the
  coarse-grid step needs.
* **Mean-zero projection for singular all-Neumann problems**
  (`driver_multigrid.c:220`).  Stops the constant mode from
  drifting under round-off during the V-cycle.
* **Sign of the prolongation.**  Both `prolong_var_3d` and
  `prolong_var_cc_3d` subtract the prolongated correction in
  place (`pval[idx] -= update`), consistent with the
  $d = \Delta_h u - f$ defect convention.

### Build system

`CMakeLists.txt` is short and explains itself.  Two cache options:

* `BUILD_TESTING` (default `OFF`) gates whether the test harness
  is compiled and registered with CTest.  Production builds are
  driver-only.
* `MULTIGRID_FAST_MATH` (auto-default tied to `BUILD_TESTING`)
  adds `-ffast-math` to the compile flags.  Auto-defaults to
  `OFF` when `BUILD_TESTING=ON` (strict IEEE so the convergence
  test measures the real discretisation error) and `ON`
  otherwise.  Either default can be overridden explicitly.  The
  resolved value is printed at configure time so the developer
  can't miss a mismatch.

MPI and HDF5 are found via `find_package`; serial HDF5 is
explicitly enough (per-rank files, no parallel-HDF5 dependency).
The build is strictly out-of-tree; the test scripts get copied
into `${CMAKE_CURRENT_BINARY_DIR}` at configure time so the
historical `./test_X` working-directory convention still works
inside the build tree.

### Test suite

15 CTest entries on the current `main`:

* 8 operator-level tests in C: `test_domain_{2d,3d}` (decomposition
  + sync), `test_child_{2d,3d}` (hierarchy construction),
  `test_project_{2d,3d}` (injection at coincident points),
  `test_prolong_{2d,3d}` (prolongation as a discrete identity),
  `test_restrict_nl_{2d,3d}` + `test_prolong_nl_{2d,3d}`
  (analytic-restriction / -prolongation of a known polynomial),
  `test_restrict_cc_3d` + `test_prolong_cc_3d` (cc operators on a
  known polynomial).  Each is run at multiple grid sizes and
  np counts via the bash scripts.
* 2 end-to-end tests: `test_parser` (driver refuses to start on
  typo'd or out-of-range TOML keys), `test_make_xdmf` (driver
  runs at np=2 → `scripts/make_xdmf.py` → validate the XMF parses
  and the per-rank slabs assemble into the expected global
  vertex count).
* 5 convergence tests (one per BC preset) -- each runs the full
  driver at N=32/64/128 on np=1 and np=8, parses the printed
  $\|u - u_\star\|_\infty$, and asserts rate $\in [1.8, 2.3]$ on
  the finest pair.

The convergence tests are particularly valuable: they catch
regressions that pure operator-level tests cannot (e.g. the
hybrid-axis prolongation bug fixed in the recent work would have
passed every operator test but produced wrong convergence rates,
exactly what the convergence test would have caught).

Verifiers are factored sensibly: `h5read.load_rank` reads a
per-rank HDF5 file and returns a dict whose keys mirror the
legacy JSON field names, so the four field-comparing verifiers
(`verify.py`, `verify_zeros.py`, `verify_nl_restrict.py`,
`verify_nl_prolong.py`) each contain only the math, not the I/O.

### Documentation

* **`doc/documentation.tex`** (21 pages) -- a real reference manual.
  Sections cover the continuous problem; discretisation including a
  TikZ figure showing the four axis layouts (D--D, N--N, D--N,
  N--D); transfer operators with explicit per-layout formulas (the
  full-weighting 27-point weights, the 8-point cc box-average, the
  $(3/4, 1/4)$ cc trilinear, and the $(1/2, 1/2)$ hybrid-D-vertex
  override); a guided tour of the implementation; and parameter /
  tuning guidance.  The PDF rebuilds cleanly from source.
* **`doc/tutorial.tex`** (8 pages) -- a follow-along guide to
  adding a new BC preset.  Two worked examples (all-Dirichlet +
  mixed), with copy-paste-ready C blocks for `problem_registry.c`
  and matching TOMLs.  Calls out the outward-normal sign
  convention for Neumann data as the "single most common gotcha".
* **`Documentation.md`** -- a Markdown summary of the algorithm
  and parameters, suitable for a project README.
* **`CELL_CENTERED.md`** -- the side-by-side benchmark report
  that motivated and justified the recent merge.
* **`Plan.md`** -- running log of completed work and outstanding
  issues, useful as project history.
* **`src/CLAUDE.md`** -- developer-facing build / test / output
  workflow notes.

Public C/C++ functions in the project's own headers carry
docstrings in a consistent "Purpose / Input / Output / Returns"
format (see e.g. `domain.h:88` onward, `gauss_seidel.h:11`,
`multigrid.h:7`).  Long algorithmic comments in `gauss_seidel.c`
and `multigrid.c` explain non-obvious choices (the half-step on
hybrid axes, the position-aware prolongation, the two-pass
post-smoother) and reference textbook sections.

### Safety

* **`HDF5BinaryWrite` refuses to overwrite existing output files**
  (`HDF5BinaryWrite.c:253`).  An accidental re-run with the same
  `[output] dir` aborts with a clear diagnostic ("delete the file
  or change `[output] dir`") rather than silently destroying the
  prior solution.  `H5F_ACC_EXCL` makes HDF5 itself enforce the
  no-clobber rule even under races.
* **TOML allowlist validation** rejects typo'd keys.  A user who
  writes `mulitgrid = true` gets a parser error rather than the
  silent missing-key default.  Sections, keys, and value types are
  all checked.
* **64-bit / range-checked 32-bit integer helpers**
  (`parameter.cc:35-60`).  A TOML value above INT_MAX bound for a
  field declared `int` produces an explicit range error rather
  than silently sign-flipping.

---

## Issues and recommended fixes

Severity-ordered, highest first.  Per-item: **what** the issue is,
**where** it lives, and a **suggested fix**.

### 1. `manufactured_neumann_inhomog` -- V-cycle plateaus (known limitation)

**What.**  The singular all-Neumann inhomogeneous preset is the
one CTest convergence entry that does not pass.  The V-cycle's
defect plateaus around $10^{-2}$ instead of reaching $10^{-9}$;
the empirical rate caps at $\sim 1.7$--$1.88$, just below the
$[1.8, 2.3]$ acceptance band.  The discretisation itself is
rate-2 (matches the other Neumann-bearing presets); the missing
rate is on the *iterative* side -- the V-cycle keeps injecting a
constant-mode component that the per-iteration mean-zero
projection then strips, so the defect oscillates around a
non-zero stationary value rather than converging.

**Where.**  The preset is registered in `problem_registry.c:322`
but excluded from `CONVERGENCE_PRESETS` in
`src/tests/CMakeLists.txt:60` (a clear comment explains why,
`tests/CMakeLists.txt:67-80`).

**Suggested fix.**  Either of:

* **Krylov-accelerated coarse solve.**  Replace the bottom-grid
  smoother iteration by a CG (or BiCGStab) inner solve that
  projects against the null space.  ~50--100 lines of new code,
  most of it boilerplate.
* **Explicit null-space orthogonalisation inside `vcycle_3d`.**
  After each smoother sweep on the singular level, project
  VAR_DEF and VAR_SOL onto the orthogonal complement of the
  constant mode.  Cheaper and more local, but harder to argue
  the rate guarantee.

This is the only outstanding correctness-adjacent gap.  Without
it, the all-Neumann-inhomog problem is the one shape of input
the solver doesn't fully handle.  If you don't need it, leaving
it excluded is fine and the comment in `tests/CMakeLists.txt`
already documents the limitation honestly.

### 2. 2D code paths are atrophied (landed: deleted)

**Resolved** by deleting the 2D path entirely.  ~2100 lines
removed across `domain.{c,h}`, `gf.{c,h}`, `comm.{c,h}`,
`multigrid.{c,h}`, `io.{c,h}`, `HDF5BinaryWrite.{c,h}`, six
`test_*_2d.c` files, the 2D blocks in six bash test scripts, and
the 2D branches in four `verify_*.py` scripts.  `struct
domain1d_st`, `setup_1d_domain`, and `automatic_topology` were
kept (used by the 3D path).  All 15 CTest entries still pass.

This removal also subsumes items 14 (`prolong_var_2d`
Dirichlet skip) and 15 (`verify_*.py` mixed 2D/3D code paths)
below.

### 3. `timer.{c,h}` is dead code (resolved)

**Resolved** by wiring it into `vcycle_3d`.  The driver now calls
`vcycle_3d_register_timers()` before the solve loop, the V-cycle
brackets each of its five phases (`pre_smooth`, `defect`,
`restrict`, `post_smooth`, `prolong`) with start/stop, and the
driver dumps the rank-0 breakdown via `print_timers()` after the
solve.  Operator-level tests that never call
`vcycle_3d_register_timers()` keep paying zero instrumentation
overhead -- the wrappers are guarded on the per-timer ID being
non-negative.

### 4. `apply_bc_3d` is six near-clones of one operation (resolved)

**Resolved** by factoring the per-face body into a single
`apply_face_3d` helper parameterised by a `struct face_geom`
(face id, axis 0/1/2, lower-vs-upper).  `apply_bc_3d` is now a
21-line dispatcher that iterates the six face descriptors and
skips faces whose owning rank has an MPI neighbour on that side.
Net change: -163 lines in `src/gauss_seidel.c` (~56% reduction in
the affected region).  All 15 CTest entries (including the
mixed-BC convergence presets that exercise every Dirichlet /
Neumann / hybrid combination on every face) still pass.  Adding
a Robin BC -- the next likely BC extension -- is now a single
switch arm inside `apply_face_3d`.

### 5. `APPLY_SOR_3D` is a 65-line macro (resolved)

**Resolved** by promoting the macro to a `static inline void
sor_update_3d(idx, i, j, k, const struct sor_ctx_3d *c)` function.
The captured context (data pointers, fast/slow-path stencil
weights, hybrid-axis Dirichlet-vertex flags, omega) is bundled
into `struct sor_ctx_3d` and constructed once at the top of
`gauss_seidel_3d`.  Convergence rates on all five BC presets
still land in `[1.8, 2.3]` at np=1 and np=8 and end-to-end suite
runtime is unchanged within noise (~93s), so the inline expands
to the same codegen as the macro at -O3.  The function form
gives real gdb line numbers and type checking.

### 6. Repetitive 6-face Neumann ghost code in `apply_bc_3d` (resolved)

**Resolved** as part of (4) -- the unified `apply_face_3d` helper
now serves both Dirichlet and Neumann branches.

### 7. `multigrid.toml` reference parameter file has stale defaults (resolved)

**Resolved** by updating the sample to `n_smooth = 2, n_iters = 40,
min_cells = 2` (the tuned cell-centred values).  `tol` was left at
`1.0e-9` rather than the convergence test's `1.0e-12` because at
the sample's `nx_cells = 256` the double-precision roundoff floor
is `~1.5e-10` and a tighter tol would never trigger.  Inline
comments explain each choice.

### 8. `min_cells` default mismatch (resolved)

**Resolved** as part of (7): the sample TOML now ships
`min_cells = 2` with an inline comment lifted from
`run_test_convergence.sh` explaining why a larger value can
plateau the V-cycle when per-rank tiles are small.

### 9. `LICENSE` is at `src/LICENSE`, not the repo root (resolved)

**Resolved**, and the deeper gap underneath it (the existing
`src/LICENSE` was *only* a third-party attribution for vendored
`toml.hpp` + the Hoehrmann UTF-8 decoder -- there was no licence
declaration for the project's own code) addressed at the same
time:

* `LICENSE` at the repo root is now the verbatim GPL-3.0 text
  (fetched from `https://www.gnu.org/licenses/gpl-3.0.txt`),
  preceded by the standard "How to Apply These Terms" short
  notice with the project copyright (Copyright (C) 2026 Yosef
  Zlochower) and a pointer to the LICENSES/ directory for the
  third-party notices.
* `LICENSES/tomlplusplus.txt` and `LICENSES/hoehrmann.txt`
  carry the two MIT notices verbatim, each with a short header
  explaining what they apply to.
* `src/LICENSE` removed via `git rm`.

GitHub / Zenodo / `licensee` will now identify the project as
GPL-3.0.

### 10. `gf_indx_3d` takes a non-const pointer (resolved)

**Resolved** by changing the parameter type to
`const struct ngfs_3d *` at `src/gf.h:79` and updating the
docstring to match.  Purely additive: no call site needed to be
modified, since C permits passing a `T*` where a `const T*` is
expected.  Future const-correct refactors (test verifiers,
restriction helpers reading the parent, etc.) can now pass a
`const` pointer through `gf_indx_3d` directly.  The 2D variant
mentioned in the original item is gone with the broader 2D
removal (item 2).

### 11. Two `name_length` macros at `20` and `1024` shadow each other (resolved)

**Resolved** by renaming `timer.c`'s `NAME_LENGTH` to
`TIMER_NAME_LENGTH`.  `gf.c`'s local `name_length = 20` stays
unchanged.  A future grep for either name now finds only one
match and the per-translation-unit purpose is clear from the
identifier alone.

### 12. `HDF5BinaryWrite.c` `MAX_TRACKED = 32` could silently overflow

**What.**  The "seen files" tracker (`HDF5BinaryWrite.c:227`)
caps at 32 distinct filenames.  Past that, `mark_file_seen`
silently returns (`HDF5BinaryWrite.c:241`), which would cause the
*next* write to the 33rd filename to trigger the "refuse to
overwrite" branch (which would abort).  In practice each rank
writes one file, so this is fine.  But the silent return is
worth a `fprintf(stderr, ...)` warning at least, so a future
extension that writes many files doesn't get a confusing
"refusing to overwrite" diagnostic on a file it just created.

**Suggested fix.**  Either dynamic-allocate the tracker (no fixed
cap), or print a warning when the cap is hit so failure is
explicit.

### 13. `TODO: FIX` markers in test_domain_3d.c (resolved)

**Resolved** by addressing each of the three markers
concretely:

* The `argc != 4` and "NX, NY, NZ all > 0" checks were moved
  ahead of `MPI_Init` so a usage error returns `EXIT_FAILURE`
  with a clean stderr line in the user's shell, rather than
  competing with MPI's own diagnostics and being truncated by
  `MPI_Abort` mid-write.
* The `px * py * pz != mpi_size` check had a broken diagnostic
  (`"PX * PY != MPI_SIZE (%d, %d, %d)"` with args `px, py,
  mpi_size` -- both message and printed values dropped `pz`).
  Fixed to `"PX * PY * PZ != MPI_SIZE (%d, %d, %d, %d)"` with
  `px, py, pz, mpi_size`.
* The six `_rank > -1` neighbour-existence checks in
  `corrupt_gf` used a bare `-1` magic constant while the rest
  of the file uses `!= INVALID_RANK` (defined in `domain.h`).
  All six replaced with the named-constant form.

The 2D variant referenced in the original item is gone with the
broader 2D removal (item 2).

### 14. `prolong_var_2d` hard-codes Dirichlet skip (resolved)

**Resolved** by removing the entire 2D code path (item 2).

### 15. `verify_*.py` mixes 2D and 3D code paths via shape probing (resolved)

**Resolved** by removing the 2D code path (item 2).  The four
field-comparing verifiers are now pure 3D, each ~20 lines
shorter and free of the `is_3d`/`ndim == 2` branches.

### 16. Driver only outputs `VAR_SOL`

**What.**  `driver_multigrid.c:256` -- `output_3d_gf(&gfs, 0,
param.output_dir)` writes only VAR_SOL.  The defect and RHS are
in the same `ngfs_3d` but never written.  For
debugging a stall the defect is exactly what you want to inspect.

**Where.**  `src/driver_multigrid.c:256`.

**Suggested fix.**  Trivial: add two more calls.  Maybe gate them
behind a `[output] write_defect = true / write_rhs = true` pair
of optional TOML keys.

### 17. No CI configuration

**What.**  No `.github/workflows/`, no GitLab CI YAML, no Jenkins
config.  The test suite is comprehensive but only runs when a
developer locally `ctest`s.

**Suggested fix.**  Add a GitHub Actions workflow that runs
`cmake -B build-test -DBUILD_TESTING=ON && cmake --build &&
ctest` on every push.  ~30 lines of YAML.  Run on at least one
Linux distro that ships HDF5 (Ubuntu LTS works); cache the apt
install of `mpi-default-dev libhdf5-dev python3-h5py` so the
job stays fast.

### 18. `Boundary_plan.md` and `CellCentred_plan.md` are historical artefacts (resolved)

**Resolved** by moving both files into `doc/history/` (preserving
git history via `git mv`) and adding a short `doc/history/README.md`
that catalogues what each plan covered.  The two source-code
comments and the one `Plan.md` table entry that pointed at the
old top-level paths were updated to the new locations.  The
project root now shows only the active `Plan.md` alongside the
top-level reports (`REVIEW.md`, `CELL_CENTERED.md`,
`Documentation.md`).

---

## Recommendations (prioritised)

If you want to ship a couple of improvements, here's the priority
order:

1. **Update `multigrid.toml` to the tuned cell-centred defaults**
   (items 7 + 8). **Done** -- sample now uses `n_smooth = 2,
   n_iters = 40, min_cells = 2` with `tol = 1.0e-9` to match the
   roundoff floor at the sample's `nx_cells = 256`.

2. **Decide on the 2D code path** (item 2). **Done** -- deleted
   entirely.  ~2100 lines removed; 15/15 CTest still pass; the
   verifiers and bash scripts are pure 3D.

3. **Wire `timer.{c,h}` into the V-cycle, or delete it** (item 3).
   **Done** -- the V-cycle now instruments `pre_smooth`, `defect`,
   `restrict`, `post_smooth`, and `prolong`; the driver prints a
   per-phase rank-0 breakdown after the solve.  Item 11
   (`NAME_LENGTH` shadow) is now disambiguatable on its own merits.

4. **Refactor `apply_bc_3d` into a per-face helper** (item 4).
   **Done** -- six clones collapsed into a single `apply_face_3d`
   helper plus a 21-line dispatcher; -163 lines in
   `src/gauss_seidel.c`.

5. **Promote `APPLY_SOR_3D` from macro to `static inline`** (item
   5).  **Done** -- macro replaced by `static inline
   sor_update_3d` with the captured context bundled into
   `struct sor_ctx_3d`.  No runtime regression.

6. **Add CI** (item 17).  The test suite is good; have it run
   automatically.

7. **Tackle the singular all-Neumann limitation** (item 1).  Only
   pursue if you have a use case that needs it -- the comment in
   `tests/CMakeLists.txt` already documents the exclusion honestly,
   which is acceptable for a teaching/research code.

Items 9 (`LICENSE` location), 10 (`const`-ness), 11 (`NAME_LENGTH`
shadowing), 12 (tracker cap), 13 (TODO markers), 16 (output more
variables), and 18 (move historical plan docs) are minor polish
worth doing if you're already touching the surrounding code.

---

## Conclusion

This is one of the cleaner research codes I've reviewed.  The
algorithmic core is correct, well-tested, and well-documented; the
build / test / output infrastructure is solid; the recent
cell-centred work made the solver substantially faster and more
accurate without breaking anything.  The issues above are
overwhelmingly tech debt rather than bugs -- code that works fine
but could be tidier, smaller, or more uniform.  The one real
correctness limitation (singular all-Neumann V-cycle plateau, item
1) is honestly documented and excluded from the test suite rather
than masked.

The path from "good" to "great" runs through items 1--6 above,
roughly in that order.
