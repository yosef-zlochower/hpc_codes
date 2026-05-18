# Code Review: Maxwell_Penalty and Maxwell_Kokkos

Both trees share an algorithm; most findings apply to both unless noted. Four
parallel investigators surveyed the codebase; the highest-impact claims were
then verified against the source. Below I keep what I confirmed, mark what
I refuted (so you can see why), and add my own observations.

## Refuted / overstated agent claims (worth flagging because they sound scary)

**"sync_vars after the RHS causes a race — boundary stencils read stale
ghosts."** Not a bug. `sync_vars(EVOLVED)` packs `gfs->vars[v]->dot`, which at
end of stage *N* points to K_N. RK4 then computes `new = old + c·dt·K_N` at
**all** `n_pts` including ghost cells. Since `old` was copied from a synced
`new` at RK4 entry and `K_N` is synced at end-of-stage, `new` enters stage
*N+1* with synced ghost cells. The aliasing trick `dot = new` in the driver's
initial sync-test (`driver.c:196–200`) bootstraps the invariant before the
first stage. Confirmed against `comm.c:71/81/136/146`, `rk4.c:52,69,87,106`,
and `maxwell_eqs.h:33–61`. The design is correct — only the *naming*
(`sync_vars(EVOLVED)` while actually moving `->dot`) is non-obvious.

**"output_3d_counter_override is consumed once → counter not preserved."** The
override is a *one-shot reset*, not a sticky value: it sets `counter` once,
then `counter++` continues. The agent's hypothetical "if output is skipped and
the next iteration calls output twice" is not a real scenario. Refuted.

**"CRITICAL: int overflow in `(int)total_buf_size` (comm.cpp:236,238)."** Real
concern but downgrade to *minor*. At 1024³ × 9 vars × gs=4, total = 37M
doubles, well under INT_MAX (2.1B). Worth a `MPI_Count` upgrade for very large
grids, not urgent.

---

## Critical / Major findings (both codebases unless noted)

### M1. Generated `simple_maxwell.h` has no consistency check
`generate_ccode.py` → `simple_maxwell.h` is mechanical, but neither tree has a
CI gate or build rule that re-runs the generator and compares. A hand-edit to
`simple_maxwell.h` (or a SymPy bump in `generate_ccode.py` left
un-regenerated) silently changes the physics with no diff signal. Both
`Maxwell_Penalty/src/CMakeLists.txt` and `Maxwell_Kokkos/src/CMakeLists.txt`
should add a `check-generated` target that runs the script into a tempfile
and `diff`s it.

### M2. `assert()` is the only safeguard on `malloc`/`calloc` failures
`gf.c:11–16,55–56,62–65,73–75,124–147` (and the Kokkos counterpart
`gf.cpp:67–86`). With `-O3 -DNDEBUG` (release builds), the asserts compile
out; a failed allocation immediately dereferences NULL. The MPI context makes
this much worse — there's no way to know which rank ran out. Promote to
explicit `if (!p) { fprintf(stderr, "rank %d: alloc failed\n", rank); MPI_Abort(...); }`.

### M3. Unchecked MPI return codes everywhere
`domain.c:165,175,183–185`, `comm.c:100,105,115,120,125,157`,
`driver.c:32–33`. Same pattern in Kokkos `comm.cpp:128,134,147,153,236–243,265`.
The `MPI_ERROR` macro in `comm.h` is defined but unused. Silent comm failure
corrupts ghost zones with no diagnostic. Wrap call sites in the existing macro.

### M4. Hardcoded BUFF_LEN ↔ checkpoint filename growth
`io.c:248–250,257–260`, `tests/test_sync.c:164`. Filenames like
`checkpoint_it_%d_rank_%d.h5.tmp` use a fixed 128-byte buffer. With iteration
counts in the millions and rank counts in the thousands, `snprintf` will
silently truncate. Adopt `PATH_MAX` and check the `snprintf` return value.

### M5. Checkpoint atomicity is per-rank, not global
`io.c:262 (barrier) → loop H5* writes → 357 (rename) → 358 (barrier)`. The
`tmp → rename` pattern protects each rank's file individually, but a job
killed *between* line 357 and the second barrier leaves a half-renamed set. A
recover script reading "checkpoint_it_42_rank_*.h5" can find rank 0/1
renamed and rank 2/3 still as `.tmp`. Also,
`find_latest_checkpoint_iteration` (`io.c:381–403`) is called from every rank
— no barrier or broadcast to confirm all ranks find the same iteration. Make
rank 0 perform discovery + broadcast, and either guard recovery against
partial sets (treat presence of *any* `.tmp` as "fail") or have rank 0 rename
all ranks' files after the barrier.

### M6. `set_output_counter_*` only exists for 3D output
`io.c:216–221`. `output_gfs_2D_xy` (`io.c:51,104`) and `output_gfs_2D_xy_h5`
(`io.c:109`) keep static counters with no reset hook. They aren't called by
the current `driver.c`, so this is latent — but if 2D output is wired into
the main loop in the future, post-recovery 2D groups will collide with
pre-checkpoint groups. Either delete the unused functions or add
`set_output_counter_2D_*` + a `driver.c` call mirroring line 181.
`Checkpoint.md` should be updated either way.

### M7. `HDF5_CHK` increments an error counter but doesn't short-circuit
`io.c:225–235` (Penalty), `HDF5BinaryWrite.cpp` (Kokkos). On a failed
`H5Fcreate`, the macro records the error and continues; the next `H5Gcreate`
is called on an invalid `file_id`, producing a flood of HDF5 errors. Worse,
the function returns "success" because `chk_errors` is only printed, never
returned. Have the macro `goto cleanup;` or have the function return
non-zero, and check `file_id >= 0` after `H5Fcreate` specifically.

### M8. Kokkos: raw `double*` extracted from `View` and captured in `KOKKOS_LAMBDA`
`maxwell_eqs.cpp:131–278` extracts `gfs->evol[v].state.data()` etc. and uses
C-style stride arithmetic with `di=1, dj=nx, dk=nx*ny` inside the lambda.
This **works** today because the Views are `LayoutLeft` (= row-major over the
i-fastest convention used) and never reallocated, but:

- A future move to `LayoutRight` for HIP or NUMA tuning silently breaks
  stride math with no compile error.
- The lambda has no Kokkos-tracked dependency on the View, so future
  async-DAG features won't see them.
- The "View lifetime ≥ pointer lifetime" invariant is invisible to the
  compiler.

At minimum add a
`static_assert(std::is_same_v<Field3D::array_layout, Kokkos::LayoutLeft>)` near
`DECLARE_EVOLVED_VARS`. Better: switch the kernel body to `state(i,j,k)`
accessors. This is the single biggest *latent* risk in the port.

### M9. Kokkos: CUDA-aware MPI path is untested and lightly synchronized
`comm.cpp:99–119,128–155`. `MAXWELL_CUDA_AWARE_MPI=ON` returns device
pointers directly to `MPI_Isend/MPI_Irecv`. The `Kokkos::fence()` at
`comm.cpp:229` before posting sends is correct; the *missing* one is between
`MPI_Waitall` (`comm.cpp:243`) and the unpack kernel (`255–260`) in the
CUDA-aware branch. Most CUDA-aware MPI stacks do synchronize the device
queue on completion, but the contract is *implementation-defined* — UCX yes,
MVAPICH historically not always. Add `Kokkos::fence()` after `MPI_Waitall`
unconditionally; the cost is negligible relative to the MPI wait.

Also per `Report.md` §8.4 there is no test that exercises this code path —
add at least a compile gate, ideally a CUDA+MPI bit-equality test against
the host-staged path.

### M10. `numerical.c:55–58` dissipation loop bounds
Loop runs `[DISSIP5_HALF=3, n-DISSIP5_HALF)`. The comment says "first three
and last three gridpoints in each direction (on global grid)" — but these
are *local* indices. With `gs=4` (the typical SBP value), the loop applies
dissipation at `i=3` (a ghost zone) and skips `i=gs+0..gs+2` of the actual
interior. The result is wasted compute in ghost zones (overwritten by the
next `sync_vars`) and *no dissipation* on three boundary-adjacent SBP rows.
The latter may be intentional (matches `SBP42_CLOSURE_ROWS=4` ≈ skip closure
region), but the bound should be `[max(DISSIP5_HALF,gs), n - max(DISSIP5_HALF,gs))`
to make it correct on any `gs`. Same pattern in Kokkos `numerical.cpp`.

### M11. CMake doesn't link OpenMP to `maxwell_core`
`Maxwell_Penalty/src/CMakeLists.txt:33–38`. The core object library is built
with `-fopenmp` (via `add_compile_options`) but `OpenMP::OpenMP_C` is only
linked to the main executable. The test binaries link against `maxwell_core`
without OpenMP and will fail to find `GOMP_*` symbols on toolchains that
don't auto-resolve. Either link OpenMP to the core target unconditionally or
strip `-fopenmp` from the core compile options.

### M12. Kokkos test parity
`Maxwell_Kokkos/src/tests/test_rk4.cpp` and `test_sync.cpp` are direct ports.
Verify they actually exercise the **device** path on a CUDA build (i.e.,
that `Kokkos::initialize` selects the GPU and the kernels run there). On many
systems a `Kokkos::OpenMP` execution space silently runs even when CUDA is
enabled if the host fallback is the default exec space. Add an assertion of
`Kokkos::DefaultExecutionSpace` in test_rk4 so a misconfigured build fails
loudly.

---

## Minor findings

### Penalty

- **`comm.c:100,105,115,120`** — `total_buff_size` is `size_t`, passed as
  `int` count. Same as M9; minor at current scales.
- **`derivatives.h:249`** — `SBP42_L3` comment claims 6 coefficients but only
  4 are non-zero; the formula in the comment doesn't match the four
  `f(i-3)`, `f(i-1)`, `f(i+1)`, `f(i+2)` actually accessed. Cosmetic.
- **`maxwell_eqs.c:606–628`** — `set_initial_data` writes `ieps/imu/sigma` at
  all local points but never calls `sync_vars(AUX)`. If the material profile
  is rough across a rank boundary (corner of the elliptical lens), ghosts
  diverge from interiors of the neighbour. Add an explicit
  `sync_vars(gfs, AUX)` after `set_initial_data`.
- **`maxwell_eqs.c:413–419`** — `apply_dissipation` is called inside the
  timer block but outside the macro `END_TIMER`. Confirm timer accounting
  (it's inside `BEGIN_TIMER(timer_dot)` … `END_TIMER(timer_dot)` block, so OK).
- **`timer.c:76`** — `active_timers` increments without bounds check against
  `NUM_TIMERS=20`. Adding a 21st timer silently corrupts adjacent globals.
  Add a hard check.
- **`timer.c:10–16`** — `gettimeofday` instead of
  `clock_gettime(CLOCK_MONOTONIC)`; subject to wallclock skew during NTP
  adjustments. Cosmetic for typical run lengths.
- **`driver.c:198`** — the `dot = new` aliasing trick leaves `dot` pointing
  at `new` until `RK4_Step` repoints it at K1 (`rk4.c:52`). If the iteration
  loop is ever skipped (e.g., `it_start ≥ max_iterations`), the
  `output_gfs_3D_h5` call at line 234 writes from `new` (fine) but any later
  code that touches `dot` would be reading from `new`. Latent, but worth a
  defensive `gfs.vars[v]->dot = gfs.vars[v]->K1;` after the sync test.
- **`driver.c:227`** — `fopen` checked only with `assert`; in release builds
  a failed open passes silently.
- **`tests/run_tests.sh:12`** — cleans `Var0_rank_*.json` even on failure,
  destroying the evidence; move cleanup to success path.
- **`tests/verify.py:141`** — pass/fail uses `global_error` AND `max_error`;
  the latter is per-rank-max-of-local, which is what `MPI_Allreduce(MAX)`
  computes. Logic is correct; variable name is confusing.

### Kokkos

- **`comm.cpp:219`** — `face_size * nvars` cast to `int`. Same as M9.
- **`rk4.cpp:39–46`** — `Kokkos::deep_copy(old_, state)` then `time_deriv`
  reads `state` (not `old_`) and writes K. The C version's matching `memcpy`
  is synchronous; the Kokkos `deep_copy` is async on CUDA but the next
  kernel launches on the same exec space, so dependencies are honored. Add
  a comment noting this.
- **`io.cpp:22–35`** — `view_to_host_buffer` does
  `create_mirror_view → deep_copy → manual triple-loop copy → vector`. The
  triple loop is redundant: `LayoutLeft` views with i-fastest already match
  the row-major flat layout HDF5 wants. Replace the loop with
  `std::vector<double>(h.data(), h.data() + h.size())`. Performance-only.
- **`gf.cpp:67–86`** — `new EvolField[n]()` value-initializes structs
  (default-constructs Views to empty), then immediately overwrites them
  with Views into the shared buffer. Wasted work; doesn't affect correctness.
- **`CMakeLists.txt`** — Kokkos linked first, before MPI/HDF5. On some
  Cray/IBM toolchains this matters; less critical with modern CMake.
- **`maxwell_eqs.cpp` fence placement** — fences appear once per function at
  end. Relying on Kokkos's implicit DAG for ordering between successive
  `parallel_for` is correct on default exec space; if streams/teams diverge
  in the future it's brittle. Add a brief comment.
- **`tests/test_sync.cpp:51–99`** — corruption logic checks `lx != self`,
  which masks topology bugs that produce spurious self-loops. Add an
  assertion that self-equality implies actual periodicity.

---

## Code-organization and process observations

- **The trees have diverged.** `parameter.cpp` (Kokkos) is 206 lines vs
  `parameter.cc` (Penalty) at 135 — features have landed in one and not
  the other. A shared parameter parser in a tiny submodule would eliminate
  this drift. Similar for `domain.c`, which is identical (332 lines) in
  both — keep it that way with a symlink or shared subdir, not by
  hand-copying.
- **`Maxwell_Penalty/src/.claude/settings.local.json`** exists. Should be
  `.gitignore`d (and likely already is at parent level).
- **`Maxwell_Penalty/bui/`** appears in `git status` as an untracked build
  directory — probably a typo'd `build`.
- **Doc/code drift on the dissipation comment** (`numerical.c:15–16`) — it
  says "global grid" but the loop is local. Either fix the comment or fix
  the bounds.
- **`Maxwell_Kokkos/src/kokkos/`** is the bundled Kokkos source — confirm
  it's a git submodule (or vendored intentionally); a stray `.git` in there
  will confuse tooling.

---

## Recommended action order

1. **M1, M2** — fix the silent failure modes first. Both are one-line fixes
   that turn "wrong answer with no warning" into "loud abort."
2. **M3, M7** — wire `MPI_ERROR` / a fixed `HDF5_CHK` everywhere. Mechanical,
   high payoff.
3. **M5, M6** — make checkpoint recovery actually robust before relying on
   it for long runs.
4. **M8** — add the `static_assert` on layout in the Kokkos lambda capture,
   before someone changes layouts and chases a ghost bug for a week.
5. **M9** — add the post-`MPI_Waitall` fence and an actual CUDA-aware MPI
   test.
6. **M10, M11, M12** — clean-up batch.
7. **Minor batch** — fold in opportunistically.

Where the two codebases agree on a finding, fix in both — most of the
high-impact items here are inherited from the C version verbatim.
