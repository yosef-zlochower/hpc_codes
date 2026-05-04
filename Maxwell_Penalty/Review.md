# Code Review: `Maxwell_Penalty`

Reviewer perspective: teaching code for a course in high-performance
computing.  Priorities are correctness → clarity for a student →
performance.

---

## 1. Current state

- Builds cleanly.
- All 14 infrastructure tests pass (covers `test_sync` across
  1/2/4/8/27 ranks in periodic, mixed, and fully-physical
  configurations, `test_rk4`, and `test_topology`).
- Convergence study hits the expected rates in every boundary
  configuration:

  | config | N=16 | N=24 | N=32 | asymptote |
  |---|---|---|---|---|
  | `fully_periodic` | 1.04e-3 | 2.08e-4 | 6.60e-5 | 3.988 |
  | `z_physical`     | 7.39e-3 | 2.27e-3 | 9.39e-4 | 3.072 |
  | `yz_physical`    | 9.49e-3 | 2.87e-3 | 1.17e-3 | 3.104 |
  | `all_physical`   | 1.09e-2 | 3.26e-3 | 1.33e-3 | 3.122 |

- `make check-generated` runs the SymPy generator in a temp
  directory and diffs against the checked-in `simple_maxwell.h`;
  currently reports clean.

### Recent changes (addressed during this review cycle)

1. **Enum-based slot indexing.**  `DECLARE_EVOLVED_VARS` and
   `DECLARE_AUX_VARS` (`src/maxwell_eqs.h`) no longer hard-code
   numeric indices; they reference `DX_SLOT`, `DY_SLOT`, …,
   `IEPS_SLOT`, … from two enums (`enum evolved_slot`,
   `enum auxilliary_slot`).  Two designated-initializer tables,
   `evolved_field_names[]` and `aux_field_names[]`
   (`src/maxwell_eqs.c`), supply the HDF5 dataset names keyed by the
   same enum, and `driver.c` renames each slot in a loop rather than
   repeating the names.  The hard-coded `n_evol_vars = 9; n_aux_vars
   = 5;` in the driver now reads `= N_EVOL; = N_AUX;`.  Reordering
   a slot is now a single-line change in the enum and everything
   else follows — name-to-storage alignment cannot silently drift.

2. **`->dot` rewiring documented.**  A comment at the top of
   `maxwell_eq_time_deriv` explains that the `dotBx`, `dotDy`, …
   pointers alias `gfs->vars[slot]->dot`, which `RK4_Step`
   rebinds between stages to point at `K1..K4`, and therefore
   `maxwell_eq_time_deriv` is only safe to call from an integrator
   that establishes the same convention.  The data flow was
   previously undocumented.

3. **Redundant `fabs` dropped.**  `l2_error_analytic`'s 9
   per-point `L2_ADD_TO_ERROR(fabs(x - y))` calls are now
   `L2_ADD_TO_ERROR(x - y)`; the macro squares its argument so the
   `fabs` was a no-op.  Each L2-error call now does 9 fewer `fabs`
   invocations per grid point.

4. **`int` / `int64_t` fully unified for grid indices and
   strides.**  A closer look found the remaining inconsistency was
   in `derivatives.h` — every stencil function (`D4CEN`,
   `D6CEN`, `D8CEN`, `SBP42_L0..L3`, `SBP42_RN..RNm3`,
   `DISSIP_any_{1,3,5,7,9}`, and the second-derivative `Dxy` /
   `Dxx` family) took `int i, int di` parameters, so every caller
   (including `apply_stencil` and the `DIFFX/Y/Z` macros in the
   deep-interior loop) was silently narrowing `int64_t` to `int`.
   All stencil signatures now take `int64_t`.  Along the way,
   `src/rk4.c:28` changed `uint64_t n_pts` to `int64_t n_pts` to
   match the signed type of `gfs->n_tot`; `src/numerical.c:25–27,
   53` switched `size_t di/dj/dk` and `ijk` to `int64_t` to match
   the rest of the codebase.  A final sweep for `(int)`-casts on
   grid-size quantities and `int`-locals fed from `int64_t` struct
   fields comes up empty: the only remaining `int` loop iterators
   are over bounded counts (`N_EVOL`, `N_AUX`, `NUM_TIMERS`, MPI
   rank counts, ≤16 dimensions).

5. **Boundary-shell loop is partitioned** (earlier change, still in
   place).  The shell is now swept as three non-overlapping regions
   — z-shell × full(j, i); z-interior × y-shell × full i; z,y-
   interior × x-shell — so each shell point is visited exactly once
   instead of iterating the whole computed range and `continue`-ing
   past the deep interior.  Same pattern applied to
   `maxwell_constraints`.

6. **`APPLY_SAT_*` macros are generated** (earlier change, still in
   place).  `generate_ccode.py` emits the six penalty macros from
   the PDE's principal-part flux matrix via SymPy diagonalisation,
   so the SAT sign conventions cannot drift from the characteristic
   analysis in `doc/documentation.tex §4.4`.

7. **`traveling_wave` mode numbers exposed in TOML.**  The three
   mode integers `t_wave_l, t_wave_m, t_wave_n` are now read from
   `[initial_data]` in the TOML, validated as positive by
   `parameters::get_positive_integer_value` (which also guarantees
   `l, m, n > 0` and eliminates the division-by-zero concern in
   `traveling_wave`), and passed through `maxwell_param_st →
   analytic_params_st`.  The `scripts/convergence_test.py` TOML
   template was updated to supply all three, and `traveling_wave`
   itself now reads `params.t_wave_l` etc. rather than the old `.l`
   / `.m` / `.n`.  A student can now run the convergence test at
   any mode triple without editing C.

   *Typo caught on review:* the driver's assignment `t_wave_n =
   maxwell_params.t_wave_m` (on line 91) would have silently
   overridden the user's `t_wave_n` with the value of `t_wave_m` —
   invisible under the stock TOML which uses `l = m = n = 2`, but
   would have broken any asymmetric-mode convergence study.  Fixed.

8. **Driver hygiene.**  `struct ngfs gfs = { .vars = NULL,
   .auxvars = NULL };` replaced with `struct ngfs gfs = {0};`
   (zero-initialises every field, including the `comm_*` substructs
   that earlier refactors added).  The stale "`Courant stability
   condition: dt < k * dx`" comment, where `k` was overloaded with
   `wave_k`, `beam_k`, etc., has been dropped; the `dt =
   cfl_factor · min(dx, dy, dz)` line now reads on its own.

9. **`DISSIP5_HALF` named constant.**  `src/numerical.c` now
   `#define`s `DISSIP5_HALF = 3` and uses it both in the
   `gs < DISSIP5_HALF` guard and in all three dissipation loop
   bounds, so the "5th-order stencil has half-width 3" invariant is
   a one-line change instead of a four-site sync.  Addresses §2.1.

10. **`apply_dissipation` ordering documented.**  The header comment
    block of `apply_dissipation` (lines 10–17 of `src/numerical.c`)
    now carries the "must be called after the PDE RHS is in ->dot"
    invariant, together with an accurate note that dissipation does
    not alter the first three and last three gridpoints in each
    direction (matching the `DISSIP5_HALF = 3` loop bounds).
    Addresses §2.2.

11. **Source parameters restructured under `[source]`.**  The three
    analytic sources each have their own `[source.<name>]` sub-
    table in the TOML (`[source.plane_wave]`,
    `[source.gaussian_beam]`, `[source.te_waveguide_mode]`) selected
    by `source.type`.  C side mirrors it with a struct-of-structs
    in `maxwell_param_st` (three `plane_wave_params` /
    `gaussian_beam_params` / `te_waveguide_mode_params` sub-structs
    from `analytic_solutions.h`), and `parameter.cc`'s helpers
    navigate dotted section paths via toml++'s `at_path`.  The
    driver now copies each source block in a single struct
    assignment instead of field-by-field.

12. **Material parameters restructured.**  `[material]` now holds
    only the `epsilon_type` selector; `[material.background]`
    carries the always-used uniform scalars (epsilon, mu, sigma),
    and `[material.elliptical]` carries the lens-bump profile with
    `eps_` prefixes stripped.  The elliptical formula was
    generalised to `(max − eps_bg) · exp(−r⁴) + eps_bg` so the
    background of the bump is the configurable `epsilon` from
    `[material.background]` rather than a hardcoded `1.0`; default
    `epsilon = 1.0` reproduces prior behaviour bit-for-bit.

13. **Gaussian-beam turn-on ramp exposed in TOML.**  The ramp
    interval for the beam's smooth `f(t, a, b)` turn-on envelope
    now comes from `[source.gaussian_beam].ramp_a` / `ramp_b`
    (defaults `0.0` / `1.0`) — previously hardcoded to `f(t, 0, 1)`
    in C.  The beam section is now configurationally on par with
    `[source.plane_wave]`, which already had its envelope exposed
    via `bump_a` / `bump_b`.  Defaults reproduce prior behaviour
    bit-for-bit; the new parameters let a student lengthen the
    ramp (e.g. to investigate paraxial-residual excitation at high
    `k·w0`) from TOML without recompiling.  Addresses §2.1.

### Structural strengths

- One source of truth for the PDE (`generate_ccode.py` →
  `simple_maxwell.h`), with a `make check-generated` target that
  fails the build if the two drift.
- Clean layering: stencils in `derivatives.h`; runtime dispatch via
  `stencil_at` / `apply_stencil`; characteristic projections in six
  machine-generated `APPLY_SAT_*` macros.
- Hertz-potential Gaussian beam satisfies `∇·D = ∇·B = 0`
  identically, so `PsiD, PsiB` stay at numerical-noise level.
- Named constants (`D4CEN_HALF`, `SBP42_CLOSURE_ROWS`) tie the loop
  bounds to the algorithmic widths they belong to.
- Non-blocking ghost exchange with pre-allocated per-axis buffers
  (`struct comm_axis` wrapping `{lower, upper}` of
  `struct face_buffers`).
- `int64_t` throughout for indices and strides, matching the pointer-
  arithmetic width and freeing the code of `(int)` casts.
- Startup check guards against a doubly-physical axis with
  `n < 2·SBP42_CLOSURE_ROWS` (would silently mis-dispatch the SBP
  closure).

---

## 2. Remaining observations

Nothing here is a correctness bug.  These are the things a student
would still trip over.

### 2.1 Generic pack/unpack in `transfer_data`

`src/comm.c:11–33` walks the 3D index box one double at a time.
For a z-direction slab (contiguous in memory) a single `memcpy`
would be ~5–10× faster; y-slabs are stride-nx per `j`; x-slabs are
strided.  The current form is maximally general but also maximally
slow.  Acceptable as teaching material; a comment pointing at the
optimisation would orient a student considering "what can I make
faster here?"

### 2.2 Ghost exchange serialised across axes

`sync_vars` completes x → y → z sequentially with a `Waitall`
between each.  Because the Maxwell stencils are axis-aligned (no
mixed `∂_{xy}` terms), corner/edge ghosts are never read — so the
serialisation isn't required for correctness.  Posting all three
axes' non-blocking sends and recvs together before a single
`Waitall` would allow network/compute overlap.  Minor optimisation,
worth noting in a comment even if left as-is.

---

## 3. Status

All of the issues surfaced during the current review cycle that had
concrete fix proposals have been applied.  The code:

- builds cleanly,
- passes all 14 infrastructure tests,
- reproduces the expected convergence rates (~3 on every physical-
  boundary configuration; 4 fully periodic),
- has a build-time check that the generated SAT / interior code
  matches its SymPy source.

The remaining items in §2 are two scope questions that are
genuinely optional for teaching code (§2.1 pack/unpack generality,
§2.2 ghost-exchange axis serialisation).  Neither is blocking and
neither changes user-facing behaviour.
