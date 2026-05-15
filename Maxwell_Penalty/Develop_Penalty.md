# Plan: Maxwell with SBP-SAT Penalty Boundaries on All Faces

> **Historical design document.** This file was written *before* the SBP-SAT
> implementation in `src/` existed; it describes the work plan that produced
> it. It is retained for design provenance, not as a current-state
> description. Where the as-built code diverges from the plan, the code wins
> — see `CLAUDE.md` for the up-to-date description.
>
> **Notes on the cross-references in this document:**
>
> - **`PenaltyImplementation.md`** (referenced at §3, §4.6, §8) was a
>   companion file in an earlier sibling directory (`Maxwell/`) that derived
>   the z-axis SAT formulas. **Those formulas had wrong signs on the EM-block
>   coupling terms** (e.g. they wrote `w⁺ = Bx + Dy/PP` for the `(Bx, Dy)`
>   z-axis block when the PDE flux signs in `simple_maxwell.h` require
>   `w⁺ = Bx − PP·Dy`). The SAT macros in this code were *re-derived from
>   scratch* against `simple_maxwell.h` and verified on the simple
>   traveling-wave solution `(Dx, By) = (ψ, ψ)`. Do not look for
>   `PenaltyImplementation.md` here — it was deliberately excluded as a
>   hazard. The `SBP42_HINV_{0..3}` constants the plan attributes to it are
>   in `src/derivatives.h`.
>
> - **`Maxwell/`** (referenced at §2, §7) was the predecessor codebase that
>   used strong characteristic injection on z-faces only. It is not part of
>   this repository; the "verbatim copy" infrastructure files described in
>   the plan have since evolved (in particular `comm.c`/`gf.c` were
>   refactored to use a `struct comm_axis` ghost-exchange abstraction, and
>   `numerical.c`/`rk4.c` use `int64_t` indices). Treat these references as
>   pointing to "the pre-SBP-SAT baseline" rather than to a directory.
>
> - **`Checkpoint.md`** (§7, item 7) is in `src/Checkpoint.md` of this
>   directory.

## 1. Scope and Design Decisions

**Boundary configuration for the new code:**
- Lower-z face: SBP-SAT with prescribed incoming plane-wave data (`g ≠ 0` for incoming characteristics).
- Upper-z, ±x, ±y faces: SBP-SAT with `g = 0` (homogeneous — absorbing for outgoing, zero incoming).
- All six faces are physical boundaries by default; the domain infrastructure *also* keeps the option to mark a face periodic per axis (useful for reference runs). The `bbox3D_st` flags already drive this.

**Why full SBP-SAT on all six faces.** The existing code is a hybrid: SBP closure + *strong* characteristic injection at z; centred stencils + periodic wrap in x and y. The new code unifies these: SBP one-sided closures near every physical boundary, plus an additive SAT penalty exactly at each boundary point. Strong injection is replaced by a weak correction added to the full RHS. This gives a provable discrete energy estimate when `τ ≥ 1/2`, and it is where the x, y face treatments naturally live.

**Target physics:** a collimated beam enters through the lower-z face, propagates through an inhomogeneous-permittivity region (the lens — already parameterised in `maxwell.toml` as `epsilon_type = "elliptical"`), and exits through the upper-z face. Transverse spreading due to diffraction and refraction through the lens is *expected and physical*; the x, y walls must merely be far enough from the beam axis that the Gaussian tail is negligible when it reaches them.

**Two configurations, both supported by the same code:**
- **Option A — six-face SAT, paraxial beam (recommended for the lens demo).** All six faces non-periodic. Lower-z injects a Gaussian beam; upper-z and ±x, ±y use `g = 0`. The beam is a paraxial solution, so it satisfies `∇·D = ∇·B = 0` only to `O(1/(k·w₀)²)` — the constraint-damping fields `PsiD`, `PsiB` (with coefficients `κ_D`, `κ_B`) are exactly the mechanism designed to absorb and radiate away that small residual. Choose `Lx, Ly ≳ 5 w₀` so the beam tail at the walls is below roundoff and the `g = 0` SAT on x, y is indistinguishable from exact.
- **Option B — x, y periodic, SAT only on z-faces (recommended for formal convergence tests).** Keeps the existing periodic-plane-wave regime in which an exact non-dispersive Maxwell solution exists, so pointwise L2 convergence of the SBP-SAT scheme can be measured without any paraxial or truncation contamination. The same code handles it by reading `periodic_x/y/z` from the TOML.

---

## 2. Directory and Build Setup

Create `Week11/Maxwell_Penalty/` as a peer of `Maxwell/`, with the same layout:

```
Maxwell_Penalty/
├── CLAUDE.md          (copy; edit the sections that describe boundary treatment)
├── doc/               (copy — will need a new PDF describing the penalty method)
├── scripts/           (copy — same post-processing)
└── src/
    ├── (most files copied verbatim from Maxwell/src/)
    ├── maxwell_eqs.c      — REWRITTEN (bulk of the work)
    ├── maxwell_eqs.h      — unchanged interface
    ├── derivatives.h      — add SBP-4-2 norm-inverse constants
    ├── simple_maxwell.h   — regenerated or hand-extended to add LOWX/HIGHX/LOWY/HIGHY boundary macros (only if we keep the macro-per-face structure; see §4.4)
    ├── generate_ccode.py  — extended to emit new macros
    ├── parameter.{h,cc}   — add `tau` (SAT penalty strength) and per-face "incoming data mode"
    ├── maxwell_parameters.cc — parse the new fields
    ├── maxwell.toml       — x,y become non-periodic; add `tau = 1.0`
    └── tests/             (copy; add a new penalty-specific test — see §6)
```

Keep the original `Maxwell/` tree completely untouched. `Maxwell_Penalty/src/Makefile` and `CMakeLists.txt` are copied with no change beyond filenames (both already produce `maxwell_system`).

---

## 3. SBP-SAT Mathematics — per-axis flux decomposition

PenaltyImplementation.md in `Maxwell/src/` already derives the z-direction. We need x and y too. Let `c = √(ieps · imu)` and `PP = √(ieps/imu)`. By the symmetry of the extended Maxwell + constraint-damping system, cycling `(x, y, z) → (y, z, x) → (z, x, y)` on the block structure gives:

| Axis | EM-block-1 vars | EM-block-2 vars | Div-B block    | Div-D block    | Zero mode |
|------|-----------------|-----------------|----------------|----------------|-----------|
| z    | (Bx, Dy)        | (By, Dx)        | (Bz, PsiB)     | (Dz, PsiD)     | rho       |
| x    | (By, Dz)        | (Bz, Dy)        | (Bx, PsiB)     | (Dx, PsiD)     | rho       |
| y    | (Bz, Dx)        | (Bx, Dz)        | (By, PsiB)     | (Dy, PsiD)     | rho       |

The 2×2 EM block has eigenvalues `±c`; the Div blocks have eigenvalues `±1`. `rho` has eigenvalue 0 and never contributes to the SAT.

**The positive-eigenvalue-flux operator `A⁺_axis · du` at the *lower* boundary of each axis:** (these must be derived and verified against `generate_ccode.py`; the z-axis version is already given in PenaltyImplementation.md:49–66). Do these derivations with SymPy in `generate_ccode.py` and emit them as C macros `SAT_LOW{X,Y,Z}_PLUS(du...)`. The corresponding `A⁻_axis` for the *upper* boundary is obtained by flipping the sign of the coupling term in each eigenvector (same pattern shown for z in PenaltyImplementation.md:133–144).

The **SAT penalty** at a boundary point on face F is:

```
dotU[ijk] -= τ · H_inv_axis_bdry_index / h_axis · (A±_axis · (u[ijk] - g_F(x,y,z,t)))
```

with `H_inv` from the SBP-4-2 norm (at i = 0,1,2,3 the boundary-row values are 48/17, 48/59, 48/43, 48/49; symmetric at the upper end). **Crucial:** the SAT weight is applied *only at the boundary point itself* (i = 0 or i = N-1). The three interior SBP rows (i = 1,2,3) receive SBP closures for their derivative, but no SAT — they're handled by the norm integral in the energy argument.

Boundary data `g_F`:
- Lower-z: `g = incoming_plane_wave(x, y, 0, t)` — the full analytic state.
- All other faces: `g = 0` for all evolved vars.

---

## 4. Rewriting `maxwell_eq_time_deriv`

### 4.1 Region decomposition

With every face potentially physical, the `(i, j, k)` grid splits into 27 regions by how close each axis is to a physical boundary. Using *axis tags* `X ∈ {L, I, R}` (lower boundary proximity / interior / upper boundary proximity), there is

- **1 interior region** (I, I, I)
- **6 face slabs** (one tag ≠ I)
- **12 edge rods** (two tags ≠ I)
- **8 corner cubes** (all three tags ≠ I)

Each "L" or "R" region is **4 grid points deep** (SBP-4-2 uses the first/last 4 rows). If a given axis is *not* a physical boundary (periodic, or an internal MPI face), its tag collapses to "I" and the regions merge.

### 4.2 Stencil choice

Per axis, per point:
- Tag `I` → `D4CEN` (4th-order centred; ghost zones already synced).
- Tag `L` with offset `p ∈ {0,1,2,3}` → `SBP42_L{p}`.
- Tag `R` with offset `p ∈ {0,1,2,3}` → `SBP42_RN{-p}` (i.e. `RN`, `RNm1`, `RNm2`, `RNm3`).

All three stencil families are already in `derivatives.h` and take a stride, so they work on any axis unchanged.

### 4.3 Penalty placement

Per point, per axis, the SAT term fires **only** at the exact boundary row:
- Tag `L`, offset `0` → add `-τ · (48/17) / h_axis · A⁺_axis · (u - g_lower)` to `dot`.
- Tag `R`, offset `0` (i.e. point `N-1`) → add `-τ · (48/17) / h_axis · A⁻_axis · (u - g_upper)` to `dot`.
- All other offsets: no SAT, only the SBP derivative replaces the centred one.

**The SAT from different faces is additive.** A corner point gets three SATs (one per face it belongs to); an edge point gets two; a face point gets one. No cross-term corrections. This additivity is the defining convenience of SBP-SAT in multi-D and is the property that keeps the energy estimate valid at edges/corners — each face's penalty absorbs *its own* boundary surface integral independently.

### 4.4 Code-layout choice

Two viable implementations. Recommend the second.

**Option (a) — 27 explicit regions.** Parallels the existing code's face-by-face structure. Each region opens its own scope, `#define`s `DIFFX/Y/Z` to the appropriate stencil, loops, emits optional SAT. Very verbose (~1500 lines of boilerplate), but every region is independently readable. Good for pedagogy, bad for maintenance.

**Option (b) — Unified sweep with per-point stencil dispatch.** One triple loop over the entire local grid. At each point compute `tag_x, tag_y, tag_z` and `off_x, off_y, off_z`. Define `DIFFX(f_)` as a small `switch`/macro that selects among six stencils based on `tag_x, off_x`. Then `SIMPLE_MAXWELL_INTERIOR_DOT` (unchanged) fills `dot`. Finally, per-axis `if (tag_? == L && off_? == 0) apply_sat_lower_?`. Compilers inline and hoist the switch out of the innermost loop when `tag` is loop-invariant per j or per i, so performance is acceptable; more importantly the *semantics* are in one place.

Recommendation is **Option (b)** with a clear split:
- The **deep interior** (all three tags = I) is peeled off into the fast, unconditional 3-deep nested loop exactly as today (it dominates the runtime).
- The **boundary shell** (any tag ≠ I) is swept by a single loop that computes tags/offsets from `(i, j, k)` and `bbox`, chooses stencils, calls the body, and emits per-face SAT. This shell is O(surface) so the overhead is immaterial.

### 4.5 Extending `simple_maxwell.h`

The generator `generate_ccode.py` currently emits `SIMPLE_MAXWELL_INTERIOR_DOT`, `SIMPLE_MAXWELL_LOWZ_BOUNDARY`, `SIMPLE_MAXWELL_HIGHZ_BOUNDARY`. For the penalty method we *keep* `SIMPLE_MAXWELL_INTERIOR_DOT` (it is the RHS everywhere — interior, face, edge, corner) and *add* six new macros:

```
SIMPLE_MAXWELL_SAT_LOWX(du_struct, tau_hinv_over_h, ieps_local, imu_local)
SIMPLE_MAXWELL_SAT_HIGHX(...)
SIMPLE_MAXWELL_SAT_LOWY(...)
...
```

Each emits `dotU -= τ H⁻¹/h · A±_axis · du`. The existing `SIMPLE_MAXWELL_LOWZ_BOUNDARY` / `HIGHZ_BOUNDARY` macros (strong injection) are **not used** and can be deleted from the new `simple_maxwell.h`. Regenerate via `python generate_ccode.py && clang-format -i simple_maxwell.h` (extended to produce the SAT macros).

### 4.6 Changes beyond `maxwell_eqs.c`

- `derivatives.h`: add the four `SBP42_HINV_{0,1,2,3}` constants from PenaltyImplementation.md:25–33. (Only index 0 is actually used by the SAT because SAT fires only at row 0.)
- `parameter.h` / `maxwell_parameters.cc` / `maxwell.toml`: add `tau` under `[solver]` (default `1.0`). Optionally add per-face knobs (`lower_z_incoming = "plane_wave"`, others `= "zero"`) so the same code can do mixed configurations.
- `maxwell.toml`: default `periodic_x = false`, `periodic_y = false`, `periodic_z = false`. Shrink default `ny` (currently 6 is tuned for periodic-in-y; the SAT-in-y test needs at least ~32 to resolve one transverse wavelength).
- `driver.c`: no logic change, but the initial `l2_error_analytic` sanity threshold (`1.0e-8`) has to be relaxed to a few × `dx^4` — the analytic incoming plane wave is *not* an exact steady solution in a finite box with zero-incoming conditions on x/y/upper-z, so the initial data won't be divergence-free in the SAT sense and the pre-evolution sync check will see O(dx^4) truncation error at the SBP rows. Document this; do not tighten back.

---

## 5. Edges and Corners — details

The additivity of SAT terms means no new code per-edge or per-corner, but three things need explicit attention:

1. **SBP stencils compose.** At a corner point `(0, 0, 0)` with all three axes physical, `DIFFX`, `DIFFY`, `DIFFZ` each use their own `SBP42_L0` independently. The stencil operates along one axis using its stride (`di`, `dj`, `dk`) and reads four points along that axis only, so there's no cross-axis issue. This works because the SBP property is a 1D property applied tensor-product-wise.

2. **SAT terms add; H⁻¹ factors do not multiply.** At the corner `(0, 0, 0)` the penalty is

   ```
   dot -= τ [ H⁻¹_x,0/h_x · A⁺_x · (u - g_x)
            + H⁻¹_y,0/h_y · A⁺_y · (u - g_y)
            + H⁻¹_z,0/h_z · A⁺_z · (u - g_z) ]
   ```

   With g_x = g_y = 0 and g_z = g_inc. There is no `H⁻¹_x · H⁻¹_y · H⁻¹_z` term — the derivation goes through the *surface* integral of each face, each weighted by the 2D boundary norm `H_{other1} ⊗ H_{other2}`, not by a 3D product.

3. **Ghost corners in the MPI halo exchange are not used by the SBP rows.** The SBP-4-2 one-sided stencil at `i = 0,1,2,3` reads `f[0..4]` along the x-axis — it never dips into the ghost zone in the x-direction. So the corner/edge cells of the MPI halo (which are only partially filled by the axis-by-axis sync) never appear in a stencil that lives on a physical boundary. Good — no changes to `comm.c`. (Centred stencils inside the interior still need ghosts, but by construction the interior is ≥ 2 points away from any MPI neighbour; the existing sync already covers that.)

4. **Periodic/physical mixing on adjacent axes is fine.** If, e.g., `periodic_x = true` but z is physical, the z-lower face is still a face (not an edge) — because x has no physical boundary to meet it at. The region tagger naturally handles this by keeping the x-tag at `I` everywhere.

---

## 6. Commentary: a paraxial beam impinging on a lens

### 6.1 Physical setup

- Lower-z (`z = 0`): paraxial Gaussian beam injected as incoming data. The beam waist `w₀` is placed somewhere on the z-axis (either at `z = 0` or inside the domain, so the entering beam is converging).
- The elliptical-permittivity region from `[material]` in the TOML (`epsilon_type = "elliptical"`, centre `(eps_x0, eps_z0)`, axes `eps_a, eps_b`, peak `eps_max`) acts as the lens. The existing `set_initial_data` already fills `ieps` from this profile; nothing new is needed for the lens geometry.
- Upper-z (`z = Lz`): `g = 0`, purely absorbing. The beam (now refracted/focused by the lens) exits here.
- ±x, ±y: `g = 0`. Valid because the domain is sized so `exp(−(Lx/2 / w(z))²) ≪ roundoff` everywhere in z.

### 6.2 Why paraxial suffices

The paraxial Gaussian beam (scalar `ψ` with `E_x = ψ`, `B_y = ψ/c` to leading order, plus `O(1/(k w₀))` longitudinal corrections that make it fully transverse-divergence-free) is a solution of the vacuum wave equation and Maxwell's equations **to `O(1/(k w₀)²)`**. For the typical collimated-beam regime `k w₀ ~ 10–50`, this residual constraint violation is 10⁻² to 10⁻⁴ — well above machine precision but far below physics-scale error.

Crucially, the extended Maxwell system this code already integrates is *designed* to absorb exactly this kind of residual:
- `∂_t PsiD = c² (∇·D − 4π ρ) − κ_D · PsiD` (and likewise for B).
- Any seeded constraint violation radiates out as a PsiD/PsiB wave at the characteristic speed and is damped on timescale `1/κ_D`.
- Because the SBP-SAT penalty on Div-D and Div-B blocks (`(Dz, PsiD)` and `(Bz, PsiB)`, eigenvalues `±1`) treats `PsiD`/`PsiB` as outgoing at each face, those transient constraint waves exit cleanly.

So the paraxial approximation is not a flaw of the test setup — it is the setting in which the constraint-damping infrastructure is *used*, rather than idling on an exactly-divergence-free analytic solution.

### 6.3 What the transverse (x, y) SAT penalties see

Away from the beam axis, the x-axis incoming characteristic is `w⁺_x = Bz + PP·Dy` (and a second component `Dz − By/PP`). For a well-contained beam `|Dy(x=0)| ∝ exp(−(Lx/2)²/w(z)²)`, so `w⁺_x` at `x = 0` is bounded by the same factor. With `Lx/2 ≥ 5 w(z)` throughout the domain, `|w⁺_x| ≲ 10⁻¹¹` and the spurious "reflection" from `g = 0` SAT is below the SBP-SAT truncation error. The x, y faces are effectively exact absorbers for this problem.

What happens *if* the domain is too narrow: the Gaussian tail that clips the wall generates a reflected wave proportional to the tail amplitude. This is the honest failure mode — it is continuous in the beam/wall ratio and is diagnosable by varying `Lx`.

### 6.4 Edges and corners under this setup

- **Lower-z × x-face edge (and y-face edge)**: the z-face injects the full beam; the x-face applies `g = 0`. They disagree at the edge by an amount equal to the beam value at `(x=0, z=0)`, which is exponentially small for a well-contained beam. The two SAT penalties add and pull `(Bz, Dy)` toward the incoming beam (from z-SAT) and toward zero (from x-SAT); in the small-tail regime the z-SAT wins numerically because the x-SAT has nothing to correct.
- **Upper-z × x-face edges and corners**: all `g = 0`, consistent. The two/three penalties cooperatively absorb whatever reaches the edge/corner (beam tail, constraint-violation waves, any back-scatter from the lens).
- **Lower-z × upper-z (no such edge)**: irrelevant, they don't touch.

### 6.5 `incoming_gaussian_beam` — new analytic source

Adds to `analytic_solutions.{c,h}`. In SI-like vacuum form for `ε = μ = 1`:

```
w(z)  = w0 * sqrt(1 + ((z - z_waist)/zR)^2),     zR = π w0^2 / λ
R(z)  = (z - z_waist) * (1 + (zR/(z - z_waist))^2)
φ(z)  = arctan((z - z_waist)/zR)                 (Gouy phase)

ψ(x,y,z,t) = (w0/w(z)) · exp(-r²/w(z)²)
           · cos(k z - ω t + k r²/(2 R(z)) - φ(z))

E_x = E0 · ψ           (transverse polarisation)
B_y = E0 · ψ / c       (leading paraxial)
E_z, B_x = O(1/(k w0)) corrections that cancel the transverse divergence

(all other components zero in the leading paraxial approximation)
```

Parameters added to TOML under a new `[beam]` section (replacing `[initial_data]` for beam runs): `w0`, `z_waist`, `lambda` (or `k`), `amplitude`, `polarisation_angle`. The driver populates `analytic_params` from whichever section is active, selected by a new `source_type = "plane_wave" | "gaussian_beam"` switch.

### 6.6 What to put in the code

- Default TOML: `source_type = "gaussian_beam"`, `epsilon_type = "elliptical"`, `periodic_x = periodic_y = false`. This reproduces the "beam into a lens" demo out of the box.
- Keep `source_type = "plane_wave"` + `periodic_x = periodic_y = true` as the reference configuration for formal convergence testing of the scheme.
- Document the `Lx/w0 ≥ 10` rule of thumb (~5 σ clearance on each side) in `doc/` alongside the expected constraint-violation magnitude `O(1/(k w0)²)` and its damping timescale.

---

## 7. Testing and Validation Strategy

1. **Rebuild existing tests** (`test_sync`, `test_rk4`, `test_topology`) — these exercise infrastructure and should be bit-identical to `Maxwell/`. They form a regression gate on the verbatim-copied files.

2. **New test: single-rank non-periodic RHS smoothness.** Run on `nx = ny = nz = 16`, all faces physical. Initialise with an analytic field that **vanishes at every physical boundary** — a rectangular-waveguide mode `Az = sin(π x/Lx) sin(π y/Ly) sin(π z/Lz)` with appropriate `E = −∂ₜA`, `B = ∇×A` is the clean constraint-preserving choice; it has `∇·E = ∇·B = 0` exactly. After one `maxwell_eq_time_deriv` call, `dot` should match the analytic time derivative to `O(dx⁴)` in the interior and `O(dx²)` at the SBP rows — the SAT term vanishes at every face because `u − g = 0 − 0 = 0`.

3. **SBP-SAT convergence test.** With Option B (periodic x, y; SAT on z) and the incoming plane wave, measure `L2` error vs `dx` on a sequence `32³, 64³, 128³` (z-only refinement suffices). Expect 3rd-order convergence — the SBP-4-2 boundary closure is 2nd-order locally but integrates to 3rd-order globally. The existing `l2_norm.dat` output is the right observable.

4. **Energy-monotonicity test.** With the upper-z face set to `g = 0` and no sources, run a Gaussian pulse initial data, monitor the discrete energy `E = ½ uᵀ H u` summed over ranks. With `τ ≥ 1/2`, `dE/dt` should be non-positive to roundoff. A Python post-processor on the HDF5 output suffices.

5. **Gaussian beam through lens — physics demonstration (Option A).** `source_type = "gaussian_beam"`, `epsilon_type = "elliptical"`, all faces non-periodic. Run long enough for the beam to traverse the domain. Observables:
   - Beam intensity profile before the lens matches paraxial propagation (`w(z) = w0·sqrt(1 + (z/zR)²)`).
   - Beam focuses / deflects inside the lens region in a way consistent with geometric optics for the chosen `eps_max`.
   - `PsiD`, `PsiB` integrated over the domain stays bounded and decays between pulses (constraint damping is working).
   - L2 norm at the upper-z face vs time is smooth (no reflections visible) once the beam has exited.

   This is a qualitative-quantitative test: no closed-form reference solution, but each of the four observables above has a sharp pass/fail.

6. **Domain-size convergence for Option A.** Repeat test 5 with `Lx = Ly ∈ {6 w0, 8 w0, 10 w0}` at fixed `dx`. Transverse-face reflection should drop like `exp(−(L/(2w))²)`. Gates the "beam is well-contained" assumption from §6.3.

7. **Checkpoint/recovery regression.** The checkpoint format (see `Checkpoint.md`) is unaffected because the boundary treatment is pure-RHS and doesn't add state. A checkpoint-then-resume run of the new code must be bit-identical to an uninterrupted run. Add this assertion to `run_tests.sh`.

---

## 8. Work Sequence (suggested)

1. Copy tree; get the verbatim build + existing tests green (no behavior change yet).
2. Add `tau` + non-periodic defaults to TOML/parameter parser.
3. Extend `generate_ccode.py` to emit `SIMPLE_MAXWELL_SAT_LOW{X,Y,Z}` / `HIGH{X,Y,Z}` macros; regenerate `simple_maxwell.h`. Verify the emitted z-SAT matches PenaltyImplementation.md by hand.
4. Rewrite `maxwell_eq_time_deriv` with the unified-sweep layout (§4.4). Start with z-faces only (to reproduce PenaltyImplementation.md behaviour under Option B) — convergence test gates this step.
5. Add x, y face penalties. Test 2 and 4 gate this step.
6. Edge/corner integration is automatic per §5; test 2 is the gate.
7. Implement `incoming_gaussian_beam` (§6.5) and the `source_type` / `[beam]` TOML plumbing.
8. Run test 5 (beam through lens) and test 6 (domain-size convergence). These are the physics acceptance gates.
9. Document in `doc/` the paraxial approximation, the constraint-damping transient, and the `Lx/w0` rule of thumb (§6).
