# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Companion to the sibling directory `Maxwell/`. The design document is `../Maxwell/Develop_Penalty.md`; read that first if the SBP-SAT boundary treatment is unclear.

## Build Commands

```bash
cd src
cmake -S . -B build && cmake --build build -j     # canonical
ctest --test-dir build                            # runs tests/run_tests.sh
make maxwell_system                               # Makefile also works
```

Toolchain: `mpicc` / `mpicxx`, `-Wall -O3 -ffast-math -g -fopenmp`, HDF5 (C+HL).

## Running

```bash
mpirun -np <N> ./maxwell_system maxwell.toml
```

Default `maxwell.toml` runs the beam-through-lens demo (Option A): Gaussian beam injected at the lower-z face, all six faces non-periodic with SBP-SAT, elliptical-permittivity lens in the middle of the box. Switch `source_type = "plane_wave"` with `periodic_x = periodic_y = true` for the Option B convergence reference.

## Key differences from `Maxwell/`

The physics (extended Maxwell with constraint damping), infrastructure (domain/comm/rk4/io/checkpoint), analytic sources beyond the beam, and test binaries are identical. What changed:

1. **`maxwell_eqs.c` is rewritten.** Strong characteristic injection on the z-faces is replaced by SBP-SAT penalties on *every* physical face. The function `maxwell_eq_time_deriv` has two loops: a fast deep-interior loop with fixed `D4CEN` stencils, and a boundary-shell sweep that dispatches per-axis stencils at runtime (`stencil_at`, `apply_stencil`). Six `APPLY_SAT_*_{X,Y,Z}` macros add the penalty at each boundary row. At edges and corners the SATs are additive — two or three fire at once; no edge-specific code is needed. `maxwell_constraints` uses the same layout so `cD`, `cB` are valid at every SBP row.

2. **`derivatives.h`** gains four `SBP42_HINV_{0..3}` constants — only `HINV_0` is actually used (SAT only fires at the exact boundary row).

3. **`analytic_solutions.{c,h}`** gains `incoming_gaussian_beam` — a paraxial collimated beam linearly polarised in x, travelling in +z. It includes a smooth `f(t, ramp_a, ramp_b)` turn-on in time (interval set under `[source.gaussian_beam]`; default `[0, 1]`): at `t ≤ ramp_a` the source is zero, ramping to full by `t ≥ ramp_b`. This turn-on is essential — filling the domain at `t = 0` with the paraxial beam (which is not an exact solution of the discrete Maxwell PDE) seeds an instability that grows exponentially even at tiny amplitudes. The ramp lets the scheme adjust to the paraxial data as it enters through the SAT boundary. Lengthen it if you see growth during turn-on at high `k·w0`. The paraxial residual is `O(1/(k·w0)²)` and is absorbed by the `PsiD`, `PsiB` constraint-damping fields.

4. **`parameter.{h,cc}`, `maxwell.toml`** gain:
   - `solver.tau` — SBP-SAT penalty strength (≥ 0.5 required; default 1.0).
   - `source.source_type` — `"plane_wave"` or `"gaussian_beam"`.
   - `[beam]` section — `w0`, `z_waist`, `k`, `amplitude`.

5. **`driver.c`** relaxes the initial-sync L2 check for the beam source (the paraxial source is not exact; roundoff-tight bound would reject the valid initial state).

## Stability notes

- **CFL is tighter than `Maxwell/`.** The existing code ran at `cfl_factor = 0.5`. Option B (z-only SAT) tolerates the same. Option A (all six faces) needs `cfl_factor ≤ 0.2` because three SATs stack at corners and the combined stiffness can overrun RK4's stability limit. The default TOML ships with `0.4` — bump to `0.2` if you see explosive L2 growth.
- **The "plane wave filled in Option A" is physically unstable, not a code bug.** A plane wave is constant in x,y, so it has full-amplitude values at the transverse walls. The `g = 0` SAT there tries to force them to zero, generating spurious reflections that cascade. This is the exact limitation discussed in `Develop_Penalty.md §6`. Use Option B (periodic x,y) for plane-wave tests, and keep Option A for well-contained beams whose tail at the x/y walls is negligible.
- **`k * w0` sets the paraxial accuracy.** Residual is `~1/(k·w0)²`. `k·w0 = 3` gives ~10 % residual (diagnostic only); `k·w0 = 6` gives ~3 % (usable for physics). Correspondingly, `Lx / w0 ≥ 10` keeps the wall tail under `exp(-25)` ≈ 1e-11.

## Visualization

After a run, produce an XDMF sidecar with

```bash
python scripts/make_xdmf.py --dir <run_dir> --dt-per-output <T>
```

where `T = output_every * cfl_factor * min(dx,dy,dz)` (the physical time
between HDF5 output groups). The resulting `maxwell.xmf` opens directly
in ParaView, VisIt, and PyVista (via `pyvista.XdmfReader`) — no data
conversion needed. The HDF5 files are unchanged; the XMF is purely a
manifest that stitches the per-rank slabs into the global grid.

The older `scripts/hdf5_to_vtk.py` (which assembles the grid serially
on rank 0 and writes per-timestep `.vti` files) is still available as a
fallback, e.g., for tools that don't support Xdmf.

## Convergence testing

A correctness / convergence-rate study is provided via
`scripts/convergence_test.py` and the `te_waveguide_mode` source type
(a TE rectangular-waveguide mode propagating in +z; dispersion
`ω² = π²(l² + m² + n²)`). The test uses the analytic
`te_waveguide_mode` as SBP-SAT boundary data on every physical face
(not just lower-z), turning the scheme into an exact-to-analytic
imposition and letting us measure the global L2 convergence rate.

```bash
python scripts/convergence_test.py \
    --solver src/build/maxwell_system \
    --resolutions 16 24 32 48 \
    --final-time 0.1 --np 4
```

The test drives four configurations that progressively add boundary
structures:

| config           | faces | edges | corners | expected L2 rate |
| ---------------- | ----- | ----- | ------- | ---------------- |
| `fully_periodic` |   0   |   0   |    0    |        4         |
| `z_physical`     |   2   |   0   |    0    |        3         |
| `yz_physical`    |   4   |   4   |    0    |        3         |
| `all_physical`   |   6   |  12   |    8    |        3         |

A successful run asymptotes to these rates within ~5 %. Measured rates
for the full run at `N=16,24,32,48` (verified on the as-shipped code):
`fully_periodic` 3.99, all three boundary configurations 3.16 at the
highest refinement — the usual SBP-4-2 signature of approaching 3 from
above. If any rate drops below ~2.5, a regression has been introduced
in the boundary closures or SAT penalty.

`source_type = "te_waveguide_mode"` is intended for this study only.
For physics runs use `"plane_wave"` (Option B reference) or
`"gaussian_beam"` (Option A lens demo).

## Running the infrastructure tests

```bash
cd src/build/tests && bash run_tests.sh           # 14 tests: topology, rk4, sync (multiple configs)
```

These are bit-identical to `Maxwell/` — the infrastructure files were copied verbatim.

## Sign convention warning

**The SAT formulas in `../Maxwell/src/PenaltyImplementation.md` have wrong signs on the coupling terms** (their `w+ = Bx + Dy/PP` should be `Bx − PP·Dy` for the `(Bx, Dy)` z-axis block, and so on). The macros here were re-derived from scratch using the PDE flux signs in `simple_maxwell.h` and verified against the simple traveling-wave solution: an x-polarised +z wave `(Dx, By) = (ψ, ψ)` with all other components zero gives `w+ = By + PP·Dx = 2·ψ` incoming — nonzero, as physics demands. Do not copy SAT formulas back from `PenaltyImplementation.md` without re-deriving; they produce a scheme in which the boundary injection is silently zero for traveling-wave states.

## Smoke-test results

- All 14 infrastructure tests pass.
- Option B (`plane_wave`, periodic x,y): stable at `cfl = 0.4`, L2 error ~0.025 at `t = 4.8` on `32²×64` grid.
- Option A (`gaussian_beam`, all faces non-periodic, elliptical lens): stable at `cfl = 0.2` with `k·w0 = 3`, L2 residual steady at ~0.10 over 500+ steps (paraxial + lens-refraction signature, not a convergence metric). `Dx` field magnitudes reach ~0.67 at mid-domain (consistent with the beam amplitude 1.0 modulated by the paraxial envelope and lens refraction).

**If you see "D is empty" in a plot:** check the time slice. The beam source has a smooth `f(t, ramp_a, ramp_b)` turn-on envelope (default `[0, 1]`), so at `t ≤ ramp_a` (group `/0` in the HDF5 output with the defaults) the field is identically zero. The first non-trivial output is at `t = output_every * dt`; for the default TOML that's at `t = 0.2` where the beam has turned on to ~3 % of full amplitude. By `t = ramp_b` the turn-on completes and the beam is at full strength.
