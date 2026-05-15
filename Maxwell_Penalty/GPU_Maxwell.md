# Kokkos port of `Maxwell_Penalty` — implementation plan

## 0. What stays, what changes, what's at issue

| Component | LOC | Action |
|---|---|---|
| `domain.{c,h}`, MPI Cartesian + decomp | ~330 | **keep verbatim** (CPU bookkeeping) |
| `parameter.{cc,h}`, `maxwell_parameters.cc`, `toml.hpp` | ~280 | **keep verbatim** (CPU TOML parsing) |
| `timer.{c,h}` | ~150 | **keep verbatim** (CPU only) |
| `analytic_solutions.{c,h}` | ~340 | **convert to header-only `KOKKOS_INLINE_FUNCTION`** so the SAT lambda can call it on device |
| `derivatives.h` | ~340 | **annotate every `static inline` with `KOKKOS_INLINE_FUNCTION`** — no other change |
| `simple_maxwell.h` (machine-generated) | ~190 | **regenerate** so the macros expand inside a Kokkos lambda (no global pointer aliases — see §2.3) |
| `gf.{c,h}` | ~325 | **replace** by C++ struct that owns `Kokkos::View`s |
| `rk4.{c,h}` | ~150 | **replace** with a Kokkos RK4 (closer to `wave_rk4.cpp`) |
| `numerical.{c,h}` (`apply_dissipation`) | ~75 | **replace** — single `parallel_for` over interior |
| `maxwell_eqs.{c,h}` | ~790 | **replace** — two `parallel_for`s (deep interior + shell) and one for the constraints; one `parallel_reduce` for L2 |
| `comm.{c,h}` | ~280 | **rewrite pack/unpack as Kokkos kernels**; keep MPI shape; deep-copy buffers to host before MPI when the build is not CUDA-aware (see §2.2) |
| `driver.c` | ~290 | **rename to `driver.cpp`**, add `Kokkos::initialize` / `finalize`, otherwise the same call sequence |
| `io.c`, `HDF5BinaryWrite.c` | ~785 | **keep**; allocate a host mirror of each output field and `Kokkos::deep_copy` before write/checkpoint |
| `tests/` (`test_sync`, `test_rk4`, `test_topology`) | — | port the first two, `test_topology` is unchanged |
| Build system | — | vendor Kokkos as a sub-tree, default `Kokkos_ENABLE_OPENMP=ON`, optional `Kokkos_ENABLE_CUDA=ON` |

Output target: same binary name `maxwell_system`, same TOML, same HDF5 schema, same convergence rates. `Maxwell_Penalty/` becomes a Kokkos build of the existing program, not a fork of the physics.

## 1. Recommendation on MPI

**Keep MPI.** The cost is ~25 lines: the pack/unpack kernels become `Kokkos::parallel_for` over an `IndexBox`, and the buffers become `Kokkos::View<double*>` in device memory. On a non-CUDA-aware MPI install, the `Isend`/`Irecv` calls operate on a host mirror copied with `Kokkos::deep_copy` (one extra copy per face per axis). On a CUDA-aware build, give the device-view `data()` directly to MPI and skip the host hop. The user sees one CMake option (`MAXWELL_CUDA_AWARE_MPI`); the source has a 4-line conditional in `comm.cpp`.

This keeps `domain.{c,h}` intact, keeps the test grid (1/2/4/8/27 ranks), and the physics path (boundary flag → SBP-SAT vs. ghost) is unchanged. Single-GPU just means `mpirun -np 1`. Worth the complication.

## 2. Architectural choices

### 2.1 View layout

One `Kokkos::View<double***, Kokkos::LayoutLeft>` per `(slot, buffer)` pair, indexed `(i, j, k)` with `i` fastest. `LayoutLeft` matches the existing `i + j*nx + k*nx*ny` and gives coalesced loads on CUDA when the inner kernel iterates `i`.

```cpp
using Field3D = Kokkos::View<double***, Kokkos::LayoutLeft>;

struct EvolField {
    Field3D state;   // current "new"
    Field3D old_;    // pre-stage snapshot
    Field3D K1, K2, K3, K4;
};
struct AuxField  { Field3D state; };

struct NGFS {
    EvolField evol[N_EVOL];   // index by slot enum
    AuxField  aux [N_AUX];
    // domain, dx, dy, dz, etc., as in current ngfs
};
```

Drop the "rebind `dot` between stages" trick. Replace it with a `WhichK` parameter to the time-derivative kernel — explicit and Kokkos-friendly. A `Kokkos::View` is a reference-counted handle, so passing it by value into a lambda is cheap; this maps cleanly to the `wave_rk4.cpp` ping-pong pattern.

### 2.2 Communication

`struct face_buffers` becomes:

```cpp
struct FaceBuffers {
    Kokkos::View<double*> src_dev, dst_dev;            // device pack/unpack target
    Kokkos::View<double*>::HostMirror src_host, dst_host;  // populated only if !CUDA-aware MPI
};
```

`transfer_data` (pack/unpack of an `IndexBox` for `nvars` slots) becomes one `parallel_for` over `MDRangePolicy<Rank<3>>` with the box bounds, fused over `vstart..vstart+nvars` as the leading rank or as a sequential outer loop (former is cleaner). `MPI_Isend`/`MPI_Irecv` then takes either `src_dev.data()` (CUDA-aware) or `src_host.data()` after `Kokkos::deep_copy(src_host, src_dev)`.

I would also act on the `Review.md §2.2` remark while I'm in there: post all three axes' non-blocking exchanges before a single `Waitall`. The Maxwell stencils are axis-aligned, so corner ghosts are never read — serialising x→y→z is leftover from a more general code, and on a GPU build the host-hop latency makes the parallelism more valuable. Small change, ~15 lines.

### 2.3 SAT macros must work inside a Kokkos lambda

`APPLY_SAT_*` and `SIMPLE_MAXWELL_INTERIOR_DOT` in `simple_maxwell.h` reference bare names `Bx`, `dotBx`, `ieps`, etc., which today are local pointers materialised by `DECLARE_EVOLVED_VARS`. Inside a `KOKKOS_LAMBDA` with `(int i, int j, int k)`, those names need to read `state(slot, i, j, k)` (or `state_Bx(i, j, k)` after capture).

Easiest: edit `generate_ccode.py` to emit each access as `Bx(i,j,k)` and to drop the `[ijk]` suffix. The lambda then captures `auto Bx = nfgs.evol[BX_SLOT].state;` etc., and `Bx(i,j,k)` is a Kokkos View access.

`make check-generated` already exists — it'll fail until the SymPy emitter is updated, so I'll ship that change with the port.

### 2.4 Kernel structure for `maxwell_eq_time_deriv`

Mirror the existing three-region partition (deep interior + the shell sweeps A/B/C) with separate Kokkos kernels:

1. `parallel_for("dot/interior", MDRange<3>(...interior bounds...), KOKKOS_LAMBDA(i,j,k){ SIMPLE_MAXWELL_INTERIOR_DOT; });` — fixed `D4CEN` everywhere, no branches, hot.
2. `parallel_for("dot/shell-A", ...)` — `k` in z-shell, full `(j,i)`, runtime `stencil_at` per axis. The `switch (s)` inside `apply_stencil` becomes an `if/else` chain that nvcc happily inlines; branch divergence is a thin-region cost.
3. `parallel_for("dot/shell-B", ...)` and `("dot/shell-C", ...)` — same pattern.

Inside the shell kernels, the SAT firing tests (`if (phys_zl && k == 0) ...`) stay; the per-face source-state evaluation moves through the device-callable `analytic_solutions` (§2.5).

After all four kernels, `apply_dissipation` (one more `parallel_for`) fires if enabled, then `sync_vars`.

### 2.5 Analytic source on device

`incoming_plane_wave`, `incoming_gaussian_beam`, `te_waveguide_mode` become `KOKKOS_INLINE_FUNCTION` in a header. They take `analytic_params_st` by value (already POD) and `eb_st *A` by pointer. `source_state` and `sat_boundary_data` move alongside them as inline helpers.

`maxwell_params` and `analytic_params` are file-scope globals in C — that won't compile inside a `__device__` lambda. Fix: capture only the fields the kernel uses, into local `const` doubles before the `parallel_for`. The driver and `maxwell_eq_time_deriv` are already structured that way (see lines 224-231 of `maxwell_eqs.c`); just extend the pattern to `tau`, `four_pi`, `kappa_*`, plus a copy of `analytic_params`.

### 2.6 RK4

Take the `wave_rk4.cpp` shape: a templated stage launcher per slot or fused over slots. Two clean options:

- **Option 1 (closest to current C):** keep the K1..K4 buffers, do four calls to a `time_deriv` lambda that fills `K_i`, then four "compute new = old + c·dt·K_i" kernels. Very small port — RK4 is ~90 lines of Kokkos code.
- **Option 2 (fused like wave_rk4):** carry an accumulator `state_new` and skip materialising K1..K4 separately. Saves four full-domain arrays × 9 fields = 9× memory, but means the time-deriv lambda has to do the accumulation in-place. This complicates the SAT shell sweep.

I recommend **Option 1** for teaching: the pedagogical mapping to the C version is one-to-one, and the memory pressure (5 buffers × 9 evolved fields × 8 bytes/double × Nx·Ny·Nz) is fine on a single GPU for the demo grid sizes (≤ 128³ ≈ 75 MB per buffer ≈ 2.7 GB total).

### 2.7 L2 reduction

`l2_error_analytic` becomes one `parallel_reduce` with sum-of-squares scalar output, then `MPI_Allreduce` on (sum, count). The 9-component contributions add into a single `double` inside the lambda — keeps it a basic reducer, no custom struct needed.

## 3. File-by-file work

```
src/
  CMakeLists.txt              ← rewrite, Kokkos sub-tree build (OpenMP + optional CUDA)
  kokkos/                     ← submodule or in-tree Kokkos
  driver.cpp                  ← from driver.c, +Kokkos init/finalize
  maxwell_eqs.{hpp,cpp}       ← rewrite as Kokkos kernels
  rk4.{hpp,cpp}               ← rewrite as Kokkos RK4
  numerical.{hpp,cpp}         ← rewrite apply_dissipation
  gf.{hpp,cpp}                ← View-owning structs replace gf
  comm.{hpp,cpp}              ← rewrite pack/unpack as Kokkos
  domain.{c,h}                ← unchanged
  parameter.{cc,h}            ← unchanged
  maxwell_parameters.cc       ← unchanged
  toml.hpp                    ← unchanged
  timer.{c,h}                 ← unchanged
  analytic_solutions.hpp      ← merged from .c + .h, all KOKKOS_INLINE_FUNCTION
  derivatives.h               ← KOKKOS_INLINE_FUNCTION on every stencil
  simple_maxwell.h            ← regenerate for View-style accesses
  generate_ccode.py           ← update emitter
  io.{c,h}, HDF5BinaryWrite.{c,h} ← keep, add deep_copy in writer
  maxwell.toml, maxwell_waveguide.toml ← unchanged
  tests/
    test_topology.c           ← unchanged
    test_sync.cpp             ← port to Kokkos buffers
    test_rk4.cpp              ← port to Kokkos buffers
```

## 4. Validation strategy (in order)

1. **Build under OpenMP backend, single rank.** `mpirun -np 1 ./maxwell_system maxwell.toml` should reproduce the C path exactly.
2. **`make check-generated` passes** — confirms the SymPy emitter and the regenerated `simple_maxwell.h` agree.
3. **Sync test (`test_sync`)** at 1/2/4/8/27 ranks under OpenMP — the existing 14-test suite from `tests/run_tests.sh`.
4. **Convergence test** (`scripts/convergence_test.py`, four configs at `N=16,24,32,48`) — must hit the same asymptotes (4.0 fully-periodic, 3.07-3.12 on the boundary configs). This is the strongest correctness check; any sign-flip in the regenerated SAT macros shows up here as a degraded rate.
5. **Switch to CUDA backend, `np 1`.** Re-run convergence test. Compare per-step L2 error against the OpenMP run — should match to roundoff (RK4 + finite-difference is deterministic).
6. **CUDA backend `np 4`.** Confirms the Kokkos pack/unpack + MPI hop is correct.
7. **Smoke runs:** the Option A beam-through-lens demo at `cfl=0.2` and the Option B plane-wave reference, both as documented in `CLAUDE.md`. Compare HDF5 output bit-for-bit (CPU vs CPU) and to within roundoff (CPU vs GPU).

## 5. Performance and teaching notes

- **Why this is a good teaching example:** the existing CPU code has a clean three-region partition (deep interior / shell / SAT), so the GPU port doesn't need new algorithmic ideas. Students see the same structure expressed in `Kokkos::parallel_for`.
- **Where the GPU win shows up:** 9 fields × 4 RK stages × Nx·Ny·Nz cell-pointers per step is ~720 MUpdates/s on a single NVIDIA H100 vs. ~30 MUpdates/s on the OpenMP build at 8 threads (extrapolated from `wave_rk4` results — should benchmark and confirm in §4.5).
- **The SAT shell is branch-heavy on GPU.** Acceptable because the shell volume is `O(N²)` while the interior is `O(N³)` — the shell kernel runtime is asymptotically negligible. A student should be told this; it's a nice illustration of why surface terms don't dominate.
- **Pack/unpack on GPU.** Make this a deliberate stop in the lecture: pure z-faces are contiguous and could be `Kokkos::deep_copy` of a subview; y-faces are stride-`nx`; x-faces are stride-1 with skips. The Kokkos pack is uniform over all three but visibly slower on x; this maps directly to `Review.md §2.1`.

## 6. Risks / things to verify before coding

- **`generate_ccode.py` access pattern.** Need to read it before promising the SymPy emitter change is small. Probable scope: change `Symbol("Bx[ijk]")` to `Symbol("Bx(i,j,k)")` and equivalently for the SAT macros. Worst case: add a templating layer.
- **`HDF5BinaryWrite.c` field ownership.** If it expects a `double *` to a contiguous block, the host mirror produced by `Kokkos::create_mirror_view` plus `Kokkos::deep_copy` gives that — but the API is currently C, so wrap the call from `io.cpp` and pass `mirror.data()`.
- **`__cplusplus` in `parameter.h`.** The header has both C and C++ visible halves through `#ifdef __cplusplus`. `driver.cpp` will compile it as C++, which is fine; the `extern "C"` linkage of `parse_maxwell_parameters` still works.
- **Build of Kokkos itself with CUDA.** On the SporC environment this is already known to work. On a fresh machine: `cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_<arch>=ON`, with the right compiler wrapper. Document in a new `Maxwell_Penalty/src/setup_kokkos.sh` script.

## 7. Suggested order of work

1. Vendor Kokkos in `src/kokkos/`, write a stub `CMakeLists.txt` that builds an empty `maxwell_system.cpp` with `Kokkos::initialize`. Verify CPU-OpenMP and CUDA backends both build on the target machine.
2. Annotate `derivatives.h` with `KOKKOS_INLINE_FUNCTION`, port `analytic_solutions` to a header, regenerate `simple_maxwell.h` via the updated `generate_ccode.py`. Run `make check-generated`.
3. Port `gf.{c,h}` → `gf.{hpp,cpp}` (View-based).
4. Port `rk4` and a stub time-deriv that just zeroes `dot`. Run; confirm RK4 step count and timer.
5. Port `maxwell_eq_time_deriv` deep-interior kernel only — test under fully-periodic config; check the convergence test (`fully_periodic` case must hit rate 4).
6. Add the SBP shell + SAT macros — re-run all four convergence configs.
7. Port `apply_dissipation`, `maxwell_constraints`, `set_initial_data`, `l2_error_analytic`.
8. Port `comm.cpp`. Run `test_sync` at 1/2/4/8/27 ranks under OpenMP.
9. Port `io.cpp` (deep-copy then HDF5). Confirm sample run produces identical HDF5 fields to the C build.
10. Switch backend to CUDA, repeat steps 5–9 verifying parity with OpenMP.

Total scope: roughly 1.4k LOC of computational code rewritten, ~600 LOC of CPU bookkeeping kept as-is, plus build-system and Kokkos vendoring. About a week of focused work for a single developer; faster if tests are parallelised with the port.
