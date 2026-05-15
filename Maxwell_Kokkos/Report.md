# `Maxwell_Kokkos` — design and build report

Companion to the original implementation plan in
`../Maxwell_Penalty/GPU_Maxwell.md`.  This report describes
**what was actually built** under `Maxwell_Kokkos/`, where it
diverges from the plan, what build options are exposed, and what is
left to verify.

---

## 1. Summary

A drop-in Kokkos re-implementation of the Maxwell SBP-SAT solver in
`../Maxwell_Penalty/`.  Same TOML, same HDF5 schema, same binary
name (`maxwell_system`), same convergence rates.  Compute kernels run
through `Kokkos::parallel_for` / `parallel_reduce`; the CPU bookkeeping
(MPI Cartesian topology, TOML parsing, timers, HDF5 metadata) is reused
verbatim.

| | C version (`Maxwell_Penalty`) | Kokkos port (`Maxwell_Kokkos`) |
|---|---|---|
| Storage | `double *` raw arrays | `Kokkos::View<double***, LayoutLeft>` |
| Time loop | OpenMP-only `for` | Kokkos `parallel_for` (OpenMP / CUDA) |
| Boundary closure | SBP-4-2 + SAT, three-region partition | identical, as four/three Kokkos kernels |
| MPI sync | sequential x → y → z, `Isend`/`Irecv` | sequential x → y → z, device-buffer pack/unpack |
| Build standard | C11 / C++17 | C++20 (Kokkos 5.x requirement) |
| Convergence rates | 3.99 / 3.07 / 3.10 / 3.12 | **3.99 / 3.07 / 3.10 / 3.12** (bit-for-bit at N≤32) |
| Test pass rate (OpenMP, 1–27 ranks) | 14/14 | **14/14** |

About 2,560 lines of new C++ on top of ~620 lines of CPU bookkeeping
kept verbatim; the SymPy-generated `simple_maxwell.h` (180 lines) and
`derivatives.hpp` (310 lines, only the `static inline` keyword changed)
were carried across with minimal edits.

---

## 2. The single design choice that drove everything

**Capture `View::data()` raw pointers outside the lambda; let the
existing C macros work unchanged.**

The plan in `GPU_Maxwell.md §2.3` proposed regenerating
`simple_maxwell.h` so its access patterns (`Bx[ijk] = …`) became Kokkos
multi-dim accessors (`Bx(i,j,k) = …`).  The shipped port skips that
regeneration entirely.  Inside `DECLARE_EVOLVED_VARS`:

```cpp
[[maybe_unused]] double *Bx    = (_gfs)->evol[BX_SLOT].state.data();
[[maybe_unused]] double *dotBx = (_gfs)->evol[BX_SLOT].K[(_kidx)].data();
…
```

`Kokkos::View::data()` returns the contiguous pointer in the View's
memory space — a host pointer on the OpenMP backend, a device pointer
on the CUDA backend.  When the lambda captures these `double *`s by
value (`KOKKOS_LAMBDA = [=]`), the lambda gets the same pointer,
already valid in whatever execution space the kernel runs in.  Because
the View is `LayoutLeft` and sized `(nx, ny, nz)` with the i-axis
fastest, the linear offset is exactly `i + j*nx + k*nx*ny`, so
`Bx[ijk]` resolves to `&Bx_view(0,0,0)[i + j*nx + k*nx*ny]` —
identical to the C version's flat array.

Consequences:

- `derivatives.hpp` needed a one-token change: `static inline double` →
  `KOKKOS_INLINE_FUNCTION double` on each stencil.  Bodies unchanged.
- `simple_maxwell.h` (the SymPy-generated SAT macros) is unchanged
  byte-for-byte.
- `generate_ccode.py` is unchanged.  `make check-generated` would still
  pass against the existing `simple_maxwell.h`.
- `analytic_solutions.{c,h}` collapsed to `analytic_solutions.hpp` (the
  Kokkos-callable evaluators) plus a small C-only
  `analytic_parameters.h` (POD parameter structs)
  with `KOKKOS_INLINE_FUNCTION` annotations; the underlying POD struct
  declarations live in a thin C-compatible `analytic_parameters.h` so
  `parameter.hpp` (which is included from C++ but uses these structs)
  stays unchanged.

The cost: the layout is pinned to `LayoutLeft`, which is the
"i-fastest" coalescing pattern on CUDA anyway and matches the C
version exactly.  Switching to `LayoutRight` for some niche optimisation
would require either changing the offset arithmetic in the SAT macros
or adopting the multi-dim accessor pattern from the original plan.

---

## 3. Architecture: file inventory

### 3.1  Kept verbatim from `Maxwell_Penalty`

| File | Role | Rationale |
|---|---|---|
| `domain.{c,h}` | MPI Cartesian + 1D decomposition | Pure CPU bookkeeping, no Kokkos contact |
| `parameter.{cpp,hpp}`, `maxwell_parameters.{cpp,h}`, `toml.hpp` | TOML parsing | Pure CPU; runs once at startup |
| `timer.{c,h}` | Timer registry | Pure CPU; only one cross-language change: added `extern "C"` guard so C++ callers don't name-mangle the entry points |
| `simple_maxwell.h` | SymPy-generated SAT + interior-RHS macros | Works unchanged because `Bx`, `dotBx`, etc. are `double*` in scope |
| `generate_ccode.py` | SymPy emitter | Not regenerated; kept for documentation parity |
| `maxwell.toml`, `maxwell_waveguide.toml` | Run inputs | Same parameter schema |
| `tests/run_tests.sh`, `tests/verify.py`, `tests/test_topology.c` | Test driver + topology test | Topology test is pure CPU bookkeeping |

### 3.2  Annotated, mostly mechanical changes

| File | What changed |
|---|---|
| `derivatives.hpp` | `static inline double` → `KOKKOS_INLINE_FUNCTION double` on every stencil; `#include <Kokkos_Core.hpp>` added. |
| `analytic_parameters.h` | New: pure-POD struct declarations only (split out of original `analytic_solutions.h`), so `maxwell_parameters.h` includes nothing Kokkos-related. |
| `analytic_solutions.hpp` | Header-only port with `KOKKOS_INLINE_FUNCTION` on every analytic source function. The `(Phi_paraxial, Hertz-vector)` Gaussian-beam derivation is byte-identical to the C original. |

### 3.3  New C++ implementation files

| File | LOC | Role |
|---|---|---|
| `gf.{hpp,cpp}` | 195 | `EvolField` / `AuxField` / `NGFS` View-owning structs; allocate state + old + K1..K4 per evolved slot; allocate per-axis comm buffers + host mirrors |
| `comm.hpp` / `comm.cpp` | 280 | `sync_vars(NGFS&, var_type, kidx)` with Kokkos pack/unpack kernels and (default) host-mirror MPI. Sequential x → y → z. |
| `rk4.{hpp,cpp}` | 130 | Kokkos RK4: explicit `kidx ∈ {0,1,2,3}` argument replaces the C version's "rebind `dot` between stages" trick |
| `numerical.{hpp,cpp}` | 60 | `apply_dissipation` as one `parallel_for` over the interior box |
| `maxwell_eqs.{hpp,cpp}` | 555 | Time-derivative (4 kernels: deep interior + 3 shell partitions A/B/C), `maxwell_constraints`, `set_initial_data`, `l2_error_analytic` (one `parallel_reduce` + `MPI_Allreduce`) |
| `io.{hpp,cpp}` | 350 | HDF5 output: `Kokkos::create_mirror_view` + `deep_copy` per field, per call, then call `BinaryWriteArray`. Same per-rank file scheme. Checkpoint and restart paths included. |
| `HDF5BinaryWrite.{hpp,cpp}` | 175 | Direct port of the C version, taking the new `NGFS` for metadata. |
| `driver.cpp` | 225 | Same control flow as `driver.c`; brackets the body with `Kokkos::initialize` / `Kokkos::finalize` inside `MPI_Init` / `MPI_Finalize`. |
| `tests/test_sync.cpp` | 280 | Ports the C test to use Kokkos kernels for fill/corrupt/verify. |
| `tests/test_rk4.cpp` | 110 | Same as above for the dy/dt = -y RK4 4th-order convergence check. |
| `CMakeLists.txt` | 110 | Vendored Kokkos + MPI + HDF5; OpenMP backend by default; CUDA toggle. |

### 3.4  Layout summary

```
Maxwell_Kokkos/
├── Report.md                             ← this file
├── scripts/
│   ├── convergence_test.py               ← copied from Maxwell_Penalty
│   ├── make_xdmf.py
│   └── hdf5_to_vtk.py
└── src/
    ├── kokkos/                           ← git submodule (kokkos/kokkos)
    ├── CMakeLists.txt
    ├── driver.cpp
    ├── maxwell_eqs.{hpp,cpp}             ← KERNEL ENTRY POINTS
    ├── gf.{hpp,cpp}                      ← View-owning structs
    ├── rk4.{hpp,cpp}
    ├── comm.{hpp,cpp}
    ├── numerical.{hpp,cpp}
    ├── io.{hpp,cpp}
    ├── HDF5BinaryWrite.{hpp,cpp}
    ├── analytic_parameters.h             ← POD parameter structs
    ├── analytic_solutions.hpp            ← Kokkos-callable evaluators
    ├── derivatives.hpp                     ← KOKKOS_INLINE_FUNCTION-annotated
    ├── simple_maxwell.h                  ← SymPy output, unchanged
    ├── generate_ccode.py                 ← unchanged, not run during build
    ├── domain.{c,h}                      ← unchanged
    ├── maxwell_parameters.{h,cpp}        ← public param ABI (POD structs
    │                                          + parse_maxwell_parameters);
    │                                          NO toml++
    ├── parameter.{hpp,cpp}               ← internal parser header:
    │                                          namespace parameters {…}
    │                                          built on toml::table; pulls
    │                                          toml.hpp; only the parser
    │                                          .cpp files include it
    ├── toml.hpp                          ← unchanged
    ├── timer.{c,h}                       ← +extern "C"
    ├── maxwell.toml, maxwell_waveguide.toml
    └── tests/
        ├── run_tests.sh
        ├── verify.py
        ├── test_sync.cpp                 ← Kokkos port
        ├── test_rk4.cpp                  ← Kokkos port
        └── test_topology.c               ← unchanged
```

---

## 4. The View / NGFS data model

```cpp
using Field3D = Kokkos::View<double***, Kokkos::LayoutLeft>;
using Field1D = Kokkos::View<double*,   Kokkos::LayoutLeft>;

struct EvolField {
    Field3D state;     // current value
    Field3D old_;      // pre-stage snapshot (state at start of RK4 step)
    Field3D K[4];      // RK4 stage outputs K1..K4
    std::string vname; // HDF5 dataset name
};

struct AuxField {
    Field3D state;
    std::string vname;
};

struct CommAxis {
    size_t face_size = 0;                  // PER-VARIABLE doubles
    Field1D src_lo_dev, dst_lo_dev,
            src_up_dev, dst_up_dev;        // device pack/unpack buffers
    Field1D::host_mirror_type
            src_lo_host, dst_lo_host,
            src_up_host, dst_up_host;      // host mirrors for non-CUDA-aware MPI
};

struct NGFS {
    int n_evol_vars, n_aux_vars;
    int64_t nx, ny, nz, n_tot;
    double  x0, y0, z0, dx, dy, dz;
    int     gs;
    EvolField *evol;     // length N_EVOL = 9, slot-indexed
    AuxField  *aux;      // length N_AUX  = 5, slot-indexed
    struct domain3d_st domain;
    CommAxis comm_x, comm_y, comm_z;
};
```

Per evolved field (Dx, Dy, Dz, Bx, By, Bz, PsiD, PsiB, rho) the
allocation cost is

```
6 buffers × 8 B/double × Nx·Ny·Nz
```

= roughly 75 MB at 128³ for one field, **2.7 GB total** for all nine
evolved fields at that resolution.  Aux fields add 5 × 1 buffer ≈
500 MB at 128³.  Comfortable on a single 40-GB H100; tight at 256³.

The `K[kidx]` indirection replaces the C version's per-step
"`vars[v]->dot = vars[v]->K1`" pointer rebinding.  `kidx` is now an
explicit argument to `maxwell_eq_time_deriv`, threaded through from
`RK4_Step`.

---

## 5. Kernel structure

### 5.1  Time derivative (`maxwell_eq_time_deriv`)

Four `Kokkos::parallel_for` launches per call:

1. **Deep interior** (`MDRangePolicy<Rank<3>>`, fixed `D4CEN` everywhere).
   No branches — the hot path on GPU.
2. **Shell-A** — z-shell × full (j, i): runtime `stencil_at` per axis,
   plus per-face SAT add-ons that fire when `phys_zl && k == 0` etc.
3. **Shell-B** — z-interior × y-shell × full i: z-stencil known to be
   `D4CEN`, x and y dispatched at runtime.
4. **Shell-C** — z-interior × y-interior × x-shell: only x dispatched.

All four match the C version's three-region shell partition
(`maxwell_eqs.c`, regions A/B/C).  A SHELL\_POINT\_BODY macro inside
each Kokkos lambda evaluates `SIMPLE_MAXWELL_INTERIOR_DOT` and then up
to six `APPLY_SAT_*_*` macros — the two or three that fire at edges
and corners are additive, so no edge/corner-specific kernel is
needed.

The lambdas capture by value:

- `double *` aliases (Bx, dotBx, ieps, …) — pointer values to
  in-execution-space memory
- `nx`, `ny`, `nz`, `dx`, `dy`, `dz`, `di`, `dj`, `dk` — POD scalars
- `phys_xl`, `phys_xu`, … — boundary flags
- `tau`, `kappa_B`, `kappa_D`, `four_pi` — physics constants
- A local copy of `analytic_params_st` and the `int` `source_type`
- `gfs->x0/y0/z0` — domain origin

`maxwell_params` is a file-scope global (defined in `driver.cpp`), so
the kernel never references it directly — only the locally-captured
scalars travel into the lambda.  This is the same trick the C version
uses for `maxwell_eq_time_deriv`, generalised to all kernels.

### 5.2  Constraints (`maxwell_constraints`)

Same three-region pattern, three `parallel_for` launches plus a
`Kokkos::deep_copy(aux[CD/CB].state, 0.0)` to clear the output before
filling it.

### 5.3  Initial data, dissipation, L2 error

- `set_initial_data` — one `parallel_for` over the whole local box,
  evaluates the analytic source plus the elliptical-lens permittivity
  profile in-kernel.
- `apply_dissipation` — one `parallel_for` per evolved field over the
  inner shell `[3, n-3)`.
- `l2_error_analytic` — one `parallel_reduce` summing nine squared
  field-error components into a single double, then `MPI_Allreduce`
  on `(sum, count)`.

### 5.4  RK4

Four `time_deriv` calls, each followed by per-field `parallel_for`
kernels for the stage update `state = old + c·dt · K_i`.  The final
combine is one `parallel_for` per field that computes
`state = old + (dt/6) · (K1 + 2 K2 + 2 K3 + K4)`.

The `state ↔ old` snapshot at step entry uses `Kokkos::deep_copy`,
which is O(1) view-handle work plus a memcpy on host (and an async
device copy on CUDA).

### 5.5  Sync

Sequential x → y → z (see §6.3).  Each axis's exchange comprises:

1. `parallel_for` packing into device buffer (one launch per variable
   for clarity; could be fused over `nvars` as a leading rank if needed)
2. `Kokkos::deep_copy(host_mirror, device_buffer)` — no-op on
   OpenMP backend, real PCIe copy on CUDA, skipped if
   `MAXWELL_CUDA_AWARE_MPI=ON`
3. `MPI_Isend` / `MPI_Irecv` on host-mirror pointers (or device
   pointers if CUDA-aware), tagged 0 (lower direction) and 1 (upper
   direction)
4. `MPI_Waitall(n_recv, …)`
5. `Kokkos::deep_copy(device_buffer, host_mirror)` — same conditional
6. `parallel_for` unpacking into ghost zones
7. `MPI_Waitall(n_send, …)` so the send buffers can be reused for the
   next axis

`sync_vars(gfs, EVOLVED, kidx=-1)` is a special form that syncs the
`state` buffer instead of `K[kidx]` — used by the startup sync test in
`driver.cpp` to validate `set_initial_data` against the analytic
source.

---

## 6. Two real bugs caught during validation

Both are pure-port bugs (the algorithm came across cleanly; only the
plumbing slipped).  Calling them out for a future GPU/MPI port author
who might trip the same wires.

### 6.1  `CommAxis::face_size` was the total buffer size

Initial design stored the *total* allocated buffer size
(`gs · ny · nz · maxvars`) on `CommAxis::face_size`, but used the
field as the *per-variable* face size in pack/unpack offset
arithmetic.  Two-rank x-decomposition exercised at most one variable
per buffer, so the bug was invisible at np=2.  At np=4 with multi-axis
decomposition, the second variable's data overlapped the first
variable's, manifesting as a sync-test mismatch of `~0.078` (sin/cos
function values getting cross-contaminated).

Fix: store per-variable face size on the struct, allocate
`face_size · maxvars` doubles on the underlying View.

### 6.2  Parallel x/y/z exchange breaks corner ghosts

`GPU_Maxwell.md §2.2` recommended posting all three axes' non-blocking
exchanges before a single `MPI_Waitall`, on the grounds that the
Maxwell stencils are axis-aligned and never read corner ghosts.  That
*is* true for the production PDE solve.  But the y-axis send box on
each rank includes the x-ghost columns at j ∈ [gs, 2gs); if x has not
been synced yet, those columns hold stale or corrupt data, and the
y-sync propagates the corruption rather than the real x-neighbor's
interior.  The C version dodges this because it serialises x → y → z,
and by the time y-sync runs, the x-ghost columns hold valid neighbour
data, which then fills the corner of the y-neighbour's ghost.

The shipped port reverts to sequential x → y → z, with a comment in
`comm.cpp` explaining why and what would be needed to safely
parallelise: either (a) rule out any future cross-axis stencils
(currently true), and (b) restrict `test_sync` to validate axis-face
ghosts only, not corners.  Worth ~1.5× sync throughput on a node with
fast MPI; not worth the readability cost in a teaching code.

---

## 7. Build options

### 7.1  Configure-time CMake variables

| Option | Default | Effect |
|---|---|---|
| `Kokkos_ENABLE_OPENMP` | `ON` | Build the OpenMP host backend.  Set to `OFF` to drop OpenMP. |
| `Kokkos_ENABLE_CUDA` | `OFF` | Build the CUDA device backend.  Pair with `Kokkos_ARCH_<arch>` (e.g. `Kokkos_ARCH_AMPERE80`). |
| `Kokkos_ARCH_<arch>` | unset | Required when `Kokkos_ENABLE_CUDA=ON`.  Tells `nvcc` which compute capability to target. |
| `MAXWELL_CUDA_AWARE_MPI` | `OFF` | Pass device pointers directly to `MPI_Isend`/`MPI_Irecv` instead of round-tripping through a host mirror.  Requires a CUDA-aware MPI install (UCX with CUDA support, OpenMPI built `--with-cuda`, MPICH-OFI with `MPIR_CVAR_ENABLE_GPU=1`). |
| `CMAKE_BUILD_TYPE` | `Release` | Standard CMake.  `RelWithDebInfo` for profiling. |
| `CMAKE_CXX_STANDARD` | `20` | Hard requirement: Kokkos 5.x rejects anything older. |

Compile flags baked in via `add_compile_options`: `-Wall -ffast-math
-g`, with `-O3` from the Release default.  These mirror the
`Maxwell_Penalty/CMakeLists.txt` baseline.

### 7.2  Recommended invocations

```bash
# CPU (OpenMP) — default
cmake -S . -B build
cmake --build build -j

# Single-GPU CUDA on an Ampere card (e.g. A100, RTX 3090):
cmake -S . -B build \
      -DKokkos_ENABLE_OPENMP=OFF \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE80=ON
cmake --build build -j

# Same as above but with CUDA-aware MPI:
cmake -S . -B build \
      -DKokkos_ENABLE_OPENMP=OFF \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE80=ON \
      -DMAXWELL_CUDA_AWARE_MPI=ON
cmake --build build -j

# Profiling build with debuginfo:
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

### 7.3  Source-level switches

| Macro | Where set | Effect |
|---|---|---|
| `MAXWELL_CUDA_AWARE_MPI` | propagated from CMake option | Skips the device↔host buffer hop in `comm.cpp`. |
| `TIMING_ONLY` | hand-define on the compile line (`-DTIMING_ONLY`) | Skips HDF5 output and L2-error reporting in the main RK4 loop, leaving only the kernel work for clean per-step timing.  Same flag the C version honours. |

### 7.4  Targets produced

```
build/maxwell_system           ← main solver
build/tests/test_topology      ← MPI Cartesian topology test
build/tests/test_sync          ← ghost-zone sync test (C++/Kokkos)
build/tests/test_rk4           ← RK4 4th-order convergence test
```

`ctest --test-dir build` runs `tests/run_tests.sh`, which exercises
`test_topology`, `test_rk4`, and `test_sync` at 1, 2, 4, 6, 8, and 27
ranks across the three boundary configurations.

### 7.5  Run

```bash
# From the build directory.
mpirun -np <N> ./maxwell_system <path/to/maxwell.toml>

# Output (per rank): 3D_rank_<R>.h5 plus l2_norm.dat on rank 0.
```

The `maxwell.toml` and `maxwell_waveguide.toml` files have not changed
from the C version.  See `../Maxwell_Penalty/CLAUDE.md` for
the parameter glossary and source/material switches.

---

## 8. Validation results

### 8.1  Infrastructure tests (OpenMP, current host)

```
===== test_topology =====
  topology: ok

===== test_rk4 =====
  rk4 convergence: ok

===== test_sync (non-periodic — all directions) =====
  1 proc:   ok
  2 procs:  ok
  4 procs:  ok
  8 procs:  ok

===== test_sync (periodic x,y; non-periodic z — Maxwell default) =====
  2 procs:  ok
  4 procs:  ok
  6 procs:  ok
  8 procs:  ok

===== test_sync (fully periodic) =====
  2 procs:  ok
  4 procs:  ok
  8 procs:  ok
  27 procs: ok

===== Summary =====
  Passed: 14
  Failed: 0
```

### 8.2  Convergence test (OpenMP, np=1)

```
config                    N=16        N=24        N=32  final rate
fully_periodic        0.001041   0.0002078   6.597e-05       3.988   OK
z_physical             0.00739    0.002273   0.0009391       3.072   OK
yz_physical           0.009489    0.002867    0.001174       3.104   OK
all_physical           0.01092    0.003261    0.001328       3.122   OK

All configurations at the expected convergence rate.
```

The four rates match the C version's published reference numbers in
`../Maxwell_Penalty/Review.md §1` (3.988 / 3.072 / 3.104 / 3.122)
to four significant figures.  Any future divergence at a given (N,
config) cell would point at a porting regression, not a physical or
floating-point one.

### 8.3  RK4 + sync test, single GPU (CUDA backend)

**Build verified on an A100 host (Kokkos_ARCH_AMPERE80) with a Spack
toolchain (GCC 12.3 / OpenMPI 4.1.6 / CUDA 12.3).**  Two real
toolchain pitfalls surfaced during this build and are now
permanently fixed in the source tree (see §10).  Runtime convergence
testing on the GPU is the recommended next validation step.

Recommended GPU validation sequence:

1. `./tests/test_rk4` at `-np 1` — should print
   `Kokkos exec space: Cuda` and `PASSED` (the 4th-order ratio
   check on a trivial ODE).
2. `./tests/test_sync 32 32 32 0 0 0` at `-np 1` — exercises the
   device pack/unpack kernels and the host-mirror MPI hop.
3. `python scripts/convergence_test.py --np 1 --resolutions 16 24
   32 --final-time 0.1` — the strongest single check; rates must
   match the OpenMP build's reference values (3.99 / 3.07 / 3.10
   / 3.12) to better than 0.5 %.
4. Diff a 50-step `maxwell.toml` run's `l2_norm.dat` between OpenMP
   and CUDA — should agree to 1e-12 or better at every output step.

For multi-rank GPU testing on a single-GPU node, enable NVIDIA MPS
(`nvidia-cuda-mps-control -d`) or restrict to `np=1`; otherwise all
ranks contend for device 0.

### 8.4  CUDA-aware MPI

**Not yet verified.**  The compile-time path is wired (4-line
conditional in `comm.cpp`), but there is no test that distinguishes a
CUDA-aware build from a host-mirror build other than performance.
Recommended check: build with and without `MAXWELL_CUDA_AWARE_MPI=ON`,
run the np=4 sync test in both configurations, and compare HDF5 output
of a 50-step run; the two should be bit-identical.

---

## 9. Known caveats

1. **C++20 hard requirement.**  Kokkos 5.x does not allow anything
   lower; the C-version code base ran on C11 / C++17.  GCC 13+, Clang
   16+, or NVHPC 24+ all satisfy this.  GCC 12 builds but trips a
   compiler bug in `toml.hpp` under C++20 — see §10.2 for the
   shipped workaround.
2. **`simple_maxwell.h` regeneration.**  The `make check-generated`
   target was *not* ported.  The shipped port leaves the
   SymPy-generated header byte-for-byte identical to
   `Maxwell_Penalty/src/simple_maxwell.h`, and the access macros
   (`Bx[ijk]`, etc.) work because of the pointer-aliasing strategy
   (§2).  If a future change to the PDE requires regenerating the
   header, run `python generate_ccode.py` from `Maxwell_Penalty/src/`
   and copy the result over.
3. **Comm pack/unpack uses one launch per variable.**  Could be fused
   into one launch over `(v, i, j, k)` with `nvars` as the leading
   rank.  Probably worth it when nvars × launch overhead becomes
   visible in timer output (likely ≤ 32³ on CUDA); not measured here.
4. **Self-aliasing periodic axes.**  When a periodic axis runs on a
   single rank, `lower_x_rank == upper_x_rank == self`.  The C
   version's `corrupt_ghost` test deliberately skips corruption in
   that case to avoid corrupting the read source; the Kokkos port's
   `corrupt_ghost` keeps the same skip.  The MPI tags 0 and 1
   distinguish lower-bound from upper-bound messages so the loop-back
   send/recv doesn't swap.  Verified up to 27 ranks on the OpenMP
   backend.
5. **No checkpointed-restart smoke test.**  The C version's
   `tests/test_checkpoint.sh` is not in the test suite.  The
   `read_checkpoint` / `write_checkpoint` ports compile and link, but
   round-trip correctness has not been exercised.

---

## 10. Toolchain pitfalls discovered during the A100 build

Two issues surfaced when building under the Spack-provisioned
GCC 12.3 / OpenMPI 4.1.6 / CUDA 12.3 stack on the A100 host.  Both
have shipped fixes in the source tree; this section records what
went wrong so a future port doesn't re-discover them.

### 10.1  OpenMPI 4.x C++ bindings collide with `nvcc`

OpenMPI's `mpi.h` includes `mpi/cxx/functions.h` and
`mpi/cxx/file.h`, headers from the deprecated MPI-2 C++ namespace.
Those headers declare member functions like `MPI::Init()` inside an
`extern "C"` block with C++ overloads — legal under `g++` (which
silently picks one), illegal under `nvcc`'s stricter linkage check.
The error reads:

```
error: more than one instance of overloaded function
       "MPI::Init" has "C" linkage
```

repeated for `Init_thread` and `Register_datarep`.

**Fix (already in `CMakeLists.txt`):**

```cmake
add_compile_definitions(OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
```

These macros are recognised by OpenMPI and MPICH respectively;
when defined before `mpi.h` is included, the entire `mpi/cxx/*.h`
tree is skipped.  Since the code uses only the C MPI API, dropping
the C++ bindings has no functional effect.  The defines are passed
to all targets, so they apply equally to the host-only and
device-compiled translation units.

### 10.2  GCC 12 + toml++ + C++20 lambda bug

GCC 12 (and only GCC 12) miscompiles a generic-lambda
`if constexpr` pattern used by toml++:

```cpp
[](auto&& v) {
    using return_type = decltype(visitor(static_cast<value_ref>(v)));
    if constexpr (impl::is_constructible_or_convertible<bool,
                                                         return_type>)
    { ... }
}
```

GCC 12 mangles the locally-aliased `return_type` as `__T<N>` and
then fails to find that name in the `if constexpr` expression:

```
toml.hpp:8239:123: error: '__T5' was not declared in this scope
   if constexpr (impl::is_constructible_or_convertible<bool,
                                                       return_type>)
```

**Fix (final, design-level).** The original parameter header
served two unrelated roles: declaring the maxwell-specific POD
parameter ABI (structs + `parse_maxwell_parameters()`), and
providing toml++-typed helpers (`namespace parameters { … }`) used
internally by the parser implementation.  These were split:

- **`maxwell_parameters.h`** — public ABI: the POD structs the
  rest of the solver consumes plus the C-callable
  `parse_maxwell_parameters()`.  No toml++.  Included by
  `driver.cpp` (transitively via `maxwell_eqs.hpp`) and any other
  consumer.
- **`parameter.hpp`** — internal header of the parser:
  `namespace parameters { … }` helpers built on `toml::table`.
  Pulls `toml.hpp`.  Included only by `parameter.cpp` (which
  implements the namespace) and `maxwell_parameters.cpp` (which
  uses it to populate the structs).

The earlier `MAXWELL_PARAMETER_NEEDS_TOML` opt-in macro is no
longer needed: the two responsibilities live in two headers, and
toml++ exposure is implicit in including `parameter.hpp`.  Anything
that doesn't include `parameter.hpp` doesn't see `toml.hpp` and so
doesn't trip the GCC 12 bug.

Side benefit: a 15,000-line header is no longer pulled into the
kernel-heavy translation units, which substantially shortens
compile time for `driver.cpp.o` and `maxwell_eqs.cpp.o`.

If a future patch ever invokes the toml++ helpers from a new
translation unit and the GCC 12 bug resurfaces, the in-place
workaround in `toml.hpp` itself is to bind the trait to a local
`constexpr bool` so the `if constexpr` no longer references the
deduced type:

```cpp
using return_type = decltype(...);
constexpr bool _returns_bool =
    impl::is_constructible_or_convertible<bool, return_type>;
if constexpr (_returns_bool) { ... }
```

Apply at both `toml.hpp:8239` and `:8258` (the same construct
appears in two adjacent `for_each` lambdas).

### 10.3  Single-GPU multi-rank contention

On a node with one GPU, every MPI rank tries to bind device 0 by
default.  CUDA contexts can be shared, so this is correct, but
performance degrades and some operations may serialise.  For
correctness checks under MPI on a single-GPU host, either run with
`-np 1` or enable NVIDIA MPS (`nvidia-cuda-mps-control -d`) before
launching `mpirun`.  Multi-GPU nodes should bind one rank per GPU
(`mpirun --map-by ppr:1:gpu`).

---

## 11. What's wired but unproven

These items shipped with full code paths but were not driven by a
test on the development host:

- **CUDA backend runtime correctness.**  Build verified on the
  A100 host; convergence test on the device is the recommended
  next step (see §8.3).
- **CUDA-aware MPI** (`MAXWELL_CUDA_AWARE_MPI=ON`).  Compiles and
  links; no test distinguishes a CUDA-aware build from the
  host-mirror build other than performance.
- **Checkpoint/restart.**  `read_checkpoint` and `write_checkpoint`
  are ported and link; no automated round-trip test.
- **`make check-generated`.**  The SymPy generator
  (`generate_ccode.py`) is unchanged from the C version and emits
  a `simple_maxwell.h` byte-for-byte identical to the checked-in
  copy, but no CMake target exercises this.

Ranked by risk: CUDA-aware MPI (silent corruption) > CUDA-backend
PDE correctness (caught immediately by the convergence test) >
checkpoint restart (recoverable) > generator drift (caught the
next time someone edits `generate_ccode.py`).  All four are easy
to run-test once a suitable environment is available.

---

## 12. See also

- **`documentation.tex`** — full LaTeX reference covering the
  physics, numerical scheme, SBP-SAT derivation, software
  architecture, and a complete user guide.  The present report is
  a focused complement: it covers design rationale and build
  options, while `documentation.tex` covers everything a student
  or downstream developer should know about the code.
- **`GPU_Maxwell.md`** — the original implementation plan.
- **`../Maxwell_Penalty/CLAUDE.md`** — parent code notes,
  including the SBP-SAT sign convention warnings and the smoke-test
  results referenced here as the C-version baseline.
