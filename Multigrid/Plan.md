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

#### 3.1 The "extra plain Gauss-Seidel pass" in `vcycle_3d`  *(decision)*

```c
gauss_seidel_3d(gfs, n_smooth, omega);   // post-smoothing
gauss_seidel_3d(gfs, n_smooth, 1.0);     // <-- extra unconditional pass
defect_norm = calc_defect_3d(gfs);
```

Inside the V-cycle subcycle loop the post-smoothing is run twice: once
with the user `omega`, once with `omega = 1.0`.  This doubles smoothing
work per subcycle and is undocumented.  The convergence test in §5.2
will confirm whether removing the second call changes the discretisation
error.

**Action.** Decide intent and either:

- (a) remove the second call;
- (b) gate behind a `polish` field in `param_st` (default false);
- (c) keep it and add a comment explaining the rationale.

Re-run `make tests` afterwards — the convergence test catches any
regression.

#### 3.2 Unconditional V-cycle chatter  *(ergonomics)*

`vcycle_3d` always prints `Starting Vcycle` plus a per-level
`defect = ...` trace via `vcycle_debug`.  Only rank 0 prints, but there
is no quiet mode for benchmark or production runs.

**Fix.** Add a `verbose` field to `param_st` (default false), gate the
prints on it.

---

### 4. Robustness

#### 4.1 `comm.c` rank comparison style  *(robustness, cosmetic)*

`comm.c` uses `if (lower_x_rank > -1)` throughout (`sync_var_2d`,
`exchange_direction`, etc.).  `domain.c` already converts `MPI_PROC_NULL`
to the symbolic `INVALID_RANK = -1`, so the comparison happens to work,
but every other file uses `!= INVALID_RANK`.

**Fix.** Replace each `*_rank > -1` with `*_rank != INVALID_RANK` in
`comm.c`.  No behaviour change.

#### 4.2 Unknown TOML keys silently ignored  *(robustness)*

`get_*_value` looks up specific keys; a user who writes
`mulitgrid = true` (typo) gets the missing-key default (`false`)
without warning, and the wrong solver runs.  Same for `nx_celss`, etc.

**Fix.** After all `get_*_value` calls in `multigrid_parameters.cc`,
walk the parsed `toml::table` and emit a warning for any key in
`[grid]` or `[solver]` that is not in the expected set.  Increment
`parser_error` if you want unknown keys to be fatal.

#### 4.3 `int` fields populated from `int64_t` parser results  *(robustness)*

```c
param->n_smooth  = (int)parameters::get_positive_integer_value(...);
param->n_iters   = (int)parameters::get_positive_integer_value(...);
param->subcycles = (int)parameters::get_positive_integer_value(...);
param->min_cells = (int)parameters::get_positive_integer_value(...);
```

The parser returns `int64_t`; the cast silently truncates to `int`.
For these fields the values are tiny in practice, but the cast is
unchecked and undermines the `int64_t` change in the previous refactor.

**Fix.** Either widen the four fields in `param_st` to `int64_t`, or
add explicit `INT_MAX` range checks in `multigrid_parameters.cc` that
fail parsing if any value would overflow.

(4.2 and 4.3 can land as a single "parser hardening" commit.)

#### 4.4 `-ffast-math` in `CFLAGS`  *(robustness, advisory)*

`-ffast-math` allows the compiler to assume no NaNs/Infs and to
reassociate floating-point operations.  Today's code is fine; the flag
becomes a footgun if anyone later adds Kahan summation, NaN tests,
signed-zero handling, or tightly-tolerant float comparisons.

**Action.** Document the use of `-ffast-math` in the `Makefile` (one
comment line) and consider dropping it.  The speed-up on this
stencil-bound code is small.

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

### 6. Extensibility scaffolding

The convergence test (§5.2) already proves that the solver as shipped
is second-order on the manufactured Poisson problem.  These items make
*using* the solver for new problems easier without touching hot-path
code.

#### 6.1 RHS / exact-solution callbacks  *(extension)*

The RHS and exact-solution loops are hard-coded in
`src/driver_multigrid.c:125-185`.  Replace with function pointers:

```c
typedef double (*scalar_field_fn)(double x, double y, double z);

void initialise_rhs(struct ngfs_3d *gfs, scalar_field_fn f);
double compute_max_error(struct ngfs_3d *gfs, scalar_field_fn u_exact);
```

Default the function pointers to the manufactured-solution choices.
A student wanting to try a different $f$ writes one C function and
re-runs.

#### 6.2 BC dispatch  *(extension)*

`apply_bc_3d` writes 0 on every physical-boundary face.  The
documentation (`doc/documentation.tex` §8) explains how to extend to
inhomogeneous Dirichlet / Neumann / Robin conditions but the code has
no scaffolding.  First step: factor `apply_bc_3d` per face, taking a
`bc_kind` and a callback per face.  More invasive than 6.1; defer
until needed.

---

## Suggested order of attack

1. **3.1** — short patch, but wants a deliberate decision; convergence
   test catches any regression for free.
2. **4.2 + 4.3** — one parser-hardening commit.
3. **6.1** — natural enabler if the code is going to be used for
   student exercises.
4. **3.2**, **5.1** — quality-of-life on big runs.
5. **4.1**, **4.4**, **6.2** — defer until they bite.
