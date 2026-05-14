# Plan: flexible boundary conditions and problem presets

This plan covers both items currently open in `Plan.md`:

- **§6.1** — RHS and `u_exact` are hard-coded in
  `src/driver_multigrid.c:125-185`; replace with function pointers so
  a different problem can be selected at run time without editing the
  driver.
- **§6.2** — `apply_bc_3d` only enforces homogeneous Dirichlet; extend
  it to inhomogeneous Dirichlet, homogeneous Neumann, and inhomogeneous
  Neumann, on each face independently.

The two items are tightly coupled: a flexible BC interface is only
useful if the corresponding RHS, exact solution, and BC values can also
vary, and the cleanest way to package them together is a single
"problem preset" record (see §3.2 below).  Landing them as one effort
keeps the configuration surface coherent.

## 1. Goals and scope

### 1.1 Boundary conditions (§6.2)

Extend the multigrid solver to support, on each of the six physical
faces of $\Omega = [0,1]^3$ independently, any of the following:

| Kind                    | Tag in code     | Boundary equation                                     |
|-------------------------|-----------------|-------------------------------------------------------|
| Homogeneous Dirichlet   | `BC_DIRICHLET`+`homogeneous=true`   | $u = 0$                                |
| Inhomogeneous Dirichlet | `BC_DIRICHLET`+`homogeneous=false`  | $u = g(\mathbf{x})$                    |
| Homogeneous Neumann     | `BC_NEUMANN`+`homogeneous=true`     | $\partial u/\partial n = 0$            |
| Inhomogeneous Neumann   | `BC_NEUMANN`+`homogeneous=false`    | $\partial u/\partial n = q(\mathbf{x})$ |

Each face carries a `kind` (Dirichlet or Neumann) plus a `homogeneous`
flag.  When the flag is `false`, an associated boundary-data callback
$g$ or $q$ supplies the per-face values.  The architecture is forward-
compatible with Robin and periodic BCs (see §10) but those are not
implemented in the first pass.

**Hard invariants:**

1. Each face is independently configurable. Mixed BCs (e.g. Dirichlet on
   $x$ faces, Neumann on $y$ faces) are first-class citizens.
2. **Coarse grids always carry the homogeneous variant of the parent's
   BC kind.**  The unknown on a coarse level is the correction $e_H$,
   whose boundary condition is the homogenised version of the original
   ($e=0$ on Dirichlet faces, $\partial e/\partial n = 0$ on Neumann
   faces) regardless of the inhomogeneous data the user supplied at the
   fine level.  Failing to enforce this introduces a spurious source on
   every V-cycle step.
3. The configuration **must reject the all-Neumann case** unless the
   user has selected a regularisation strategy (mean-zero projection;
   see §6).  Pure Neumann makes the discrete operator singular up to a
   constant, and silent V-cycle drift is the worst possible failure mode
   for a teaching code.

### 1.2 Problem presets (§6.1)

Today, `src/driver_multigrid.c` hard-codes:

- the right-hand side initialisation loop at lines 125–139,
  $f(\mathbf{x}) = -3\pi^2 \sin(\pi x)\sin(\pi y)\sin(\pi z)$;
- the exact-solution evaluation block at lines 168–185,
  $u^*(\mathbf{x}) = \sin(\pi x)\sin(\pi y)\sin(\pi z)$.

Replace both with pluggable function pointers ("`scalar_field_fn`")
bundled into a `problem_t` record alongside the `bc_spec_t`.  Each
record is a complete problem definition: BC kinds + values, RHS
function, optional exact-solution function, and a `singular` flag for
the all-Neumann case.

The driver registers a small static table of presets and selects one
at startup via the TOML.  Adding a new problem becomes a one-function
edit-and-rebuild; the V-cycle, smoother, transfer operators, and BC
plumbing are all problem-agnostic.

**Hard invariants for the preset machinery:**

4. The default preset reproduces today's behaviour bit-for-bit.  Every
   existing operator-level test and the existing convergence test
   continue to pass with no changes other than (optionally) selecting
   the new default preset name in their TOML.
5. Each preset is self-consistent: its `bc_spec_t`, `rhs`, and
   `u_exact` describe the same continuous problem.  Mismatched
   bundles (e.g. a Dirichlet BC face with an exact solution that
   doesn't satisfy it) are a static authoring error, not a runtime
   one — there is no auto-derivation.

## 2. Mathematical background

### 2.1 Discretisation per BC kind

Throughout, the centred 7-point Laplacian is preserved as the interior
operator.  Boundary handling is done by per-face stencil modification.

#### Dirichlet (constraint)
The boundary node value is *prescribed*, not solved for.  The
smoother does not update boundary nodes; the defect is identically zero
there.  For inhomogeneous Dirichlet, the boundary nodes are written
once at solver start with $u = g(\mathbf{x})$ and never modified again.

#### Neumann (ghost-row reflection)
The boundary node *is* solved for.  The 7-point stencil at the boundary
node (e.g. $i = 0$ at the lower-$x$ face) reaches outside the domain to
$u_{-1,j,k}$, which is treated as a one-cell-wide ghost row and is
chosen so the centred difference reproduces the prescribed normal
derivative:

$$
\frac{u_{1,j,k} - u_{-1,j,k}}{2 h_x} \;=\; \partial u/\partial x|_{i=0}.
$$

With outward normal $\hat{\mathbf{n}} = -\hat{\mathbf{x}}$ at the
lower-$x$ face, $\partial u/\partial n = -\partial u/\partial x$, so

$$
u_{-1,j,k} \;=\; u_{1,j,k} + 2 h_x \, q(\mathbf{x}_{0,j,k}).
$$

Substituting into the standard 7-point stencil at $i=0$ and grouping
terms gives the boundary equation

$$
\frac{2(u_{1,j,k} - u_{0,j,k})}{h_x^2} + \frac{2 q}{h_x}
\;+\; (\text{$y$- and $z$-second differences}) \;=\; f_{0,j,k}.
$$

This is the form the smoother and the defect kernel must produce when
they sweep over the Neumann boundary node.

The five other faces are symmetric, with appropriate sign on $q$
depending on the outward normal.

### 2.2 Solvability

For the pure-Neumann problem $\Delta u = f$, $\partial u/\partial n = q$
on $\partial\Omega$, the divergence theorem gives the compatibility
condition

$$
\int_\Omega f \, d\mathbf{x} \;=\; \int_{\partial\Omega} q \, dS.
$$

The discrete analogue must hold to within discretisation error;
otherwise the linear system has no solution.  The solution is then
unique only up to an additive constant — the constant function is in
the null space of $\Delta_h$ with all-Neumann BCs.

The mixed case (at least one Dirichlet face) is non-singular and needs
no special treatment.

## 3. Architecture

### 3.1 Per-face BC specification

New header `src/bc.h`:

```c
typedef enum {
    BC_DIRICHLET = 0,
    BC_NEUMANN   = 1,
    /* room for future kinds: BC_ROBIN, BC_PERIODIC */
} bc_kind_t;

/* Boundary value/flux callback.  For Dirichlet, returns u; for Neumann,
 * returns du/dn (outward-normal derivative).  The `face` argument lets
 * a single callback dispatch per-face logic if convenient. */
typedef enum {
    FACE_LOWER_X = 0, FACE_UPPER_X,
    FACE_LOWER_Y,     FACE_UPPER_Y,
    FACE_LOWER_Z,     FACE_UPPER_Z,
    NUM_FACES = 6,
} face_id_t;

typedef double (*bc_fn_t)(double x, double y, double z, face_id_t face);

struct bc_face_t {
    bc_kind_t kind;
    bool      homogeneous;
    bc_fn_t   value;   /* may be NULL when homogeneous == true */
};

struct bc_spec_t {
    struct bc_face_t face[NUM_FACES];
};

/* Construct the homogeneous variant: same kind on each face, but
 * value = NULL and homogeneous = true.  Used by the hierarchy
 * constructor to seed coarse-level BCs. */
void bc_spec_homogenize(const struct bc_spec_t *src, struct bc_spec_t *dst);
```

A pointer is added to `struct ngfs_3d`:

```c
struct ngfs_3d {
    /* ... existing fields ... */
    struct bc_spec_t *bc;   /* owned by this struct; freed in
                             * ngfs_3d_deallocate.  NULL means
                             * "homogeneous Dirichlet on all faces"
                             * (back-compat for code that doesn't set
                             * BCs explicitly, e.g. unit tests). */
};
```

### 3.2 Function-pointer dispatch (problem presets) — closes §6.1

A "problem preset" is a static record bundling every problem-specific
callback: BC spec, RHS, and (optionally) an exact solution for use in
manufactured-solution tests.  This is the single mechanism that closes
both `Plan.md` §6.1 (RHS/exact callbacks) and §6.2 (BC dispatch).

#### Header `src/problem.h`

```c
#include "bc.h"

/* Scalar field on the unit cube.  Used both for the RHS f(x,y,z) and
 * for the exact solution u_exact(x,y,z) when one is available. */
typedef double (*scalar_field_fn)(double x, double y, double z);

struct problem_t {
    const char        *name;
    struct bc_spec_t   bc;        /* per-face BC kinds + value callbacks */
    scalar_field_fn    rhs;       /* f(x,y,z); must be non-NULL */
    scalar_field_fn    u_exact;   /* optional; NULL if no closed form */
    bool               singular;  /* true => mean-zero projection on the
                                   * V-cycle, plus a compatibility check
                                   * at configure time.  Required for
                                   * all-Neumann presets. */
};

/* Registry of built-in presets; terminated by .name = NULL.  Defined
 * in problem_registry.c, populated by the driver. */
extern const struct problem_t g_problems[];

/* Look up a preset by name.  Returns NULL on miss; the caller is
 * expected to treat that as a parser error. */
const struct problem_t *problem_lookup(const char *name);
```

#### Header layering and ownership

- `src/bc.h` — POD types only (`bc_kind_t`, `bc_face_t`, `bc_spec_t`,
  `face_id_t`, `bc_fn_t`).
- `src/problem.h` — `problem_t` and the registry API; depends on
  `bc.h`.
- `src/problem_registry.c` — defines `g_problems[]` and the
  per-preset RHS / exact-solution / BC-value implementations.  This is
  the **one** file a user edits to add a new problem: write a few
  pure functions, append a `problem_t` record to the table, rebuild.
- `src/driver_multigrid.c` — calls `problem_lookup(name)` at
  startup, hands the result to the new initialisation helpers
  (§4.7), and never sees `sin(pi x)` etc. directly.

#### Built-in presets

| name (TOML)                        | BC config                                    | $u^*$ | RHS |
|------------------------------------|----------------------------------------------|-------|-----|
| `manufactured_dirichlet_homog`     | All six faces Dirichlet, homogeneous         | $\sin(\pi x)\sin(\pi y)\sin(\pi z)$ (current default) | $-3\pi^2 u^*$ |
| `manufactured_dirichlet_inhomog`   | All six faces Dirichlet, inhomogeneous       | $\cos(\pi x)\cos(\pi y)\cos(\pi z)$ | $-3\pi^2 u^*$ |
| `manufactured_neumann_homog`       | All six faces Neumann, homogeneous; singular | $\cos(\pi x)\cos(\pi y)\cos(\pi z)$ | $-3\pi^2 u^*$ |
| `manufactured_neumann_inhomog`     | All six faces Neumann, inhomogeneous; singular | $\sin(\pi x/2)\sin(\pi y/2)\sin(\pi z/2)$ (subject to compat. check) | $-3(\pi/2)^2 u^*$ |
| `manufactured_mixed`               | Dirichlet (homog.) on lower faces, Neumann (inhomog.) on upper | $u^* = x^2 y^2 z^2$ | $2(y^2 z^2 + x^2 z^2 + x^2 y^2)$ |

`manufactured_dirichlet_homog` is the existing test problem and stays
the default for back-compat; absence of `[problem]` in the TOML
resolves to it.

### 3.3 Coarse-level homogenisation

`ngfs_3d_create_child` allocates a fresh `bc_spec_t` for the child via
`bc_spec_homogenize(parent->bc, child->bc)`.  This forces every face to
`homogeneous = true` while preserving the `kind`.  `ngfs_3d_deallocate`
frees `bc`.

The invariant: after `ngfs_3d_create_hierarchy`, level 0 carries the
user's spec, every deeper level carries its homogenised counterpart.

## 4. Per-routine changes

### 4.1 `apply_bc_3d` (`src/gauss_seidel.c:21`)

Today: writes 0 to every physical-boundary face of the named variable.

After: per face, behaviour depends on `gfs->bc->face[f]`:

| Kind & variable     | What `apply_bc_3d` does                                                           |
|---------------------|-----------------------------------------------------------------------------------|
| Dirichlet, `VAR_SOL` | Write $g(\mathbf{x})$ (or 0 if homogeneous) to the boundary face nodes.          |
| Dirichlet, `VAR_DEF` | Write 0 to the boundary face nodes.  (Defect on a Dirichlet boundary is 0.)      |
| Neumann, `VAR_SOL`   | **Do not touch the face nodes.**  Instead, write the ghost row using $u_{ghost} = u_{interior} + 2 h_n \cdot q$ (or 0 if homogeneous). |
| Neumann, `VAR_DEF`   | Same ghost-row treatment as `VAR_SOL` but with $q = 0$ (the defect equation has homogeneous Neumann data; only $u$ carries the user's $q$). |

The "ghost row" of a physical-boundary face is the local cell at index
$i = 0$ (or $j$, $k$) — the same cell that for an MPI-interior face
holds a copy from a neighbour rank.  When the rank owns a physical
boundary that cell is unused by the existing code, so writing into it
is safe.

### 4.2 `gauss_seidel_3d` (`src/gauss_seidel.c:86`)

Today: sweeps over $[gs, n-gs)$ in each axis.

After: sweep range depends on the BC kind on the face that bounds that
axis.  The sweep is widened by 1 on each face whose kind is Neumann
(the boundary node is an unknown).  A small helper computes per-axis
loop bounds:

```c
static inline void gs_loop_bounds_3d(
    const struct ngfs_3d *gfs,
    int64_t *i_lo, int64_t *i_hi,
    int64_t *j_lo, int64_t *j_hi,
    int64_t *k_lo, int64_t *k_hi)
{
    *i_lo = gfs->gs;
    *i_hi = gfs->nx - gfs->gs;
    /* if this rank owns the lower-x physical boundary AND that face is
     * Neumann, the boundary node is solved for: include i=0 in the sweep. */
    if (gfs->domain.lower_x_rank == INVALID_RANK
            && gfs->bc->face[FACE_LOWER_X].kind == BC_NEUMANN)
        *i_lo = 0;
    /* ... symmetric for the other 5 faces ... */
}
```

The inner stencil at the boundary node is unchanged from the interior
form — the ghost-row write done by `apply_bc_3d` makes the centred
difference produce the correct boundary equation automatically.  This
is the central invariant that keeps the smoother kernel branch-free in
the hot loop.

The red-black colouring is unaffected: the boundary node's parity is
just $(g_i + g_j + g_k) \bmod 2$ as before.

### 4.3 `calc_defect_3d` (`src/gauss_seidel.c:196`)

Identical loop-range adjustment as `gauss_seidel_3d`.  After the loop,
`apply_bc_3d(gfs, VAR_DEF)` is called as today; the new `apply_bc_3d`
correctly leaves Neumann boundary defects untouched (and writes the
ghost row needed for `restrict_var_3d` to pick them up).

### 4.4 `prolong_var_3d` (`src/multigrid.c:559`)

Today: skips fine-grid physical-boundary nodes via `if (pg_x == 0 || pg_x == gni - 1) continue`.

After: skip only on Dirichlet faces.  On Neumann faces the boundary
node is an unknown and the correction is non-zero there, so it must be
prolonged.

```c
const bool skip_lower_x = (gfs->bc->face[FACE_LOWER_X].kind == BC_DIRICHLET);
const bool skip_upper_x = (gfs->bc->face[FACE_UPPER_X].kind == BC_DIRICHLET);
/* ... etc ... */

for (...) {
    if (skip_lower_x && pg_x == 0) continue;
    if (skip_upper_x && pg_x == gni - 1) continue;
    /* ... */
}
```

Note that on coarse levels the BC is always homogeneous but the
\emph{kind} carries through, so this check still does the right thing.

### 4.5 Restriction / injection

No changes needed.  `inject_var_3d` already copies the fine boundary
defect to the coarse boundary; for Dirichlet faces this value is 0
(set by `apply_bc_3d` for `VAR_DEF`), for Neumann faces it carries the
true boundary defect.

### 4.6 Hierarchy construction

`ngfs_3d_create_child` allocates `child->bc` and homogenises from
`parent->bc`.  `ngfs_3d_deallocate` frees `child->bc` if non-NULL.

### 4.7 Driver-loop changes — closes §6.1

`src/driver_multigrid.c` lines 125–139 (RHS init) and lines 168–185
(error vs. exact) are replaced with calls into per-preset helpers.
Concrete signatures:

```c
/* Initialise VAR_RHS at every grid point from problem->rhs(x,y,z).
 * Initialises VAR_SOL = 0.  If singular, also subtracts the discrete
 * mean of VAR_RHS so the compatibility condition is satisfied. */
void problem_initialise_rhs(struct ngfs_3d *gfs,
                            const struct problem_t *problem);

/* Initialise the boundary nodes of VAR_SOL from the per-face BC
 * callbacks.  This is the only place inhomogeneous Dirichlet data
 * enters the fine-grid solution; coarse levels never call it. */
void problem_apply_initial_bc(struct ngfs_3d *gfs,
                              const struct problem_t *problem);

/* If problem->u_exact is non-NULL, compute |u_h - u_exact|_inf over
 * the local interior and reduce across the Cartesian communicator
 * with MPI_MAX.  Returns -1.0 (and prints nothing) if u_exact == NULL.
 * The driver prints the result; the convergence test parses it. */
double problem_compute_max_error(struct ngfs_3d *gfs,
                                 const struct problem_t *problem);
```

`main()` becomes:

```c
struct param_st param;
parse_parameter_file(&param, argv[1]);

const struct problem_t *problem = problem_lookup(param.problem_name);
if (!problem) {
    fprintf(stderr, "unknown preset '%s'\n", param.problem_name);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

/* ... topology and root-grid allocation as today ... */

/* Stamp the user's BC spec onto the root level; the hierarchy
 * constructor homogenises it for every coarse level. */
gfs.bc = malloc(sizeof(struct bc_spec_t));
*gfs.bc = problem->bc;

if (param.use_multigrid)
    ngfs_3d_create_hierarchy(&gfs, param.min_cells);

problem_initialise_rhs    (&gfs, problem);
problem_apply_initial_bc  (&gfs, problem);

/* ... outer iteration loop unchanged ... */

const double err = problem_compute_max_error(&gfs, problem);
if (mpi_rank == 0 && err >= 0.0)
    printf("\n|u - u_exact|_inf = %12.6e\n", err);
```

The driver no longer references `sin`, `cos`, `pi`, or any specific
manufactured solution.  Adding a new problem is now: write three
small functions in `problem_registry.c`, append a `problem_t` to
`g_problems[]`, rebuild.

## 5. TOML schema

Add a `[problem]` section that selects a named preset:

```toml
[problem]
name = "manufactured_dirichlet_inhomog"
```

The driver looks the name up in `g_problems[]` and uses the bundled
`bc_spec_t`, RHS, and exact-solution callback.

The existing `[grid]` and `[solver]` sections are unchanged.  All BC
plumbing is hidden behind the preset name; users who want custom
problems edit `g_problems[]` in the driver and rebuild — the same
edit-and-rebuild workflow we agreed on for §6.1.

The parser allowlist (`Plan.md` §4.2, now landed) is extended: the
new section `problem` is recognised, with the single key `name`.  An
unknown preset name (one not in `g_problems[]`) is reported as a
parser error.

**Optional section.**  The `[problem]` section is optional — when it
is absent, `param.problem_name` defaults to
`"manufactured_dirichlet_homog"`, which reproduces today's hard-coded
behaviour exactly.  This is the back-compat invariant that lets every
pre-existing TOML file (including `multigrid.toml` and the
auto-generated TOMLs in `run_test_convergence.sh` and
`run_test_parser.sh`) continue to work without modification.  The
parser logic is therefore: if `[problem]` is present, use the
unknown-key validator on it (rejecting typos); if absent, skip the
validator and keep the default name.

## 6. The singular case

When all six faces are Neumann (`g_problems[i].singular == true`), the
following must happen:

### 6.1 Configuration-time check

In the driver's preset-resolution code:

```c
if (problem->singular) {
    /* Compatibility check on the manufactured RHS: the discrete
     * volume integral of f plus the surface integral of -q must
     * vanish to within discretisation error. */
    double cfd = compute_compatibility_residual(problem, &gfs);
    if (fabs(cfd) > some_threshold * h^2)
        warning_or_abort("compatibility violated: ...");
}
```

This guards against typos in the manufactured solution.

### 6.2 Mean-zero projection during the V-cycle

After every smoothing step at the root level, project `VAR_SOL` onto
the mean-zero subspace:

```c
double mean = global_mean(gfs, VAR_SOL);
for every interior point p:
    u[p] -= mean;
```

Symmetric projection of the right-hand side at solver start ensures
$\sum_i f_i \approx 0$, which is the discrete compatibility condition.

This is the simplest workable scheme.  An alternative (anchoring one
node) is asymmetric and trickier to get right under MPI decomposition.

### 6.3 Convergence-test threshold

For the all-Neumann test, the convergence rate is unaffected by the
projection (it only resolves the constant null-space component).  The
existing rate-of-2 assertion in `verify_convergence.py` continues to
work; the only change is that the test case sets up the problem with
projection enabled.

## 7. Test plan

### 7.1 Test cases

Each new convergence test runs the driver at three resolutions
($32^3$, $64^3$, $128^3$) on np=1 and np=8, parses
`|u - u_exact|_inf`, and asserts rate $\in [1.8, 2.3]$ on the finest
pair — exactly the existing pattern from `run_test_convergence.sh`.

| Test                  | Preset                            | $u^*$                                | Why it's a useful test |
|-----------------------|-----------------------------------|--------------------------------------|------------------------|
| `convergence_dirichlet_homog` | `manufactured_dirichlet_homog` | $\sin(\pi x)\sin(\pi y)\sin(\pi z)$ | Existing test (rename only). Regression baseline. |
| `convergence_dirichlet_inhomog` | `manufactured_dirichlet_inhomog` | $\cos(\pi x)\cos(\pi y)\cos(\pi z)$ | Tests inhomogeneous $g$ application; non-zero on every face. |
| `convergence_neumann_homog`     | `manufactured_neumann_homog`     | $\cos(\pi x)\cos(\pi y)\cos(\pi z)$ | All-Neumann (singular) with $q = 0$.  Exercises the projection.  $u^*$ is mean-zero so the projected solution matches it. |
| `convergence_neumann_inhomog`   | `manufactured_neumann_inhomog`   | TBD (smooth, non-zero normal derivatives on at least one face) | Non-zero $q$. |
| `convergence_mixed`             | `manufactured_mixed`             | $u^* = x^2 y^2 z^2$                  | Dirichlet on lower-$x$/-$y$/-$z$ (homogeneous, $u^* = 0$) and Neumann on upper-$x$/-$y$/-$z$ (inhomogeneous, $\partial u^*/\partial n = 2 \cdot \cdot \cdot$). |

#### Notes on the manufactured solutions

- **Dirichlet inhomogeneous.**  $u^* = \cos(\pi x)\cos(\pi y)\cos(\pi z)$
  has $u^*(0,y,z) = \cos(\pi y)\cos(\pi z) \neq 0$ in general.  The
  Dirichlet data is therefore a real test of the inhomogeneous code
  path.  $f^* = -3\pi^2 u^*$.

- **Neumann homogeneous.**  Same $u^*$: $\partial u^*/\partial x =
  -\pi \sin(\pi x)\cos(\pi y)\cos(\pi z)$, which vanishes at $x=0$ and
  $x=1$.  Symmetrically for $y$ and $z$.  Volume integral
  $\int_\Omega f^* \, d\mathbf{x} = -3\pi^2 \int (\cos\pi x)^3
  \,d\mathbf{x} = 0$, so the compatibility condition is satisfied
  exactly in the continuous problem (and to order $h^2$ on the
  discrete mesh).

- **Neumann inhomogeneous.**  Pick $u^* = \sin(\pi x/2)\sin(\pi y/2)\sin(\pi z/2)$.
  At $x = 0$ the normal derivative
  $-\partial u^*/\partial x = -(\pi/2)\cos(0) \cdot \cdot = -(\pi/2) \sin(\pi y/2)\sin(\pi z/2)$
  is non-zero generically.  At $x = 1$ the normal derivative
  $\partial u^*/\partial x = (\pi/2)\cos(\pi/2)\cdot\cdot = 0$ — only
  three of the six faces carry inhomogeneous data, but that is
  sufficient as a test.  $f^* = -3(\pi/2)^2 u^*$.  Compatibility must
  be verified explicitly.

- **Mixed.**  $u^* = x^2 y^2 z^2$ vanishes on $x = 0$, $y = 0$, $z = 0$
  (homogeneous Dirichlet on the three lower faces).  On the upper
  faces, $\partial u^*/\partial x|_{x=1} = 2 y^2 z^2$,
  $\partial u^*/\partial y|_{y=1} = 2 x^2 z^2$,
  $\partial u^*/\partial z|_{z=1} = 2 x^2 y^2$ — inhomogeneous Neumann.
  $f^* = 2(y^2 z^2 + x^2 z^2 + x^2 y^2)$.  Non-singular, so no
  projection.

### 7.2 Test infrastructure

The current `run_test_convergence.sh` hard-codes the manufactured-
solution preset.  Generalise it:

```bash
# tests/run_test_convergence.sh
# Usage (called by CMake as one CTest entry per preset):
#     run_test_convergence.sh <preset_name>
PRESET="${1:-manufactured_dirichlet_homog}"
# generate TOML with [problem] name = $PRESET
# run at three resolutions, np=1 and np=8
# python3 verify_convergence.py <logs>
```

`tests/CMakeLists.txt` adds one CTest entry per preset:

```cmake
foreach(preset
    manufactured_dirichlet_homog
    manufactured_dirichlet_inhomog
    manufactured_neumann_homog
    manufactured_neumann_inhomog
    manufactured_mixed)
    add_test(NAME convergence_${preset}
        COMMAND bash run_test_convergence.sh ${preset}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(convergence_${preset}
        PROPERTIES LABELS end_to_end)
endforeach()
```

`verify_convergence.py` is unchanged — it only consumes the printed
`|u - u_exact|_inf` line.

### 7.3 Operator-level tests

The existing operator-level tests (domain, child, project, prolong,
restrict_nl, prolong_nl) all use the default
"homogeneous Dirichlet on all faces" path.  After the refactor,
`gfs->bc == NULL` continues to map to that default behaviour, so the
existing tests are expected to pass without change.  This is a
deliberate back-compat invariant.

We additionally want operator-level coverage of the Neumann ghost-row
write.  Two new lightweight tests:

- `test_apply_bc_neumann_3d`: build a small grid, set Neumann on one
  face with a non-zero $q$ callback, call `apply_bc_3d`, verify by
  finite-difference that the centred derivative at the boundary node
  matches the prescribed $q$ to round-off.
- `test_prolong_neumann_3d`: a prolongation regression that confirms
  the correction at a Neumann boundary node is non-zero (today's code
  always produces zero there because of the `pg_x == 0 || ...` skip).

## 8. Implementation phasing

The plan is large enough that landing it in one commit is risky.
Four-phase rollout, each with passing CI before moving to the next.
Phase 1 closes `Plan.md` §6.1 in isolation; phases 2–4 then close §6.2
incrementally.

### Phase 1 — Problem-preset scaffolding (closes §6.1; no new BC capability)

- Add `src/bc.h` with the POD type definitions only (`bc_kind_t`,
  `bc_face_t`, `bc_spec_t`, `face_id_t`, `bc_fn_t`).  No behavioural
  hookup yet.
- Add `bc` pointer to `struct ngfs_3d`; allocate/free in
  `ngfs_3d_allocate` / `ngfs_3d_deallocate`; implement
  `bc_spec_homogenize`; plumb through `ngfs_3d_create_child`.  Default
  `bc = NULL` → `apply_bc_3d` and friends keep today's homogeneous-
  Dirichlet behaviour.
- Add `src/problem.h` with `problem_t`, `problem_lookup`, and the
  `g_problems[]` extern.
- Add `src/problem_registry.c` populated with a single preset,
  `manufactured_dirichlet_homog`, that bundles today's RHS, exact
  solution, and BC spec.
- Add `param.problem_name` to `param_st`; default to
  `"manufactured_dirichlet_homog"`; add the `[problem] name = "..."`
  TOML section and update the parser allowlist.
- Implement `problem_initialise_rhs`, `problem_apply_initial_bc`
  (no-op for homogeneous), `problem_compute_max_error`.
- Rewrite `driver_multigrid.c` to use them; remove the inline
  manufactured-solution code at lines 125–139 and 168–185.
- All existing tests pass unchanged: the default preset reproduces
  today's behaviour bit-for-bit.

After Phase 1, `Plan.md` §6.1 is closed.  No new boundary types
exist yet, but the registry machinery is in place.

### Phase 2 — Inhomogeneous Dirichlet (first slice of §6.2)
- Update `apply_bc_3d` to respect `face[f].kind == BC_DIRICHLET` with
  `homogeneous == false`: write $g(\mathbf{x})$ at face nodes.  All
  other faces still use the homogeneous-Dirichlet path.
- Update `problem_apply_initial_bc` to write inhomogeneous Dirichlet
  data on the root level.  (Coarse levels are untouched: hierarchy
  homogenisation already strips the inhomogeneous data.)
- Register `manufactured_dirichlet_inhomog`.
- Add `convergence_dirichlet_inhomog` CTest entry; verify rate-of-2.
- The existing `convergence` CTest is renamed to
  `convergence_dirichlet_homog` and continues to pass.

### Phase 3 — Homogeneous Neumann
- Update `apply_bc_3d` to respect `face[f].kind == BC_NEUMANN` with
  `homogeneous == true`: write the mirror ghost row $u_{ghost} =
  u_{interior}$.
- Add `gs_loop_bounds_3d` helper; update `gauss_seidel_3d` and
  `calc_defect_3d` to use it (loops widen on Neumann faces).
- Update `prolong_var_3d` to skip only Dirichlet faces.
- Implement compatibility check + mean-zero projection for the
  singular case (handles all-Neumann presets).
- Register `manufactured_neumann_homog`; add convergence test.
- Add operator-level test `test_apply_bc_neumann_3d` (centred-
  derivative round-off check at the boundary).

### Phase 4 — Inhomogeneous Neumann + mixed
- Extend `apply_bc_3d` to handle inhomogeneous Neumann: ghost-row
  write $u_{ghost} = u_{interior} \pm 2 h_n q$ with the per-face
  outward-normal sign.
- Register `manufactured_neumann_inhomog` and `manufactured_mixed`;
  add their convergence tests.
- Add operator-level test `test_prolong_neumann_3d` (verifies a
  non-zero correction at a Neumann boundary node).
- Verify the singular-case compatibility tolerance of §9 question 1
  on the inhomogeneous-Neumann manufactured solution; tighten the
  threshold or pick a different $u^*$ if needed.

After Phase 4, `Plan.md` §6.1 and §6.2 are both fully closed.  §5.1
(output directory) is independent and can land before, after, or
alongside any phase.

## 9. Open questions

1. **Compatibility-condition tolerance.**  In §6.1, what threshold for
   "$\sum_i f_i \approx 0$" should trip a warning?  Proposal: $h^2$
   times a small constant, since the trapezoid-rule error is $O(h^2)$.
   Resolved during Phase 3.

2. **Should mean-zero projection be optional?**  A user might want to
   set up a singular problem deliberately and anchor by hand.  Proposal:
   add a `[solver] mean_zero_projection = true|false` TOML key,
   defaulting to `true` for `singular` presets.  Punt to a follow-up.

3. **Robin BCs.**  The architecture above accommodates Robin
   ($\alpha u + \beta \partial u/\partial n = r$) by adding
   `BC_ROBIN` and a third callback for $\alpha$/$\beta$/$r$ per face.
   The boundary equation becomes a modified-diagonal stencil
   (already documented in `doc/documentation.tex` §9.3).  Plan to add
   in a Phase 4 if/when needed.

4. **Periodic BCs.**  More invasive than Dirichlet/Neumann because they
   require flipping `MPI_Cart_create`'s `periods` argument and removing
   the `prolong_var_3d` boundary skip on periodic axes.  Out of scope
   for this plan; tracked in `Plan.md` §6.2 / extension docs.

5. **Manufactured solution for `manufactured_neumann_inhomog`.**
   $u^* = \sin(\pi x/2)\sin(\pi y/2)\sin(\pi z/2)$ is one candidate.
   Verify the discrete compatibility residual is small enough that
   the rate-of-2 assertion holds.  Resolved during Phase 3.

## 10. Risks

- **Loop-bounds bugs at the boundary.**  The widened-sweep logic in
  `gauss_seidel_3d` and `calc_defect_3d` is the most error-prone piece.
  Mitigation: keep the bounds in a single helper (`gs_loop_bounds_3d`),
  test each face independently, and run the convergence test on the
  mixed-BC preset where every face takes a different path.
- **Coarse-level BC drift.**  If `bc_spec_homogenize` is called with
  the wrong sign convention, the V-cycle will diverge silently on
  Neumann problems.  Mitigation: a Phase-3 unit test that constructs a
  two-level hierarchy on a Neumann problem, applies one V-cycle, and
  asserts the defect strictly decreases.
- **Singular-case test flakiness.**  Mean-zero projection introduces an
  extra global reduction per smoothing step.  Combined with floating-
  point rounding, the convergence rate near machine epsilon may dip
  below 1.8.  Mitigation: stop the convergence test at $128^3$ where
  the discretisation error is comfortably above the floor.
