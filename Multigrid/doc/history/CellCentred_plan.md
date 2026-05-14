# Plan: cell-centred Neumann boundaries

## 1. Goals and motivation

This refactor delivers two coupled changes.

### 1.1 User-specified Cartesian domain and per-axis cell counts

The current code is hard-coded to the unit cube $\Omega = [0,1]^3$
(the driver passes `global_x0 = 0.0` and `dx = 1.0/(global_nx - 1)`
to `setup_3d_domain`).  The refactor makes both the per-axis bounds
$(a_a, b_a)$ and the per-axis cell count $N_a$ user-specified via the
TOML, so a user can solve on, for example,
$\Omega = [-1, 1] \times [0, 2] \times [0, 1]$ with
$N_x = 64$, $N_y = 128$, $N_z = 32$ (and consequently
$h_x \neq h_y \neq h_z$).  The kernels already track $h_x$, $h_y$,
$h_z$ as separate quantities; the work is in the parser, the driver,
and the user-facing callback contracts.

### 1.2 Cell-centred Neumann boundaries

Refactor the discretisation so that, **per axis**, the grid layout is
chosen to match the boundary condition kind:

- **Dirichlet boundary** → vertex-centred: a grid point lies *exactly
  on* the physical boundary and stores the prescribed value.
- **Neumann boundary** → cell-centred: a grid point lies one half-cell
  *outside* the physical boundary; the centred difference between this
  ghost point and its in-domain neighbour spans exactly one cell, and
  reproduces the prescribed normal derivative to second order.

The two changes are coupled because the layout is naturally specified
relative to the user's box bounds — once the bounds are user-data,
"the boundary is at $x_a = a_a$" becomes the natural place to write
the layout description, rather than the implicit $x_a = 0$.

Adopting this hybrid layout fixes two problems with the current
all-vertex node-centred code:

1. The singular all-Neumann inhomogeneous case has a discrete
   compatibility condition $\sum f_h = 2 \sum_{\partial\Omega} q/h$
   that doesn't match the continuous $\int f = \int q$.  See the
   "Known limitations" entry in `Plan.md` for the current workaround
   (the `manufactured_neumann_inhomog` preset is excluded from CTest).
   With cell-centred Neumann, the discrete compatibility recovers
   $\sum f_h \cdot h^d = \sum q \cdot h^{d-1} \approx \int f = \int q$
   --- the half-cell weight on the boundary cell row exactly removes
   the factor-of-two.

2. The smoother kernel currently widens its sweep on every Neumann
   face and substitutes a ghost mirror inline for off-domain
   neighbours.  In the cell-centred layout the ghost is a *real,
   stored* grid point that `apply_bc_3d` writes; the smoother sweep
   then runs over fixed bounds with the standard 7-point stencil and
   no per-stencil branching.  The kernel becomes simpler and faster.

The refactor is large.  This document plans the changes phase by
phase, with the same back-compat invariant as the boundary plan: the
default preset (`manufactured_dirichlet_homog`) and every existing
operator-level test must continue to produce bit-identical numerical
output after the refactor lands.

## 2. Layout specification

### 2.1 Domain bounds and per-axis spacing

The user specifies a Cartesian box $\Omega = [a_x, b_x] \times [a_y, b_y]
\times [a_z, b_z]$ via the TOML, with independent cell counts
$N_x, N_y, N_z$.  The current code is hard-coded to
$\Omega = [0,1]^3$ (the driver assigns
`global_x0 = global_y0 = global_z0 = 0.0` and
`dx = 1.0 / (global_nx - 1)`); the refactor lifts that assumption.

Per-axis grid spacing:
$$
h_a \;=\; \frac{b_a - a_a}{N_a},\qquad a \in \{x,y,z\}.
$$

The three axes are completely independent: $N_x \neq N_y$ and
$h_x \neq h_y$ are first-class configurations, with no special-casing
required in the kernels (the smoother coefficients in
`gauss_seidel_3d` are already computed per level from $h_x$, $h_y$,
$h_z$ as separate quantities).

### 2.2 Per-axis layout under the BC choice

Let $a_a$, $b_a$ be the user's domain bounds and $h_a = (b_a - a_a)/N_a$
the spacing.  The grid layout along axis $a$ depends on the BC pair
(`face[FACE_LOWER_a].kind`, `face[FACE_UPPER_a].kind`):

| Lower BC | Upper BC | Lowest grid point $x_0$ | Highest grid point $x_{M-1}$ | Total count $M$ | Spacing |
|----------|----------|--------------------------|------------------------------|-----------------|---------|
| **D**    | **D**    | $a_a$                   | $b_a$                       | $N_a + 1$       | uniform $h_a$ throughout |
| **N**    | **N**    | $a_a - h_a/2$           | $b_a + h_a/2$               | $N_a + 2$       | uniform $h_a$ throughout |
| **D**    | **N**    | $a_a$                   | $b_a + h_a/2$               | $N_a + 2$       | uniform $h_a$ between consecutive interior pairs; **half-step $h_a/2$ at the Dirichlet vertex** (between $a_a$ and $a_a + h_a/2$) |
| **N**    | **D**    | $a_a - h_a/2$           | $b_a$                       | $N_a + 2$       | uniform $h_a$ between consecutive interior pairs; **half-step $h_a/2$ at the Dirichlet vertex** (between $b_a - h_a/2$ and $b_a$) |

$N_a$ is the user-facing count of physical cells of width $h_a$ inside
$[a_a, b_a]$.  The layout adds zero, one, or two **ghost cells**
outside the physical box on Neumann sides.

#### The half-step on hybrid axes is unavoidable

For a hybrid axis (D on one end, N on the other), three desirable
properties cannot all hold simultaneously:

1. $N_a$ is the integer cell count inside $[a_a, b_a]$, with
   $h_a = (b_a - a_a) / N_a$.
2. The Dirichlet boundary lies *exactly on* a grid point.
3. The Neumann boundary lies *exactly halfway between* two grid points.
4. Grid spacing is uniform throughout.

Properties (2) and (3) together force the grid to extend a half-cell
beyond the box on the N side (placing the Neumann boundary at the
midpoint of two grid points), while the D vertex coincides with the
box edge.  The N side contributes $N_a + 1/2$ "cell-widths" of grid
between the box edge and the outermost ghost; the D side contributes
the box edge itself.  With $h_a = (b_a - a_a)/N_a$, the total grid
span along the axis is $(b_a - a_a) + h_a/2 = (N_a + 1/2) h_a$, which
cannot be partitioned uniformly into integer multiples of $h_a$.
**One spacing must be $h_a/2$.**  The plan picks the half-step to
sit at the Dirichlet vertex, where it has the cleanest stencil
treatment (see §4.2).

For the example in the user's prompt (axis $x$ with N at lower, D at
upper, on the unit cube $[0,1]$ with $N_x$ cells, $h_x = 1/N_x$):
$$
\underbrace{-\tfrac{h_x}{2}}_{\text{ghost}},\;
\underbrace{\tfrac{h_x}{2}, \tfrac{3 h_x}{2}, \ldots, 1 - \tfrac{h_x}{2}}_{\text{cell centres (uniform spacing $h_x$)}},\;
\underbrace{1}_{\text{Dirichlet vertex (last gap $= h_x/2$)}}.
$$
Total $N_x + 2$ grid points; $N_x + 1$ uniform $h_x$ gaps; **one**
half-step gap of $h_x/2$ at the Dirichlet end.

The non-uniform spacing only affects the boundary stencil at the
Dirichlet vertex on a hybrid axis: the second derivative there has
to use the standard non-uniform-grid formula
$$
u''(x_v) \;\approx\; \frac{2}{(h_a + h_a/2)}\!\left[ \frac{u_{\text{interior}} - u_v}{h_a/2} - \frac{u_v - u_{\text{interior, deeper}}}{h_a}\right]
$$
or, equivalently, a one-sided three-point formula using the Dirichlet
value at the vertex and the two nearest interior cell centres.  All
*interior* stencils are unchanged because the spacing is uniform $h_a$
between every consecutive pair of interior nodes.

Alternative designs considered and rejected:

- **Cell-centred everywhere on hybrid axes.**  Drop the "Dirichlet on
  vertex" stipulation on hybrid axes only; handle Dirichlet via a
  ghost cell with linear extrapolation $u_{\text{ghost}} =
  2g - u_{\text{interior}}$.  Restores uniform $h_a$ throughout but
  contradicts the user's explicit design choice that Dirichlet sits
  on a vertex.  Could be revisited if the half-step boundary stencil
  proves troublesome in practice.
- **Half-integer $N_a$ on hybrid axes.**  Make $h_a$ depend on the
  BC pair by setting $h_a = (b_a - a_a)/(N_a + 1/2)$ on hybrid axes,
  preserving uniform spacing.  Awkward because the "cell count" is
  no longer the natural physical count; multigrid coarsening also
  becomes awkward (halving an axis with the half-integer denominator
  produces a quarter-integer one).

### 2.3 Where the boundary lives in each layout

| Face | Physical boundary location | Grid representation |
|---|---|---|
| Dirichlet | Boundary $\equiv$ a grid point at $x_a = a_a$ or $b_a$ | Driver writes $u = g(\mathbf{x})$ at that node every time `apply_bc_3d` runs |
| Neumann   | Boundary lies *between* two grid points (the ghost and the first interior cell centre); the boundary plane is at $x_a = a_a$ (lower) or $b_a$ (upper) | The "ghost" node on a Neumann lower face is at $a_a - h_a/2$; the boundary $x_a = a_a$ is the midpoint between the ghost and the first interior cell centre |

The user's BC callbacks (Dirichlet $g$ and Neumann $q$) receive the
**physical** coordinate of the boundary plane, *not* the ghost-cell
coordinate.  This is unchanged from the current code — see §5.

### 2.4 The Neumann ghost-cell formula

For a cell-centred Neumann face at $x_a = a_a$ with
$\partial u/\partial n = q$ (outward normal $-\hat{x}_a$, so
$\partial u/\partial x_a = -q$), the centred difference *across the
boundary* is
$$
\frac{u_{x_1} - u_{x_0}}{h_a} \;=\; \frac{\partial u}{\partial x_a}\bigg|_{x_a=a_a} \;=\; -q.
$$
This is the natural finite-volume Neumann encoding.  Solving for the
ghost:
$$
u_{x_0} \;=\; u_{x_1} + h_a\,q\qquad\text{(lower-$a$, outward $-\hat{x}_a$)}.
$$
On the upper face the formula is identical but with the indices on the
other side: $u_{x_{M-1}} = u_{x_{M-2}} + h_a\,q$ (outward $+\hat{x}_a$).

**Crucially, this is *half* the magnitude of the current node-centred
mirror $u_{-1} = u_1 + 2 h q$**, because the cell-centred difference
spans $h$ rather than $2h$.

### 2.5 TOML schema extension

The `[grid]` section grows to specify the box bounds:

```toml
[grid]
nx_cells = 64
ny_cells = 32          # cell counts may differ per axis
nz_cells = 64

# Optional: domain bounds.  Default to [0, 1] per axis when absent
# (back-compat with all existing TOML files).
x0 = 0.0
xN = 1.0
y0 = 0.0
yN = 2.0               # arbitrary box, e.g. [0,1] x [0,2] x [0,1]
z0 = 0.0
zN = 1.0
```

The parser allowlist for `[grid]` (currently
`{nx_cells, ny_cells, nz_cells}`) extends to include `x0, xN, y0, yN,
z0, zN`.  When all six are absent, the driver falls back to the unit
cube — the back-compat path that lets every existing TOML keep
working.

## 3. Architectural changes

### 3.0 Naming convention: `global_n*` always means cells

In the current code, the domain field `domain.global_ni` is the
**grid-point count** ($N_a + 1$ in today's all-vertex layout); the
TOML key `nx_cells`, in contrast, is the cell count $N_a$, and the
driver computes `global_nx = nx_cells + 1` on the way into
`setup_3d_domain`.  This dual convention worked when there was a
constant relationship `points = cells + 1`, but it breaks under the
hybrid layout of §2.2 where the point count varies per BC pair
(`cells + 1`, `cells + 2`, depending on the kinds on each end).

The refactor aligns the convention: **`global_ni`, `global_nj`,
`global_nk` always mean cell counts, identical to the user-facing
TOML values `nx_cells, ny_cells, nz_cells`**.  In particular:

- `domain.global_ni == param.global_nx_cells == N_x`, exactly.
- The grid-point count along axis $a$ becomes a derived quantity
  computed inside `setup_1d_domain` from $N_a$ plus the BC pair on
  that axis (using the table in §2.2: $N_a + 1$, $N_a + 2$, $N_a + 2$,
  $N_a + 2$).
- The half-cell extension on a Neumann face is *not* counted by
  `global_n*`.  Two Neumann faces on the same axis still report
  `global_ni = N_x`; the two extra ghost cells are an implementation
  detail of the layout, hidden from the user-facing accounting.

The `local_nx`, `local_ny`, `local_nz` fields keep their existing
meaning as **the local data array length along axis $a$** (interior +
MPI ghost layer + Neumann ghost cell, as applicable).  These are
implementation-dependent because they include MPI ghost layers
already; adding the Neumann half-cell extension is just one more
contributor.  Code that needs the local *cell* count (the multigrid
hierarchy minimum-size check, for example) computes it from
`local_n*` minus the ghost contributions, the same pattern already
used in `ngfs_3d_create_child`.

Migration: any code reading `domain.global_ni` and *assuming* it is
the grid-point count must be updated.  A single grep over the source
tree pinpoints them — currently the calls are concentrated in
`multigrid.c` (hierarchy halving), `gauss_seidel.c` (loop bounds
derived from $N_a$), and `io.c` (JSON metadata fields written
verbatim from `global_ni`).  In particular, `io.c` writes
`"global_ni": <value>` to the per-rank JSON and any post-processing
that reads it must be aware that the meaning changed; document this
in CLAUDE.md.

### 3.1 Domain setup needs the BC spec *and* the user-supplied bounds

`setup_3d_domain` already accepts `global_x0/y0/z0` and `dx/dy/dz` as
parameters; the driver simply hard-codes them to $0.0$ and $1/(N-1)$
respectively.  After the refactor:

1. The driver reads $a_a$, $b_a$, $N_a$ from the parsed TOML.  These
   become $h_a = (b_a - a_a)/N_a$ for each axis.
2. The driver passes the per-face BC kinds *into* `setup_3d_domain`,
   which already handles the bounds and spacing as data.

`setup_3d_domain` and `setup_1d_domain` need the new BC-kind arguments
so they can decide the per-axis layout (number of grid points,
`local_i0` offset, physical coordinate of the local origin):

```c
int setup_3d_domain(int nx_cpu, int ny_cpu, int nz_cpu,
                    int rank,
                    int64_t nx_cells, int64_t ny_cells, int64_t nz_cells,
                    bc_kind_t lower_x, bc_kind_t upper_x,
                    bc_kind_t lower_y, bc_kind_t upper_y,
                    bc_kind_t lower_z, bc_kind_t upper_z,
                    int gs,
                    double a_x, double a_y, double a_z,    /* lower bounds */
                    double dx, double dy, double dz,       /* spacings */
                    struct domain3d_st *domain);
```

Note that $a_a$ here is the user's lower bound, *not* $a_a - h_a/2$:
the half-cell offset for a Neumann lower face is added inside the
function as $\texttt{global\_x0} \mathrel{-}= h_x/2$ when
`lower_x == BC_NEUMANN`.

The driver flow becomes:

1. Parse TOML → $a_a, b_a, N_a$ for each axis, plus the problem-preset
   name.
2. Look up the preset → obtain its `bc_spec_t`, which carries the
   per-face kinds.
3. Compute $h_a = (b_a - a_a)/N_a$ per axis.
4. Call `setup_3d_domain` with the bounds, spacings, and BC kinds.
5. Allocate `gfs` and stamp the *full* `bc_spec_t` (kinds + value
   callbacks) onto `gfs.bc`.
6. Build the hierarchy.  Coarse levels inherit per-axis layout flags;
   their bounds and spacings are derived from the fine level.

Today's order of operations is steps 1, 4, 5, 2, 6 — the BC spec is
stamped *after* domain setup.  The refactor reorders to 1, 2, 3, 4,
5, 6 so the BC kinds are available before the layout is fixed.

### 3.2 `struct ngfs_3d` changes

Add per-axis layout flags:

```c
struct ngfs_3d {
    /* ... existing ... */
    bool   neumann_lower_x, neumann_upper_x;  /* layout flags: where do
    bool   neumann_lower_y, neumann_upper_y;   * grid points lie relative
    bool   neumann_lower_z, neumann_upper_z;   * to the physical box? */
};
```

The struct already has `bc` (the full per-face spec including value
callbacks); these new flags duplicate the *kind* bit for hot-path
decisions in the kernels.

### 3.3 Hierarchy construction

The hierarchy halves cell counts.  For cell-centred axes the coarse
grid has $N_a/2$ cells of width $2 h_a$; the layout flags propagate
unchanged from parent to child (`bc_spec_homogenize` already preserves
`kind` while flipping `homogeneous`).

A subtle point: with cell-centred Neumann, the fine and coarse grid
points are **never coincident**.  Fine cell centres are at
$(i_f + 1/2) h_f$; coarse cell centres are at $(i_c + 1/2)(2 h_f) =
(2 i_c + 1) h_f$.  These have offsets of $h_f$ from each other —
the coarse cell at $i_c$ encloses two fine cells at $i_f = 2 i_c$
and $i_f = 2 i_c + 1$, both of which are interior to the coarse cell
(neither lies *on* a coarse grid point).

This is well-known in cell-centred multigrid and the standard remedy
is **trilinear prolongation from the eight surrounding coarse
cells** — see §4.4 below.  The vertex-centred trilinear prolongation
the code currently uses (which has a coincident-point case) does not
apply to cell-centred axes.

### 3.4 Operator-level identity tests

For axes that remain vertex-centred (D-D), all existing transfer
operators continue to work bit-for-bit.  For axes that become
cell-centred, the identity-on-bilinear-functions check that the
existing tests rely on must be reformulated, since the prolongation
operator changes shape.  Plan to add new tests for cell-centred
transfers (`test_prolong_cc_3d`, `test_restrict_cc_3d`).

## 4. Per-routine changes

### 4.1 `apply_bc_3d`

The rewrite makes Neumann faces *write* the ghost value into the array
(rather than be skipped, as in the current implementation):

```c
/* Neumann lower-x at i = 0: ghost cell.  Write u_xm = u_x1 + h_x q. */
if (gfs->domain.lower_x_rank == INVALID_RANK
        && gfs->bc->face[FACE_LOWER_X].kind == BC_NEUMANN)
{
    bc_fn_t cb = (gfs->bc->face[FACE_LOWER_X].homogeneous)
                  ? NULL : gfs->bc->face[FACE_LOWER_X].value;
    /* x at the boundary plane (not the ghost coordinate) is 0. */
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++) {
            const int64_t idx0 = gf_indx_3d(gfs, 0, j, k);
            const int64_t idx1 = gf_indx_3d(gfs, 1, j, k);
            const double y = gfs->y0 + j * gfs->dy;
            const double z = gfs->z0 + k * gfs->dz;
            const double q = cb ? cb(0.0, y, z, FACE_LOWER_X) : 0.0;
            v[idx0] = v[idx1] + gfs->dx * q;
        }
}
```

Dirichlet faces continue to work as today (write the prescribed value
at the boundary vertex).  Neumann faces no longer require any
"sweep widening" or in-stencil substitution.

### 4.2 `gauss_seidel_3d` and `calc_defect_3d`

Massive simplification for D-D and N-N axes: the per-stencil
ghost-mirror branches and the sweep-widening logic both disappear.
The sweep range becomes the *interior* in every direction, where
"interior" means *the indices the smoother updates*:

| Lower BC | Smoother starts at $i_{lo} =$ |
|----------|-------------------------------|
| D        | `gs + 1` if owned, else `gs`  |
| N        | `gs + 1` if owned, else `gs`  |

(In both cases the interior starts one node inside whatever boundary
or ghost the rank owns.)  Likewise on the upper side.  No per-axis
Neumann widening, no per-stencil mirror substitution — the kernel is
simply a 7-point stencil on a uniform grid, the standard form found in
every multigrid textbook.

#### Hybrid axes: the half-step at the Dirichlet vertex

On a hybrid D-N (or N-D) axis the layout has **one** non-uniform
spacing: the gap between the Dirichlet vertex and the nearest cell
centre is $h_a/2$ rather than $h_a$ (see §2.2).  This affects the
stencil in two places.

**At the Dirichlet vertex itself**: the vertex value is fixed by
`apply_bc_3d` (the user's $g$ is written there) and the smoother
does not update it.  No stencil concern.

**At the cell centre adjacent to the Dirichlet vertex** (call it
$x_v - h_a/2$, where $x_v$ is the vertex): the standard centred
second derivative
$$
u_{xx} \approx \frac{u_{i+1} - 2 u_i + u_{i-1}}{h_a^2}
$$
is wrong here because the gap to the Dirichlet vertex side is $h_a/2$
while the gap to the deeper-interior side is $h_a$.  Use the standard
non-uniform-grid second-difference instead:
$$
u_{xx}\!\bigm|_{x_a = x_v - h_a/2}
\;\approx\;
\frac{2}{h_L + h_R}\!\left( \frac{u_v - u_i}{h_R} - \frac{u_i - u_{i-1}}{h_L}\right),
\qquad h_L = h_a,\; h_R = h_a/2.
$$
Substituting $h_L = h_a, h_R = h_a/2$ and simplifying:
$$
u_{xx}\!\bigm|_{x_v - h_a/2}
\;\approx\;
\frac{4}{3 h_a^2}\!\left[ 2(u_v - u_i) - (u_i - u_{i-1}) \right]
\;=\;
\frac{4}{3 h_a^2}\,(2 u_v - 3 u_i + u_{i-1}).
$$
This is a one-line correction in the kernel: a single per-stencil
branch on "is this the cell adjacent to a hybrid-axis Dirichlet
vertex?" that picks the non-uniform formula instead of the uniform
one.  The branch fires on a single 2D plane per hybrid-axis Dirichlet
face — $\bigO(N^2)$ cells out of $\bigO(N^3)$ interior — so the cost
is negligible and the rest of the inner loop stays branch-free.

The same modification applies to `calc_defect_3d`.

The truncation error of the non-uniform formula is $\bigO(h_a)$
locally — one order lower than the $\bigO(h_a^2)$ uniform formula
elsewhere.  Globally this **does not** spoil the second-order rate
because the affected nodes form a measure-zero (codimension-1)
subset of the grid and the global error norm is still dominated by
the $\bigO(h^2)$ interior contribution.  This is a standard fact
about boundary-condition discretisations on regular grids; see
\cite[\S2.4]{leveque2007finite} for the supporting analysis.

The same lower-order-at-the-boundary statement holds on every level
of the multigrid hierarchy (each level uses the same 3-point
non-uniform formula at *its* boundary cell with its own $h$).
Since the multigrid V-cycle converges to the *fine* discrete
solution $u_h^*$ defined by $A_h u_h^* = f_h$ — independent of the
coarse operators $A_H$ — coarser-level inaccuracy does not affect
the fine-grid solution accuracy.  It can only affect the
per-V-cycle rate constant via the variational-property bound
$\rho(M_h) \le \rho_{\mathrm{smooth}} \cdot \rho_{\mathrm{approx}}$
of \cite[Ch. 6]{hackbusch1985multi}, where the coarse-grid
stencil's $\bigO(h)$ truncation contributes to
$\rho_{\mathrm{approx}}$.  Both factors stay bounded away from 1
uniformly in $h$, so $h$-independent convergence is preserved.

#### Higher-order option (4-point asymmetric stencil)

If the 3-point formula's local $\bigO(h)$ truncation proves
problematic in practice — see "Practical recommendation" below for
the criteria — replace it with a 4-point stencil that uses one
extra interior cell:
$$
u''(x_i) \;\approx\;
\frac{8}{15 h_a^2} u_v
\;-\; \frac{4}{5 h_a^2} u_i
\;+\; \frac{2}{3 h_a^2} u_{i-1}
\;-\; \frac{2}{15 h_a^2} u_{i-2}.
$$
Truncation error: $-\tfrac{h_a^2}{12} u^{(4)}(x_i) + \bigO(h_a^3)$,
i.e., $\bigO(h_a^2)$ — same order as the interior stencil.  The
row sum is zero, so the linear part still has constants in its
null space (consistent with the row-sum analysis of §6).
Implementation cost: one extra column in the boundary row of
$A_h$, which means one extra term in the smoother kernel branch
and the defect-kernel branch at this cell.  No effect on the
sweep range or stencil width elsewhere.

#### Practical recommendation

You can use the simple 3-point non-uniform formula at the boundary
cell on every level of the hierarchy without worrying about it
degrading the fine-grid solution accuracy.  The 4-point
higher-order alternative would only matter if either:

- the rate constant proves disappointingly large (say
  $\rho(M_h) > 0.4$ per V-cycle), making the V-cycle slow, or
- the convergence test verifier comes back with an empirical rate
  below $1.8$ on real problems.

Both are quantitative concerns about the iterative solver's
efficiency, not about the correctness or order of the discrete
solution.  Implement the 3-point form first as Phase 2 of §8
specifies, run the convergence tests, and only escalate to the
4-point form if either red flag appears.

### 4.3 `inject_var_3d` and `restrict_var_3d`

Vertex-centred axes: existing implementation works unchanged.

Cell-centred axes: there is **no coincident-point case**.  Injection
becomes one of:

- The 1D coarse cell value is the average of the two enclosed fine
  cells: $u_H[i_c] = \tfrac{1}{2}(u_h[2 i_c] + u_h[2 i_c + 1])$.
- For 3D, the coarse cell is the average of the eight enclosed fine
  cells.

Restriction (full-weighting) likewise requires a different stencil for
cell-centred axes, since the "centre + 6 face neighbours + 12 edges +
8 corners" weighting only applies when fine and coarse points
coincide.  The cell-centred analogue is described in
\cite{trottenberg2001multigrid} §2.8 and is essentially the adjoint of
the trilinear prolongation in §4.4.

### 4.4 `prolong_var_3d`

Vertex-centred axes (D-D): existing implementation works.

Cell-centred axes: each fine cell is interior to exactly one coarse
cell.  For trilinear prolongation, the fine value is the trilinear
combination of the 8 *surrounding* coarse cell centres — and crucially,
the 8 surrounding cells are all on the same coarse grid, with the fine
cell offset from the central coarse cell by $\pm h_f$ in each direction.
The eight stencil weights are $(3/4)^a (1/4)^{3-a}$ where $a$ is the
number of "near" cell centres in the trilinear stencil
(see `Boundary_plan.md` §2.8 in Trottenberg-Oosterlee-Schüller for the
explicit formulae).

Mixed axes (D on one end, N on the other) need careful per-axis
handling.  In practice this means the prolongation kernel walks each
fine point and decides per-axis whether to use vertex-centred or
cell-centred interpolation.

The plan recommends a **separate `prolong_var_cc_3d` function** for
the all-cell-centred case, with the *current* `prolong_var_3d` left
unchanged for the vertex-centred case.  The hierarchy chooses which
to call based on the axis layout flags.  Mixed-axis problems use a
hybrid kernel that branches per axis.

## 5. Coordinate API

User-facing callbacks (RHS, Dirichlet $g$, Neumann $q$, exact
solution) take physical coordinates $(x, y, z)$ in the user's box
$\Omega = [a_x, b_x] \times [a_y, b_y] \times [a_z, b_z]$ and *do not
need to change shape*.  The driver evaluates them at:

- **For RHS**: every grid point (interior + ghosts).  Ghost values of
  $f$ are not used by the kernel but are harmless to compute (and may
  be useful diagnostically).  Coordinates passed to the callback can
  lie *outside* $[a_a, b_a]$ on Neumann ghost cells; the user's RHS
  function should be defined at least to one cell beyond the box if
  the user wants finite values there, but in practice the kernel
  doesn't read the ghost-cell RHS so the value doesn't matter.
- **For Dirichlet $g$**: each Dirichlet boundary node at
  $x_a \in \{a_a, b_a\}$.
- **For Neumann $q$**: each Neumann face plane, with the face $x_a$
  coordinate at the *physical boundary* ($a_a$ or $b_a$), not at the
  ghost cell-centre.  In code, `apply_bc_3d` evaluates the callback
  at the boundary coordinate, not at the ghost-cell location.

This means *the user writes their callbacks in terms of the natural
coordinates of their problem* — there is no normalisation to $[0,1]^3$
or to a cell-centre frame.  A callback for a problem on
$[-1, 1]^3$ (for example) accepts coordinates in $[-1, 1]^3$
directly, with $a_a = -1$, $b_a = +1$, $h_a = 2/N_a$.

## 6. Compatibility for singular all-Neumann

With the cell-centred discretisation, the row sums of $A_h$ continue
to vanish (the constant function is in the null space).  The discrete
compatibility condition is now
$$
\sum_i \tilde{f}_i \cdot V_i \;=\; 0
$$
where $V_i = h_a^d$ for every node (cell volume).  This collapses to
the natural midpoint-rule $\int_\Omega f \, d\mathbf{x} \approx \sum_i f_i \, V_i$
on the grid, and the boundary-cell ghost contribution is automatically
the half-cell weight times $q/h$ — which exactly matches the
continuous $\int_{\partial\Omega} q \, dS$ as $h \to 0$.

Net effect: `manufactured_neumann_inhomog` should converge at rate 2
*without* any RHS shift, and could be re-enabled in the convergence
test suite.

## 7. Test plan

### 7.1 Existing tests

- **Operator-level tests (`test_*_2d/3d`).**  Every test currently
  uses Dirichlet-on-all-faces (default `gfs->bc == NULL` semantics).
  Under the refactor that path is unchanged — vertex-centred grids
  with the original `apply_bc_3d` and original prolongation/restriction.
  These should all pass without modification.  Plan: keep
  `test_*` as the regression suite for vertex-centred behaviour.

- **Convergence tests.**  The five existing convergence presets
  (`manufactured_dirichlet_homog/inhomog`, `manufactured_neumann_homog`,
  `manufactured_mixed`, `manufactured_mixed_inhomog`) all need re-validation
  on the new grid layout.  The numerical *error* may shift slightly
  because the discretisation has changed; the *rate* must remain 2.
  Plan: re-run all five and update any baseline error magnitudes in the
  test expectations (the verifier only checks rate, so this should be
  automatic).

### 7.2 New tests

- `test_prolong_cc_3d` and `test_restrict_cc_3d` — cell-centred
  transfer operators on identity-of-bilinear-functions check, with
  three resolutions to confirm second-order accuracy of the transfer.

- `convergence_manufactured_neumann_inhomog` — re-enable in the
  CTest list once the cell-centred Neumann discretisation is in place;
  expect rate 2.

- New presets exercising mixed *axes* (e.g. one axis Dirichlet,
  another Neumann): essentially the existing `manufactured_mixed`
  problem, but now the underlying grid genuinely has different layouts
  on different axes.

## 8. Phasing

The refactor is large enough that landing it in one commit is
indefensible.  Five-phase rollout, each phase ending with passing
CI:

### Phase 1 — Domain plumbing, no behavioural change
- Extend the TOML schema with optional `[grid] x0, xN, y0, yN, z0, zN`
  keys (default to $0.0$, $1.0$ when absent).  Update the
  `multigrid_parameters.cc` parser allowlist.  All existing TOML files
  (which omit the new keys) continue to produce a unit-cube domain.
- **Rename `domain.global_n*` semantics from grid-point count to cell
  count** (§3.0).  Touch every reader: `multigrid.c` (hierarchy halving
  test now divides cell counts not point counts -- already true for
  the user-facing test, but the internal field changes meaning),
  `gauss_seidel.c` (loop bounds derived from cell count + BC kind),
  `io.c` (JSON `global_ni` field renamed to `global_cells_x` for
  clarity, with a CLAUDE.md note about the meaning change).  Update
  the existing `verify_*.py` scripts that read JSON.  This rename is
  the **largest** mechanical change of the phase but is purely a
  semantic renaming -- when all faces are Dirichlet (today's case)
  the new "cell count" code path produces bit-identical numerical
  output to the old "point count - 1" path.
- Extend `setup_3d_domain` and `setup_1d_domain` to receive per-axis
  BC kinds in addition to the already-present bounds/spacing
  parameters.  When all six BC kinds are `BC_DIRICHLET` and the bounds
  default to $[0,1]^3$, the layout is identical to today's
  (vertex-centred everywhere) and every test should pass bit-for-bit.
- Plumb the BC kinds through the driver.  The driver now resolves the
  preset *before* domain setup so the kinds are known at layout time.
- Add the per-axis Neumann layout flags to `struct ngfs_3d`, leave
  them all `false` for now.
- Existing tests pass unchanged.  Add one new test that exercises a
  non-unit-cube domain (e.g. $[-1, 1]^3$ or $[0,2] \times [0,1]^2$)
  to verify the bounds-plumbing alone, before any layout change
  lands.

### Phase 2 — Cell-centred N-N axes only
- Implement the cell-centred grid for axes where *both* ends are
  Neumann; mixed axes still error out with `MPI_Abort` (the Phase 4
  scope).
- Rewrite `apply_bc_3d`, `gauss_seidel_3d`, `calc_defect_3d` per §4.1
  and §4.2.  The changes are large but localised.
- Add cell-centred prolongation/restriction (§4.3, §4.4) for the
  all-cell-centred case.
- Re-validate `manufactured_neumann_homog`.  Re-enable
  `manufactured_neumann_inhomog` (Phase 6 of `Boundary_plan.md` was
  deferred for this exact reason; Phase 2 here closes it).

### Phase 3 — Mixed axes (D on one end, N on the other)
- Extend the prolongation/restriction kernels to handle the per-axis
  branching for mixed axes.  Each fine point's per-axis offset is
  decided independently.
- Re-enable convergence tests for `manufactured_mixed` and
  `manufactured_mixed_inhomog` and verify they still converge at rate 2.

### Phase 4 — Test suite refresh
- Add the new `test_prolong_cc_3d` / `test_restrict_cc_3d` operator-
  level tests.
- Update `Boundary_plan.md` to remove the singular-Neumann-inhomog
  limitation entry; update `Plan.md`.
- Update `doc/documentation.tex` §3 (Discretisation) to describe the
  hybrid layout, and §6 (Implementation) to describe the per-axis
  layout flags and the simplified kernel.

### Phase 5 — Smoother retuning (optional)
- The current convergence test uses $\omega = 1.0$ and
  `n_iters = 60` to work around the slow rate observed for Neumann/
  mixed problems with the current discretisation (rate ~0.34).  After
  Phase 2 lands the cell-centred layout, the smoother kernel becomes
  the *standard* 7-point Gauss-Seidel SOR — the same operator that
  has rate 0.001 for pure Dirichlet.  Plan to re-test whether
  $\omega = 1.5$ now works for Neumann/mixed and whether `n_iters`
  can drop back to 20.

## 9. Risks

- **Hierarchy depth on cell-centred axes.**  Halving cells on an
  axis with Neumann ghosts means the ghost cells halve too, which is
  fine.  But the minimum cell count `min_cells` (currently 4) must
  be reinterpreted as "interior cells", not "total cells including
  ghosts".  The MPI-Allreduce check in `ngfs_3d_create_child` already
  computes `local_x - (lower_x_rank != INVALID_RANK ? gs : 0) - ...`,
  i.e., the interior count; this should remain correct under the
  refactor.

- **Mixed-axis prolongation off-by-one.**  Cell-centred prolongation
  has a different stencil shape than vertex-centred.  Getting the
  per-axis branching right at every fine-grid point will be the
  hairiest part of Phase 3.  Mitigation: write the new
  `prolong_var_cc_3d` first as a pure cell-centred function (no axis
  branching), validate it against `test_prolong_cc_3d`, then layer
  the per-axis dispatch on top.

- **MPI face exchange semantics.**  Currently the rank that owns a
  Neumann boundary has no MPI ghost on that side; under the refactor
  it has a *physical* ghost (the cell-centred grid's outermost cell)
  but still no *MPI* ghost.  `sync_var_3d` must continue to skip
  faces where `lower_*_rank == INVALID_RANK`, regardless of BC kind
  — which is what it already does.

- **Test baseline shifts.**  Existing convergence tests assert rate
  $\in [1.8, 2.3]$ on the finest pair, which is robust to a shift in
  the discretisation constant.  No baseline updates expected.

- **Output JSON reading.**  Per-rank JSON output now includes ghost
  cell coordinates that lie outside $[0,1]$.  Any post-processing
  scripts that assume all coordinates are in $[0,1]$ may need to
  filter ghost cells.  None of the existing `verify_*.py` scripts
  do, so no expected fallout, but document this in CLAUDE.md.

## 10. Open questions

1. ~~**Should `nx_cells` continue to mean the user-facing cell count,
   or should it be the total stored grid-point count?**~~  *Resolved
   per §3.0: `nx_cells`, the corresponding TOML keys, and the internal
   `domain.global_n*` fields all uniformly mean cell counts.  The
   half-cell ghost extension on a Neumann face is excluded from
   `global_n*` — the user sees an $N_a$-cell grid inside the box,
   regardless of which boundary kinds are in use on each face.*

2. **What if `min_cells = 4` produces a coarsest level where a
   Neumann face has only 4 cells of which 2 are ghosts?**  At that
   level, the cell-centred discretisation has only 2 interior cells
   per direction — close to the "no multigrid acceleration possible"
   limit.  Recommended: bump `min_cells` default to 6 or 8 when any
   axis is cell-centred, with a CMake configure-time comment.

3. **Can the singular-case mean-zero projection be retired?**  In the
   cell-centred discretisation, the discrete operator is still
   singular for all-Neumann (constant null space).  Mean-zero
   projection is still needed to control round-off drift along the
   constant mode.  Plan to keep it.

4. **Coupling to the §6.2 "sub-optimal SOR rate on Neumann"
   investigation.**  The current ω=1.0 + n_iters=60 setting was
   chosen because the node-centred Neumann discretisation produced
   slow convergence on cos-based manufactured solutions at fine $h$.
   If the cell-centred refactor restores h-independent convergence
   (as the analysis suggests), this remediation can be retired.
   Tracked in Phase 5.
