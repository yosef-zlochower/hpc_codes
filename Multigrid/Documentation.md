# Multigrid Poisson Solver — Algorithm and Usage

## 1. Problem and discretisation

The driver in `driver_multigrid.c` solves the 3D Poisson equation

$$\nabla^2 u(x,y,z) \;=\; f(x,y,z) \quad \text{on } [0,1]^3, \qquad u\big|_{\partial\Omega}=0$$

with a manufactured source
$f(x,y,z) = -3\pi^2 \sin(\pi x)\sin(\pi y)\sin(\pi z)$, whose exact solution is $u_\text{exact}=\sin(\pi x)\sin(\pi y)\sin(\pi z)$.

- **Grid.** Node-based, uniform on $[0,1]^3$. The TOML specifies cell counts; the code uses `NX = nx_cells + 1` grid points and spacing `dx = 1/(NX-1)` (similarly y, z). One layer of ghost cells (`gs = 1`) on every face.
- **Operator.** Standard 7-point finite-difference Laplacian (`calc_defect_3d`, `src/gauss_seidel.c:196`):
  $L u_{ijk} = \tfrac{u_{i+1jk}-2u_{ijk}+u_{i-1jk}}{h_x^2} + \tfrac{u_{ij+1k}-2u_{ijk}+u_{ij-1k}}{h_y^2} + \tfrac{u_{ijk+1}-2u_{ijk}+u_{ijk-1}}{h_z^2}.$
- **Defect.** $d_{ijk} = (Lu - f)_{ijk}$. (Note the sign convention: the code uses *defect = Lu − f*, not the residual *f − Lu*. This determines the sign of the prolongation update — see §6.)
- **BCs.** Homogeneous Dirichlet, enforced by `apply_bc_3d` writing 0 to physical-boundary faces (any face not adjacent to an MPI neighbour).

## 2. MPI domain decomposition

`automatic_topology` (`domain.c`) computes a Cartesian process grid `(px, py, pz)` from `mpi_size` by greedy prime factorisation, biased toward the longer axes. `setup_3d_domain` then builds the Cartesian communicator and the per-rank local extents (`local_nx, local_ny, local_nz`) and ghost-aware origin (`local_i0, local_j0, local_k0`). All levels of the multigrid hierarchy share the same `(px, py, pz)`; coarsening reduces `local_n*` but never agglomerates ranks.

`sync_var_3d` performs the ghost exchange one axis at a time using non-blocking sends/receives over pre-allocated face buffers in `struct ngfs_3d`.

## 3. Smoother — red-black Gauss–Seidel SOR

`gauss_seidel_3d` (`src/gauss_seidel.c:86`) does `n_smooth` complete iterations, each consisting of a red half-sweep, ghost+BC sync, black half-sweep, ghost+BC sync.

- **Colouring** uses the *global* index parity, $\text{colour}(i,j,k) = (\,g_i + g_j + g_k\,) \bmod 2$, so the pattern is consistent across rank boundaries regardless of how the domain is decomposed.
- **Update.** With weights computed once per level from the spacings,
  $D = 2(h_x^2 h_y^2 + h_x^2 h_z^2 + h_y^2 h_z^2)$, $c_x = h_y^2 h_z^2/D$, $c_y = h_x^2 h_z^2/D$, $c_z = h_x^2 h_y^2/D$, $c_s = h_x^2 h_y^2 h_z^2/D$,
  the GS step is $\tilde u_{ijk} = c_x(u_{i\pm 1jk}) + c_y(u_{ij\pm 1k}) + c_z(u_{ijk\pm 1}) - c_s f_{ijk}$, and SOR mixes:
  $u_{ijk} \leftarrow (1-\omega) u_{ijk} + \omega \tilde u_{ijk}$.

## 4. Multigrid hierarchy construction

`ngfs_3d_create_hierarchy` (`src/multigrid.c:228`) repeatedly calls `ngfs_3d_create_child`, halving `cells_x, cells_y, cells_z` (so grid points become `cells/2 + 1`) and doubling `dx, dy, dz`. Coarsening stops when either:

1. any direction has an odd cell count (prevents exact 2:1 coarsening), or
2. the minimum interior cell count per rank, reduced via `MPI_Allreduce` with `MPI_MIN`, falls below `min_cells`.

Levels are stored as a parent → child linked list rooted at the finest grid.

For a fully resolvable hierarchy you want each global cell count to be of the form $2^p \cdot c$ with $c$ odd, so coarsening can recurse $p$ times.

## 5. Transfer operators (3D)

| Operator | Function | Stencil |
|----------|----------|---------|
| Injection | `inject_var_3d` | Direct copy at coincident points. Covers all non-ghost child points (incl. physical boundary). Used to seed coarse RHS before restriction. |
| Restriction | `restrict_var_3d` | Full-weighting on a 3×3×3 fine stencil: weights 8 (centre), 4 (faces), 2 (edges), 1 (corners), divisor 64. Applied to interior child points only. |
| Prolongation (correction) | `prolong_var_3d` | Trilinear interpolation by parity of the fine-grid global index: coincident → copy; face-midpoint → 2-pt avg; edge-midpoint → 4-pt avg; cell-centre → 8-pt avg. **Subtracts** the interpolated correction from the fine variable (`pval -= update`); skips physical boundary points to preserve Dirichlet BCs. Requires fresh ghost zones on both grids. |

The unusual *minus* in prolongation is consistent with the defect convention $d = Lu - f$: solving $Le = d$ on the coarse level and subtracting $e$ from $u$ corrects the fine solution.

## 6. V-cycle (`vcycle_3d`)

```
vcycle_3d(level, n_smooth, omega, tol, subcycles):
    gauss_seidel_3d(level, n_smooth, omega)        # pre-smooth
    norm = calc_defect_3d(level)                   # writes VAR_DEF, syncs ghosts
    if level.child and norm > tol:
        repeat up to `subcycles` times:
            zero child.VAR_SOL; apply_bc_3d
            inject_var_3d(level, VAR_DEF, child, VAR_RHS)
            restrict_var_3d(level, VAR_DEF, child, VAR_RHS)
            vcycle_3d(child, ...)                  # recurse — child prolongs into level.VAR_SOL
            sync_var_3d(level, VAR_SOL)
            gauss_seidel_3d(level, n_smooth, omega)   # post-smooth
            gauss_seidel_3d(level, n_smooth, 1.0)     # extra plain GS pass
            norm = calc_defect_3d(level)
            if norm <= tol: break
    if level.parent:
        prolong_var_3d(level, VAR_SOL, level.parent, VAR_SOL)   # subtract correction
    return norm
```

With `subcycles = 1` this is a standard V-cycle. With `subcycles > 1` and the early-exit on `tol`, the inner loop adds extra coarse-grid visits per level — closer to a W-cycle but data-dependent. The two post-smooths (one with `omega`, one with `omega = 1.0`) provide a slight smoothing finish before checking the defect.

## 7. Driver flow (`driver_multigrid.c`)

1. `parse_parameter_file` reads the TOML.
2. Compute `(px, py, pz)` and per-rank domain.
3. Allocate the root grid with 3 variable slots: `VAR_SOL = 0`, `VAR_RHS = 1`, `VAR_DEF = 2`.
4. If `multigrid = true`, build the hierarchy.
5. Initialise `VAR_RHS = -3π² sin(πx) sin(πy) sin(πz)` and `VAR_SOL = 0` (with BCs applied).
6. Outer loop, up to `n_iters`: either one `vcycle_3d` or one `gauss_seidel_3d + calc_defect_3d`, breaking when the defect falls below `tol`.
7. Compute the max-norm error against `u_exact`, print, and write per-rank JSON via `output_3d_gf`.

## 8. User-settable parameters (TOML)

The driver takes a single argument: the path to a TOML file (see `multigrid.toml`).

### `[grid]`

| Key | Type | Meaning |
|---|---|---|
| `nx_cells`, `ny_cells`, `nz_cells` | positive int | Cells per direction; grid points = cells + 1. For deepest hierarchy, prefer values of the form $2^p \cdot c$ with $c$ odd. |

### `[solver]`

| Key | Type | Meaning |
|---|---|---|
| `multigrid` | bool | `true` = V-cycle solver; `false` = single-grid Gauss–Seidel |
| `omega` | float, $(0,2)$ | SOR relaxation parameter; `1.0` = plain GS, `1<ω<2` = over-relaxation |
| `n_smooth` | positive int | Red-black GS sweeps per smoothing call |
| `n_iters` | positive int | Maximum outer iterations (V-cycles or GS calls) |
| `tol` | non-negative float | Convergence threshold on $\|d\|_\infty$ |
| `subcycles` | positive int *(multigrid only)* | Max coarse-grid visits per level (1 = V-cycle, >1 = W-like) |
| `min_cells` | positive int *(multigrid only)* | Minimum interior cells per rank, per direction, before coarsening stops |

When `multigrid = false`, the parser still expects the file to be parseable but the driver ignores `subcycles` / `min_cells`.

## 9. Extending to other source functions

The grid hierarchy, smoother, and transfer operators are entirely source-agnostic. Only the right-hand side, BCs, and the verification block are problem-specific. To switch to a different $f$ on the same domain with the same Dirichlet BC $u=0$, you only need the RHS-init loop in `src/driver_multigrid.c:125–139`:

```c
for (int64_t k = 0; k < gfs.nz; k++) {
    const double z = gfs.z0 + k * gfs.dz;
    for (int64_t j = 0; j < gfs.ny; j++) {
        const double y = gfs.y0 + j * gfs.dy;
        for (int64_t i = 0; i < gfs.nx; i++) {
            const double  x   = gfs.x0 + i * gfs.dx;
            const int64_t idx = gf_indx_3d(&gfs, i, j, k);
            rhs[idx] = my_f(x, y, z);   // <-- new source
            sol[idx] = 0.0;
        }
    }
}
```

The exact-error block at `src/driver_multigrid.c:168–185` should be replaced or removed unless you have a known $u_\text{exact}$.

The smoother coefficients in `gauss_seidel_3d` are derived from `dx, dy, dz` per level, so they automatically rescale on every coarse grid; nothing in the multigrid machinery depends on the form of $f$.

### Non-homogeneous Dirichlet BCs ($u = g$ on the boundary)

Two changes are needed:

1. In `apply_bc_3d` (`src/gauss_seidel.c:21`), replace the `0.0` writes with the boundary value $g(x,y,z)$ at each face point. Compute the global coordinates from `gfs->x0 + i*gfs->dx`, etc.
2. The coarse-level correction must remain *zero* on physical boundaries. The code already does the right thing: `vcycle_3d` zeroes `child.VAR_SOL` before recursion, `apply_bc_3d` is then called on the child (with `u = 0` semantics for the correction), and `prolong_var_3d` deliberately skips fine-grid physical boundary points so the BC values you set in step 1 are preserved.

In other words, **only `apply_bc_3d` for the fine grid needs to know about $g$** — but on coarse grids the BC for the *correction* must stay 0, so do not change the coarse-level `apply_bc_3d` calls invoked by `vcycle_3d`. The cleanest way is to add a separate `apply_bc_3d_inhomogeneous` for fine-grid initialisation only, and leave the existing `apply_bc_3d` (called on coarse correction grids and on `VAR_DEF`) alone.

### Periodic or Neumann BCs

Larger surgery:

- **Periodic.** `setup_3d_domain` currently passes `periods = {0, 0, 0}` to `MPI_Cart_create`; switch those to 1 in the relevant axes. `apply_bc_3d` should be made a no-op on periodic faces. The smoother and defect already work cleanly because periodic ghost zones become valid after `sync_var_3d`. `prolong_var_3d`'s `if (pg_x == 0 || pg_x == gni - 1) continue` guards must be removed on periodic axes.
- **Neumann.** Modify `apply_bc_3d` to enforce zero-derivative (mirror) values into the boundary face, and adjust `prolong_var_3d` similarly. Note that pure Neumann renders the operator singular up to a constant; you will likely need to project the RHS to be mean-zero each iteration.

### Different elliptic operators (variable coefficient, anisotropic, …)

The 7-point stencil and full-weighting restriction work well together for the constant-coefficient Laplacian. For $\nabla\!\cdot\!(a(\mathbf x)\nabla u)$ or strongly anisotropic problems:

- Rewrite `calc_defect_3d` and the inner update of `gauss_seidel_3d` (only the stencil; the colouring, BC handling, and ghost sync stay the same).
- Re-evaluate the transfer operators. For variable-coefficient or strongly anisotropic problems, geometric full-weighting often degrades; consider Galerkin coarse operators ($A_c = R A_f P$) or operator-dependent interpolation. That would also require representing the operator on every level, which currently it isn't (the smoother just needs `dx, dy, dz`).
- If you keep geometric coarsening but switch to a non-symmetric or non-self-adjoint $L$, the ω that worked for Poisson likely needs to drop.

### Multiple right-hand sides

Each call to the driver solves a single problem. To reuse a hierarchy across multiple RHSs, factor the setup (`setup_3d_domain` + `ngfs_3d_create_hierarchy`) out of the per-solve loop and call the V-cycle in a loop over RHSs, re-zeroing `VAR_SOL` and refilling `VAR_RHS` each time.

---

The cleanest refactor for extension would be to split `driver_multigrid.c` into a thin driver plus two user-supplied callbacks `f(x,y,z)` and `g(x,y,z)` (and optionally `u_exact(x,y,z)` for diagnostics) — at present these are inlined into `main`, which is why a new source needs an edit-and-rebuild.
