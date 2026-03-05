# Multigrid Solver Code Review

Files reviewed:
- `multigrid_numba.py` — 1D multigrid solver for general second-order ODEs
- `multigrid2d_numba.py` — 2D Poisson multigrid solver
- `multigrid3d_numba.py` — 3D Poisson multigrid solver

---

## 1. Overview and Architecture

All three files follow the same architecture: a `Multigrid` class whose constructor
recursively builds the grid hierarchy (halving the cell count at each level), and a
set of Numba `@njit`-compiled kernel functions for performance-critical inner loops.
The class methods delegate to these kernels and manage the multigrid V-cycle logic.

**Shared design features across all files:**
- `@njit`-compiled smoothers, defect kernels, prolongation, and restriction
- SOR acceleration with an adaptive `omega` strategy
- Optional red-black Gauss-Seidel decoupling (`red_black_decouple=True`)
- Optional direct (injection) or full-weighting restriction (`direct_restriction`)
- V-cycle with optional sub-cycling for W-cycle-like behaviour (`subcycles > 1`)
- User-supplied, JIT-compiled `boundary` callback, enabling non-homogeneous BCs

---

## 2. Correctness of Core Equations

### 2.1 Gauss-Seidel Smoother

**1D** (`multigrid_numba.py:342–347`) solves `d2(x)u'' + d1(x)u' + d0(x)u = f`
with second-order central differences. Isolating `u[i]` from the 5-point stencil:

```
u[i] = [(u[i+1]*(d2 + ½h·d1) + u[i-1]*(d2 - ½h·d1) - h²·f)] / (2·d2 - h²·d0)
```

The implementation with SOR is:

```python
sol[i] = (1 - omega)*sol[i] + omega * (
    sol[i+1]*(d2coef[i] + 0.5*dx*d1coef[i])
    + sol[i-1]*(d2coef[i] - 0.5*dx*d1coef[i])
    - dx2*src[i]
) / (2*d2coef[i] - dx2*d0coef[i])
```

This is correct.

**2D** (`multigrid2d_numba.py:354–360`) solves `∆u = f`. Isolating `u[i,j]` from
the 5-point Laplacian with variable mesh spacing:

```
u[i,j] = [(u[i+1,j]+u[i-1,j])·dy² + (u[i,j+1]+u[i,j-1])·dx² - dx²dy²·f]
          / [2(dx²+dy²)]
```

The implementation precomputes `ifac = 1/(2*(dx²+dy²))` and applies SOR. Correct.

**3D** (`multigrid3d_numba.py:391–399`) solves `∆u = f` in 3D. The stencil coefficients are precomputed:

```python
denom = 2*(dx²dy² + dx²dz² + dy²dz²)
cx = dy²dz²/denom,  cy = dx²dz²/denom,  cz = dx²dy²/denom
cs = dx²dy²dz²/denom
```

The update `sol[i,j,k] = (1-ω)*sol[i,j,k] + ω*((sol[i±1,j,k])*cx + (sol[i,j±1,k])*cy + (sol[i,j,k±1])*cz - cs*src[i,j,k])` is the correct Gauss-Seidel update for the 3D 7-point Laplacian. Correct.

### 2.2 Defect Calculation

All three files define the defect as `L(u) - f` (residual), using second-order
central differences.

**1D** (`multigrid_numba.py:423–430`): standard 3-point stencil for a general ODE. ✓

**2D** (`multigrid2d_numba.py:444–450`): standard 5-point Laplacian. ✓

**3D** (`multigrid3d_numba.py:511–516`): standard 7-point Laplacian. ✓

All three return `np.abs(defect).max()` (the L∞ norm) as `defect_linf`.

### 2.3 Prolongation

The prolongation operators implement bilinear (2D) / trilinear (3D) interpolation,
subtracting the coarse-grid correction from the parent (fine-grid) solution.

**1D** (`multigrid_numba.py:229–231`): linear interpolation. For coarse index `i`:
- Coincident fine point: `parent[2i] -= sol[i]`
- Midpoint: `parent[2i-1] -= 0.5*(sol[i-1]+sol[i])`

**2D** (`multigrid2d_numba.py:456–475`): bilinear interpolation. For coarse `(i,j)`:
- Coincident fine point: `parent[2i,2j] -= sol[i,j]`
- x-midpoint: `parent[2i-1,2j] -= 0.5*(sol[i-1,j]+sol[i,j])`
- y-midpoint: `parent[2i,2j-1] -= 0.5*(sol[i,j-1]+sol[i,j])`
- xy-corner: `parent[2i-1,2j-1] -= 0.25*(sol[i-1,j-1]+sol[i-1,j]+sol[i,j-1]+sol[i,j])`

**3D** (`multigrid3d_numba.py:521–565`): trilinear interpolation. Covers all 8 fine-cell
vertex types (coincident, 3 edge types, 3 face types, and body-centre) with the
correct tensor-product weights (1, ½, ¼, ⅛). ✓

All three loop ranges cover exactly the non-boundary coarse interior points.
Boundary contributions are zero-valued by the `boundary()` function and do not
corrupt parent boundary conditions. ✓

**Sign convention:** the "subtract" convention is consistent with the defect sign
`L(u)-f` and the correction equation `L(e)=defect`, giving `u_new = u - e`. ✓

### 2.4 Restriction

**Direct (injection):**
- 1D: `child.src[i//2] = defect[i]` for even `i`.
- 2D: `child.src[1:-1,1:-1] = defect[2:-2:2, 2:-2:2]` (NumPy slice). ✓
- 3D: `child.src[1:-1,1:-1,1:-1] = defect[2:-2:2,2:-2:2,2:-2:2]` (NumPy slice). ✓

**Full-weighting (adjoint of prolongation):**
- 1D: 3-point stencil `[¼, ½, ¼]`. ✓
- 2D (`multigrid2d_numba.py:480–493`): 9-point tensor-product stencil:

  ```
         [1  2  1]
  (1/16)·[2  4  2]
         [1  2  1]
  ```
  Weights sum to 1. ✓

- 3D (`multigrid3d_numba.py:569–592`): 27-point tensor-product stencil with weights
  by neighbour type: centre 8/64, face 4/64, edge 2/64, corner 1/64. Total = 1. ✓

Both full-weighting operators are the exact adjoint of their respective prolongation
operators, satisfying the Galerkin condition `R = c·Pᵀ`.

### 2.5 Red-Black Gauss-Seidel

**1D:** Red sweep covers even indices `{2, 4, ...}`, black covers odd `{1, 3, ...}`.
Together they span all interior indices `{1, ..., length-2}`. ✓

**2D:** For row `i`, `ieven = 1 - i%2`. Red sweep starts `j` at `1 + ieven` with
step 2; black starts at `2 - ieven` with step 2.
- `i` even: red = `j` even (i+j even), black = `j` odd (i+j odd). ✓
- `i` odd: red = `j` odd (i+j even), black = `j` even (i+j odd). ✓

**3D:** `ijeven = 1 - (i+j)%2`. Red starts `k` at `1 + ijeven`, black at `2 - ijeven`,
both with step 2. Red: `i+j+k` even; black: `i+j+k` odd. ✓

All red-black implementations correctly partition interior points.

### 2.6 V-cycle and Subcycle Structure

The V-cycle (`vcycle()`) follows the standard structure in all three files:

1. Pre-smooth (SOR Gauss-Seidel, `n_iters`)
2. Calculate defect
3. If child exists and defect > tolerance:
   - Restrict defect to child (child solution zeroed first)
   - Recursive `child.vcycle()`  (which itself calls `child.prolongate()` at its end)
   - Post-smooth
   - Re-compute defect
4. Prolongate correction to parent

Setting `subcycles > 1` causes the inner loop (steps 3a–3d) to execute multiple
times before prolongation, giving a W-cycle-like behaviour.

---

## 3. Memory Layout and Cache Efficiency

All Numba kernels iterate with the last index as the innermost loop, matching
NumPy's default C-contiguous (row-major) memory layout:

| Kernel | Loop order | Fast index |
|--------|-----------|-----------|
| 2D standard smoother | `i → j` | `j` ✓ |
| 2D red-black smoother | `i → j` (stride-2) | `j` ✓ |
| 2D defect | `i → j` | `j` ✓ |
| 3D standard smoother | `i → j → k` | `k` ✓ |
| 3D red-black smoother | `i → j → k` (stride-2) | `k` ✓ |
| 3D defect | `i → j → k` | `k` ✓ |
| 2D/3D `_prolongate_ext` | `i → j (→ k)` | ✓ |
| 2D/3D `_restrict_ext` | `i → j (→ k)` | ✓ |

---

## 4. Remaining Issues

### 4.1 `__str__` Omits `num_zcells` (3D)

`multigrid3d_numba.py:161`:

```python
string = str(self.num_xcells) + ", " + str(self.num_ycells)
```

The third dimension (`num_zcells`) is not included in the string representation.

### 4.2 `_solution_found` Flag Is Never Used

All three files set `self._solution_found = True` in `calculate_defect_2nd()` when
`defect_linf < tol`. This flag is never read anywhere in `vcycle()`,
`solve_to_tolerance()`, or the recursive child logic. It is a dead field — the
actual termination check in `solve_to_tolerance()` is `err > self.tol` and in
`vcycle()` it is `self.defect_linf > self.tol`.

---

## 5. Design Notes

### 5.1 Adaptive Omega

All three solvers implement adaptive SOR control via two thresholds:

- **`omega_high_error_limit`** (default 100): when `defect_linf > threshold`, `omega`
  is forced to 1.0 to avoid divergence during the early, large-error phase.
- **`omega_low_error_limit`** (default `tol * 10`): when `defect_linf < threshold`,
  `omega` is also forced to 1.0 for stability near convergence.

The ordering in `calculate_defect_2nd()` means that when `tol < defect_linf < omega_low_error_limit`,
`omega=1.0` is active. Once `defect_linf < tol`, `_solution_found` is set and
`omega` is restored to its standard value (which is inconsequential since the solve
has converged).

### 5.2 Double Post-Smoothing

After `child.vcycle()` returns, all three solvers apply two post-smoothing steps:

```python
self.gauss_seidel_smoothing(n_iters=smoothing_iters)               # SOR
self.gauss_seidel_smoothing(n_iters=smoothing_iters, omega_override=1.0)  # plain GS
```

The SOR pass reduces mid-frequency error aggressively; the plain GS pass stabilises
the solution.

A consequence of the `omega_low_error_limit` logic is that once the defect falls
below this threshold, the adaptive mechanism also switches the first pass to plain
GS (`omega=1`). At that point both passes are plain GS, effectively doubling the
post-smoothing work in the final convergence decade without the speed-up of SOR.

### 5.3 Grid Arrays in Numba Kernel Signatures

All `@njit` smoothers and defect kernels accept `gridx`, `gridy` (and `gridz` in 3D)
as arguments, even though these arrays are not used inside the kernel body. They are
passed through to the user-supplied `boundary()` callback, which may need grid
coordinates for non-homogeneous or Neumann boundary conditions. This design choice
preserves generality at the cost of slightly larger kernel call overhead.

### 5.4 `boundary()` Call Frequency

In the standard Gauss-Seidel smoother, `boundary()` is called at the start and end
of each outer iteration, for a total of `2 * n_iters` calls per smoothing invocation.
In the red-black smoother, it is called three times per outer iteration
(before red, before black, after black), for `3 * n_iters` calls. For the default
homogeneous Dirichlet condition (a no-op on interior points), this overhead is
negligible. For expensive user-supplied boundary functions the overhead may be more
significant, but correctness requires applying boundary conditions between the two
red-black half-sweeps so they see consistent ghost-point values.

### 5.5 Child Constructor Passes All Parameters

All three constructors correctly forward `omega_high_error_limit`,
`omega_low_error_limit`, `red_black_decouple`, and `direct_restriction` to child
constructors, ensuring consistent behaviour at all grid levels.

---

## 6. Functional Summary

The code is functionally correct for homogeneous Dirichlet boundary conditions in
all three dimensions. The core finite-difference stencils, prolongation operators
(linear/bilinear/trilinear), and full-weighting restriction operators (adjoint of
prolongation) are all correctly implemented. Cache-efficient loop ordering is used
throughout. All three solvers share a uniform feature set: adaptive omega, double
post-smoothing, and `omega_override` support. The remaining minor issues are: a
cosmetic omission in the 3D `__str__` method (`num_zcells` not printed); and an
unused `_solution_found` flag.
