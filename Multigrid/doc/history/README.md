# doc/history

Historical planning documents preserved for project history.  Their
content has been absorbed into the production code and the formal
reference manual at `doc/documentation.tex` -- they are no longer
load-bearing for understanding the current implementation but capture
the reasoning that led to it.

| File | Phase | Purpose |
|------|-------|---------|
| `Boundary_plan.md` | Boundary-condition work | Four-phase plan for the `bc_spec_t` per-face BC machinery, `problem_t` registry, and the smoother / defect / prolongation changes needed to support every Dirichlet / Neumann combination. |
| `CellCentred_plan.md` | Cell-centred discretisation work | Plan for the cell-centred and hybrid axis layouts, the cell-centred transfer operators (box-average restriction, position-aware trilinear prolongation), and the 4-point Lagrange stencil at hybrid Dirichlet vertices. |

The active project log is `Plan.md` at the repository root; the
reference manual at `doc/documentation.tex` carries the algorithmic
description of what landed.
