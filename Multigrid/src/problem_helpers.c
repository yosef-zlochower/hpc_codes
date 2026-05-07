/*
 * Driver-side helpers that consume a problem_t.  They are the only
 * place the solver evaluates problem->rhs and problem->u_exact, so
 * the V-cycle, smoother, and transfer operators stay problem-
 * agnostic.
 */
#include "problem.h"
#include "gauss_seidel.h" /* VAR_SOL, VAR_RHS */
#include "gf.h"
#include <math.h>
#include <mpi.h>
#include <stddef.h>

/* Range over which a variable is "owned" by this rank, i.e. the local
 * indices that hold real *unknown* data (not MPI ghost cells, not
 * physical-boundary ghost cells either).  Used for whole-domain
 * reductions (mean projection, error norms).
 *
 * For cell-centred axes (both ends Neumann under CellCentred Phase 2)
 * the boundary-cell row at index 0 / nx-1 is a *ghost*: its value
 * u_int + h*q is a slave of the interior cell, not an independent
 * unknown.  Including it in a reduction over "owned" cells would
 * bias mean-projections by the per-axis inhomogeneous-Neumann
 * contribution and contaminate the convergence rate.  For
 * vertex-centred axes (DD or hybrid in Phase 2) the boundary node
 * IS the unknown (Dirichlet constraint or Neumann unknown updated
 * by the smoother), so it stays in.
 *
 * MPI-shared faces always trim the gs-wide ghost layer regardless of
 * BC kind. */
static inline void owned_bounds_3d(const struct ngfs_3d *gfs,
                                   int64_t *i_lo, int64_t *i_hi,
                                   int64_t *j_lo, int64_t *j_hi,
                                   int64_t *k_lo, int64_t *k_hi)
{
    const bool x_cc = gfs->domain.neumann_lower_x && gfs->domain.neumann_upper_x;
    const bool y_cc = gfs->domain.neumann_lower_y && gfs->domain.neumann_upper_y;
    const bool z_cc = gfs->domain.neumann_lower_z && gfs->domain.neumann_upper_z;

    *i_lo = (gfs->domain.lower_x_rank == INVALID_RANK) ? (x_cc ? 1 : 0)
                                                       : gfs->gs;
    *i_hi = (gfs->domain.upper_x_rank == INVALID_RANK) ? (x_cc ? gfs->nx - 1
                                                              : gfs->nx)
                                                       : gfs->nx - gfs->gs;
    *j_lo = (gfs->domain.lower_y_rank == INVALID_RANK) ? (y_cc ? 1 : 0)
                                                       : gfs->gs;
    *j_hi = (gfs->domain.upper_y_rank == INVALID_RANK) ? (y_cc ? gfs->ny - 1
                                                              : gfs->ny)
                                                       : gfs->ny - gfs->gs;
    *k_lo = (gfs->domain.lower_z_rank == INVALID_RANK) ? (z_cc ? 1 : 0)
                                                       : gfs->gs;
    *k_hi = (gfs->domain.upper_z_rank == INVALID_RANK) ? (z_cc ? gfs->nz - 1
                                                              : gfs->nz)
                                                       : gfs->nz - gfs->gs;
}

/* Compute the discrete mean of `var` over the owned nodes of every
 * rank in the Cartesian communicator.  Returns 0 if the count is 0
 * (defensive; never happens in practice). */
static double global_mean_3d(struct ngfs_3d *gfs, int var)
{
    int64_t i_lo, i_hi, j_lo, j_hi, k_lo, k_hi;
    owned_bounds_3d(gfs, &i_lo, &i_hi, &j_lo, &j_hi, &k_lo, &k_hi);

    const double *v = gfs->vars[var]->val;
    double  local_sum   = 0.0;
    int64_t local_count = 0;
    for (int64_t k = k_lo; k < k_hi; k++)
        for (int64_t j = j_lo; j < j_hi; j++)
            for (int64_t i = i_lo; i < i_hi; i++) {
                local_sum += v[gf_indx_3d(gfs, i, j, k)];
                local_count++;
            }

    double  global_sum;
    int64_t global_count;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                  gfs->domain.cart_comm);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT64_T, MPI_SUM,
                  gfs->domain.cart_comm);
    if (global_count == 0) return 0.0;
    return global_sum / (double)global_count;
}

void problem_project_mean_zero(struct ngfs_3d *gfs, int var)
{
    int64_t i_lo, i_hi, j_lo, j_hi, k_lo, k_hi;
    owned_bounds_3d(gfs, &i_lo, &i_hi, &j_lo, &j_hi, &k_lo, &k_hi);

    const double mean = global_mean_3d(gfs, var);
    double *v = gfs->vars[var]->val;
    for (int64_t k = k_lo; k < k_hi; k++)
        for (int64_t j = j_lo; j < j_hi; j++)
            for (int64_t i = i_lo; i < i_hi; i++)
                v[gf_indx_3d(gfs, i, j, k)] -= mean;
}

void problem_initialise_rhs(struct ngfs_3d *gfs,
                            const struct problem_t *problem)
{
    double *rhs = gfs->vars[VAR_RHS]->val;
    double *sol = gfs->vars[VAR_SOL]->val;

    for (int64_t k = 0; k < gfs->nz; k++)
    {
        const double z = gfs->z0 + k * gfs->dz;
        for (int64_t j = 0; j < gfs->ny; j++)
        {
            const double y = gfs->y0 + j * gfs->dy;
            for (int64_t i = 0; i < gfs->nx; i++)
            {
                const double  x   = gfs->x0 + i * gfs->dx;
                const int64_t idx = gf_indx_3d(gfs, i, j, k);
                rhs[idx] = problem->rhs(x, y, z);
                sol[idx] = 0.0;
            }
        }
    }

    /* Singular problems (all-Neumann) require the discrete
     * compatibility condition sum f_h = sum q/h_normal (in 3D, summed
     * over every face).  For homogeneous Neumann this collapses to
     * sum f_h = 0 -- which is what `problem_project_mean_zero` enforces
     * -- but for inhomogeneous Neumann the boundary contribution is
     * non-zero, and naively zeroing the mean of f_h breaks the
     * compatibility.  In the cell-centred discretisation the
     * continuous compatibility int f = int q transfers to discrete
     * compatibility automatically (both sides are midpoint-rule
     * approximations of the same continuous integrals), so we leave
     * f_h alone and rely on the per-iteration mean-zero projection of
     * VAR_SOL to control round-off drift along the constant mode. */
    (void)problem;
}

void problem_apply_initial_bc(struct ngfs_3d *gfs,
                              const struct problem_t *problem)
{
    /* Driver-side hook for any initial boundary write that should
     * happen exactly once before the first iteration.  In practice
     * apply_bc_3d (called by the smoother every half-sweep) already
     * enforces Dirichlet values on every iteration, and Neumann faces
     * have no per-iteration write at all (the boundary node is
     * updated by the smoother sweep itself), so this function is
     * currently a no-op.  Kept as a stable extension point. */
    (void)gfs;
    (void)problem;
}

double problem_compute_max_error(struct ngfs_3d *gfs,
                                 const struct problem_t *problem)
{
    if (!problem->u_exact) return -1.0;

    int64_t i_lo, i_hi, j_lo, j_hi, k_lo, k_hi;
    owned_bounds_3d(gfs, &i_lo, &i_hi, &j_lo, &j_hi, &k_lo, &k_hi);

    const double *sol = gfs->vars[VAR_SOL]->val;

    /* For singular problems the discrete solution is determined only
     * up to an additive constant.  Compute the global mean of
     * (u_h - u_exact) and subtract before measuring the L-infinity
     * error -- otherwise the reported error is dominated by the
     * arbitrary choice of constant. */
    double mean_diff = 0.0;
    if (problem->singular)
    {
        double  local_sum   = 0.0;
        int64_t local_count = 0;
        for (int64_t k = k_lo; k < k_hi; k++) {
            const double z = gfs->z0 + k * gfs->dz;
            for (int64_t j = j_lo; j < j_hi; j++) {
                const double y = gfs->y0 + j * gfs->dy;
                for (int64_t i = i_lo; i < i_hi; i++) {
                    const double  x   = gfs->x0 + i * gfs->dx;
                    const int64_t idx = gf_indx_3d(gfs, i, j, k);
                    local_sum += sol[idx] - problem->u_exact(x, y, z);
                    local_count++;
                }
            }
        }
        double  global_sum;
        int64_t global_count;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                      gfs->domain.cart_comm);
        MPI_Allreduce(&local_count, &global_count, 1, MPI_INT64_T, MPI_SUM,
                      gfs->domain.cart_comm);
        if (global_count > 0) mean_diff = global_sum / (double)global_count;
    }

    double local_err = 0.0;
    for (int64_t k = k_lo; k < k_hi; k++) {
        const double z = gfs->z0 + k * gfs->dz;
        for (int64_t j = j_lo; j < j_hi; j++) {
            const double y = gfs->y0 + j * gfs->dy;
            for (int64_t i = i_lo; i < i_hi; i++) {
                const double  x       = gfs->x0 + i * gfs->dx;
                const int64_t idx     = gf_indx_3d(gfs, i, j, k);
                const double  u_exact = problem->u_exact(x, y, z);
                const double  err     = fabs(sol[idx] - u_exact - mean_diff);
                if (err > local_err) local_err = err;
            }
        }
    }

    double global_err;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX,
                  gfs->domain.cart_comm);
    return global_err;
}
