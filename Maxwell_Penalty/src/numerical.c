#include "numerical.h"
#include "gf.h"
#include "derivatives.h"
#include "timer.h"
#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* dissipation step. Damps high-frequency modes. Only needed in
 * nonlinear case.
 * !! Modifies ->dot in place.
 * !! must be called after the PDE RHS is in ->dot
 * !!
 * !! Dissipation is applied only on the locally-owned interior: the
 * !! ghost zones are skipped (they are overwritten by the next
 * !! sync_vars, so dissipating them is wasted work) and so are the
 * !! boundary-closure rows.  These are LOCAL indices, not global.
 * */

#define DISSIP5_HALF 3  /* half-width of dissipation stencil */
void apply_dissipation(struct ngfs *gfs, double diss_coeff)
{

    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t nz = gfs->nz;
    const double dx = gfs->dx;
    const double dy = gfs->dy;
    const double dz = gfs->dz;
    const double coefx = -diss_coeff / dx;
    const double coefy = -diss_coeff / dy;
    const double coefz = -diss_coeff / dz;

    const int64_t di = 1;
    const int64_t dj = nx;
    const int64_t dk = nx * ny;

    static int diss_timer = -1;
    if (diss_timer < 0)
    {
        diss_timer = register_timer("/evol/dissipation");
    }

    if (gfs->gs < DISSIP5_HALF)
    {
        fprintf(stderr, "Ghost size >= 3 required for 5th-order dissipation\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Skip the ghost layer (width gs) as well as keeping the 5-point
     * stencil in bounds (half-width DISSIP5_HALF).  gs >= DISSIP5_HALF
     * is asserted above, so this is just gs in practice, but the max()
     * keeps it correct if that ever changes. */
    const int64_t lo = (gfs->gs > DISSIP5_HALF) ? gfs->gs : DISSIP5_HALF;

    BEGIN_TIMER(diss_timer)
    {
        for (int var = 0; var < gfs->n_evol_vars; ++var)
        {
            double *var_dot = gfs->vars[var]->dot;
            const double *var_new = gfs->vars[var]->new;
            for (int64_t k = lo; k < nz - lo; k++)
            {
                for (int64_t j = lo; j < ny - lo; j++)
                {
                    for (int64_t i = lo; i < nx - lo; i++)
                    {
                        const int64_t ijk = ijk_indx(i, j, k, gfs);
                        var_dot[ijk] += coefx * DISSIP_any_5(var_new, ijk, di) +
                                      coefy * DISSIP_any_5(var_new, ijk, dj) +
                                      coefz * DISSIP_any_5(var_new, ijk, dk);
                    }
                }
            }
        }
    }
    END_TIMER(diss_timer)
}
