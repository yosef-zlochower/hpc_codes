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
 * !! Dissipation does not alter the first three and last three gridpoints in
 * !! each direction (on global grid). This is a design choice. 
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

    BEGIN_TIMER(diss_timer)
    {
        for (int var = 0; var < gfs->n_evol_vars; ++var)
        {
            double *var_dot = gfs->vars[var]->dot;
            const double *var_new = gfs->vars[var]->new;
            for (int64_t k = DISSIP5_HALF; k < nz - DISSIP5_HALF; k++)
            {
                for (int64_t j = DISSIP5_HALF; j < ny - DISSIP5_HALF; j++)
                {
                    for (int64_t i = DISSIP5_HALF; i < nx - DISSIP5_HALF; i++)
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
