#include "numerical.hpp"
#include "derivatives.hpp"
#include "timer.h"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cstdio>

#define DISSIP5_HALF 3

void apply_dissipation(NGFS *gfs, double diss_coeff, int kidx)
{
    const int64_t nx = gfs->nx;
    const int64_t ny = gfs->ny;
    const int64_t nz = gfs->nz;
    const double  dx = gfs->dx;
    const double  dy = gfs->dy;
    const double  dz = gfs->dz;
    const double coefx = -diss_coeff / dx;
    const double coefy = -diss_coeff / dy;
    const double coefz = -diss_coeff / dz;

    const int64_t di = 1;
    const int64_t dj = nx;
    const int64_t dk = nx * ny;

    static int diss_timer = -1;
    if (diss_timer < 0) diss_timer = register_timer("/evol/dissipation");

    if (gfs->gs < DISSIP5_HALF)
    {
        std::fprintf(stderr,
                     "Ghost size >= 3 required for 5th-order dissipation\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    using Range3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
    Range3D pol({DISSIP5_HALF,        DISSIP5_HALF,        DISSIP5_HALF},
                {nx - DISSIP5_HALF,   ny - DISSIP5_HALF,   nz - DISSIP5_HALF});

    BEGIN_TIMER(diss_timer)
    {
        for (int v = 0; v < gfs->n_evol_vars; ++v)
        {
            double *var_dot = gfs->evol[v].K[kidx].data();
            const double *var_new = gfs->evol[v].state.data();
            Kokkos::parallel_for("apply_dissipation", pol,
                KOKKOS_LAMBDA(int64_t i, int64_t j, int64_t k) {
                    const int64_t ijk = i + j * nx + k * nx * ny;
                    var_dot[ijk] += coefx * DISSIP_any_5(var_new, ijk, di) +
                                    coefy * DISSIP_any_5(var_new, ijk, dj) +
                                    coefz * DISSIP_any_5(var_new, ijk, dk);
                });
        }
    }
    END_TIMER(diss_timer)
}
