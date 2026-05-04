#ifndef MAXWELL_EQS_HPP
#define MAXWELL_EQS_HPP
#include "gf.hpp"
#include "analytic_solutions.hpp"
#include "maxwell_parameters.h"

/* Single source of truth for evolved/auxiliary variable order.
 * Reordering an entry reorders storage, the name table, and the
 * DECLARE_*_VARS macros below — nothing else needs to track the change. */
enum evolved_slot { DX_SLOT = 0, DY_SLOT, DZ_SLOT, BX_SLOT, BY_SLOT,
                    BZ_SLOT, PSID_SLOT, PSIB_SLOT, RHO_SLOT, N_EVOL };

enum auxilliary_slot { IEPS_SLOT = 0, IMU_SLOT, SIGMA_SLOT, CD_SLOT, CB_SLOT,
                       N_AUX };

extern const char *const evolved_field_names[N_EVOL];
extern const char *const aux_field_names    [N_AUX];

/* Materialise raw double* aliases for every evolved field's state and
 * for the K[kidx] buffer that the current RK4 stage writes into. The
 * pointers are captured by value into KOKKOS_LAMBDAs, so they are
 * well-defined on host and on device alike. nx, ny, nz, dx, dy, dz,
 * di, dj, dk match the C version exactly; the simple_maxwell.h macros
 * use them through DIFFX/DIFFY/DIFFZ. */
/* The [[maybe_unused]] attribute silences unused-variable warnings on
 * any name a particular caller doesn't reference (e.g. set_initial_data
 * doesn't use the dot* pointers; maxwell_constraints doesn't use the
 * SAT scaling; etc.) — same pattern the C-version macros use with
 * __attribute__((unused)). */
#define DECLARE_EVOLVED_VARS(_gfs, _kidx)                                       \
    [[maybe_unused]] double *Dx   = (_gfs)->evol[DX_SLOT  ].state.data();       \
    [[maybe_unused]] double *Dy   = (_gfs)->evol[DY_SLOT  ].state.data();       \
    [[maybe_unused]] double *Dz   = (_gfs)->evol[DZ_SLOT  ].state.data();       \
    [[maybe_unused]] double *Bx   = (_gfs)->evol[BX_SLOT  ].state.data();       \
    [[maybe_unused]] double *By   = (_gfs)->evol[BY_SLOT  ].state.data();       \
    [[maybe_unused]] double *Bz   = (_gfs)->evol[BZ_SLOT  ].state.data();       \
    [[maybe_unused]] double *PsiD = (_gfs)->evol[PSID_SLOT].state.data();       \
    [[maybe_unused]] double *PsiB = (_gfs)->evol[PSIB_SLOT].state.data();       \
    [[maybe_unused]] double *rho  = (_gfs)->evol[RHO_SLOT ].state.data();       \
    [[maybe_unused]] double *dotDx   = (_gfs)->evol[DX_SLOT  ].K[(_kidx)].data();\
    [[maybe_unused]] double *dotDy   = (_gfs)->evol[DY_SLOT  ].K[(_kidx)].data();\
    [[maybe_unused]] double *dotDz   = (_gfs)->evol[DZ_SLOT  ].K[(_kidx)].data();\
    [[maybe_unused]] double *dotBx   = (_gfs)->evol[BX_SLOT  ].K[(_kidx)].data();\
    [[maybe_unused]] double *dotBy   = (_gfs)->evol[BY_SLOT  ].K[(_kidx)].data();\
    [[maybe_unused]] double *dotBz   = (_gfs)->evol[BZ_SLOT  ].K[(_kidx)].data();\
    [[maybe_unused]] double *dotPsiD = (_gfs)->evol[PSID_SLOT].K[(_kidx)].data();\
    [[maybe_unused]] double *dotPsiB = (_gfs)->evol[PSIB_SLOT].K[(_kidx)].data();\
    [[maybe_unused]] double *dotrho  = (_gfs)->evol[RHO_SLOT ].K[(_kidx)].data();\
    [[maybe_unused]] const int64_t nx = (_gfs)->nx;                             \
    [[maybe_unused]] const int64_t ny = (_gfs)->ny;                             \
    [[maybe_unused]] const int64_t nz = (_gfs)->nz;                             \
    [[maybe_unused]] const int     gs = (_gfs)->gs;                             \
    [[maybe_unused]] const double  dx = (_gfs)->dx;                             \
    [[maybe_unused]] const double  dy = (_gfs)->dy;                             \
    [[maybe_unused]] const double  dz = (_gfs)->dz;                             \
    [[maybe_unused]] const int64_t di = 1;                                      \
    [[maybe_unused]] const int64_t dj = nx;                                     \
    [[maybe_unused]] const int64_t dk = nx * ny

#define DECLARE_AUX_VARS(_gfs)                                                  \
    [[maybe_unused]] double *ieps  = (_gfs)->aux[IEPS_SLOT ].state.data();      \
    [[maybe_unused]] double *imu   = (_gfs)->aux[IMU_SLOT  ].state.data();      \
    [[maybe_unused]] double *sigma = (_gfs)->aux[SIGMA_SLOT].state.data();      \
    [[maybe_unused]] double *cD    = (_gfs)->aux[CD_SLOT   ].state.data();      \
    [[maybe_unused]] double *cB    = (_gfs)->aux[CB_SLOT   ].state.data()

void   maxwell_eq_time_deriv(NGFS *gfs, double t, int kidx);
void   set_initial_data    (NGFS *gfs, double t);
void   maxwell_constraints (NGFS *gfs);
double l2_error_analytic   (NGFS *gfs, double t);

extern struct maxwell_param_st   maxwell_params;
extern struct analytic_params_st analytic_params;

#endif /* MAXWELL_EQS_HPP */
