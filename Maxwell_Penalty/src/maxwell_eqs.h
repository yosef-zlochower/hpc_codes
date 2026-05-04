#ifndef MAXWELL_EQS_H

#include "gf.h"
#include "analytic_solutions.h"
#include "parameter.h"


/* Storage slots in gfs->vars[] (evolved) and gfs->auxvars[] (auxiliary).
 * The slot enums are the single source of truth for variable order; the
 * name tables below use designated initialisers keyed by the enum so
 * that reordering the enum entries reorders storage consistently. */
enum evolved_slot { DX_SLOT = 0, DY_SLOT, DZ_SLOT, BX_SLOT, BY_SLOT,
                        BZ_SLOT, PSID_SLOT, PSIB_SLOT, RHO_SLOT, N_EVOL };

enum auxilliary_slot { IEPS_SLOT = 0, IMU_SLOT, SIGMA_SLOT, CD_SLOT, CB_SLOT,
                       N_AUX };

/* Name for each slot, used to tag HDF5 datasets via gf_rename. */
extern const char *const evolved_field_names[N_EVOL];
extern const char *const aux_field_names[N_AUX];

/* Extended Maxwell system with constraint-damping fields.
 * 9 evolved variables:
 *   Dx, Dy, Dz      (electric displacement)
 *   Bx, By, Bz      (magnetic field)
 *   PsiD, PsiB       (constraint-damping scalars)
 *   rho              (charge density)
 * The macro below provides convenient pointers to each variable's
 * current state (new), time derivative (dot), grid dimensions,
 * ghost size, and grid spacing.  The unused attribute suppresses
 * warnings for variables not referenced in a given function.
 */
#define DECLARE_EVOLVED_VARS(_ptr_to_gfs)                                      \
    double __attribute__((unused)) *Dx = (_ptr_to_gfs)->vars[DX_SLOT]->new;          \
    double __attribute__((unused)) *Dy = (_ptr_to_gfs)->vars[DY_SLOT]->new;          \
    double __attribute__((unused)) *Dz = (_ptr_to_gfs)->vars[DZ_SLOT]->new;          \
    double __attribute__((unused)) *Bx = (_ptr_to_gfs)->vars[BX_SLOT]->new;          \
    double __attribute__((unused)) *By = (_ptr_to_gfs)->vars[BY_SLOT]->new;          \
    double __attribute__((unused)) *Bz = (_ptr_to_gfs)->vars[BZ_SLOT]->new;          \
    double __attribute__((unused)) *PsiD = (_ptr_to_gfs)->vars[PSID_SLOT]->new;        \
    double __attribute__((unused)) *PsiB = (_ptr_to_gfs)->vars[PSIB_SLOT]->new;        \
    double __attribute__((unused)) *rho = (_ptr_to_gfs)->vars[RHO_SLOT]->new;         \
    double __attribute__((unused)) *dotDx = (_ptr_to_gfs)->vars[DX_SLOT]->dot;       \
    double __attribute__((unused)) *dotDy = (_ptr_to_gfs)->vars[DY_SLOT]->dot;       \
    double __attribute__((unused)) *dotDz = (_ptr_to_gfs)->vars[DZ_SLOT]->dot;       \
    double __attribute__((unused)) *dotBx = (_ptr_to_gfs)->vars[BX_SLOT]->dot;       \
    double __attribute__((unused)) *dotBy = (_ptr_to_gfs)->vars[BY_SLOT]->dot;       \
    double __attribute__((unused)) *dotBz = (_ptr_to_gfs)->vars[BZ_SLOT]->dot;       \
    double __attribute__((unused)) *dotPsiD = (_ptr_to_gfs)->vars[PSID_SLOT]->dot;     \
    double __attribute__((unused)) *dotPsiB = (_ptr_to_gfs)->vars[PSIB_SLOT]->dot;     \
    double __attribute__((unused)) *dotrho = (_ptr_to_gfs)->vars[RHO_SLOT]->dot;      \
    const int64_t __attribute__((unused)) nx = (_ptr_to_gfs)->nx;               \
    const int64_t __attribute__((unused)) ny = (_ptr_to_gfs)->ny;              \
    const int64_t __attribute__((unused)) nz = (_ptr_to_gfs)->nz;             \
    const int __attribute__((unused)) gs = (_ptr_to_gfs)->gs;                  \
    const double __attribute__((unused)) dx = (_ptr_to_gfs)->dx;               \
    const double __attribute__((unused)) dy = (_ptr_to_gfs)->dy;               \
    const int64_t __attribute__((unused)) di = 1;                              \
    const int64_t __attribute__((unused)) dj = nx;                             \
    const int64_t __attribute__((unused)) dk = nx * ny;                        \
    const double __attribute__((unused)) dz = (_ptr_to_gfs)->dz;

/* 5 auxiliary (non-evolved) variables:
 *   ieps, imu, sigma  (material properties: 1/epsilon, 1/mu, conductivity)
 *   cD, cB            (constraint violations: div D - 4*pi*rho, div B)
 */
#define DECLARE_AUX_VARS(_ptr_to_gfs)                                          \
    double __attribute__((unused)) *ieps = (_ptr_to_gfs)->auxvars[IEPS_SLOT]->new;     \
    double __attribute__((unused)) *imu = (_ptr_to_gfs)->auxvars[IMU_SLOT]->new;      \
    double __attribute__((unused)) *sigma = (_ptr_to_gfs)->auxvars[SIGMA_SLOT]->new;    \
    double __attribute__((unused)) *cD = (_ptr_to_gfs)->auxvars[CD_SLOT]->new;       \
    double __attribute__((unused)) *cB = (_ptr_to_gfs)->auxvars[CB_SLOT]->new;

void maxwell_eq_time_deriv(struct ngfs *gfs, const double t);

void set_initial_data(struct ngfs *gfs, const double t);

void maxwell_constraints(struct ngfs *gfs);

double l2_error_analytic(struct ngfs *gfs, const double t);


/* Global parameter structs, populated from TOML file in main() */
extern struct maxwell_param_st maxwell_params;

extern struct analytic_params_st analytic_params;

#define MAXWELL_EQS_H
#endif


