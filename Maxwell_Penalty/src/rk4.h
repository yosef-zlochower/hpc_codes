#ifndef RK4_H
#define RK4_H
#include "gf.h"

typedef void (*dotfunction)(struct ngfs *gfs, const double t0);

int RK4_Step(struct ngfs *gfs, const double t0, const double dt,
             dotfunction time_deriv);
#endif
