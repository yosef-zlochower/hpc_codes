#ifndef RK4_HPP
#define RK4_HPP
#include "gf.hpp"

/* Time-derivative function signature for RK4_Step.
 *
 * `kidx` selects which K buffer (0..3 = K1..K4) to write the RHS into.
 * The C version of this code re-pointed each var's `dot` pointer at K1
 * .. K4 between stages; in the Kokkos port the K-buffer index is passed
 * through explicitly so the kernels know where to write without relying
 * on per-View pointer aliasing. */
using dotfunction = void (*)(NGFS *gfs, double t, int kidx);

int RK4_Step(NGFS *gfs, double t0, double dt, dotfunction time_deriv);

#endif
