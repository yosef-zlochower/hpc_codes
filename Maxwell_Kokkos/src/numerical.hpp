#ifndef NUMERICAL_HPP
#define NUMERICAL_HPP
#include "gf.hpp"

/* Adds Kreiss-Oliger 5th-order dissipation to dot[ijk] using state[ijk].
 * Must be called after the PDE RHS has been written into K[kidx]; the
 * dissipation is added on top, in place. */
void apply_dissipation(NGFS *gfs, double diss_coeff, int kidx);

#endif
