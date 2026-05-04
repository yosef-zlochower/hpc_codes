#ifndef DERIVATIVES_H
#include <stdint.h>
#include <Kokkos_Core.hpp>

/* First order accurate first-derivatives */
KOKKOS_INLINE_FUNCTION double D1UP(const double *f, int64_t i, int64_t di, double h)
{
    const double ooh = 1.0 / (h);
    return (-f[i] + f[di + i]) * ooh;
}
KOKKOS_INLINE_FUNCTION double D1DN(const double *f, int64_t i, int64_t di, double h)
{
    const double ooh = 1.0 / (h);
    return (-f[i - di] + f[i]) * ooh;
}

/* Second order accurate first-derivative operators */
KOKKOS_INLINE_FUNCTION double D2CEN(const double *f, int64_t i, int64_t di, double h)
{
    const double oo2h = 1.0 / (2 * h);
    return (-f[-di + i] + f[di + i]) * oo2h;
}
KOKKOS_INLINE_FUNCTION double D2UP(const double *f, int64_t i, int64_t di, double h)
{
    const double oo2h = 1.0 / (2 * h);
    return (-3 * f[i] + 4 * f[i + di] - f[i + 2 * di]) * oo2h;
}
KOKKOS_INLINE_FUNCTION double D2DN(const double *f, int64_t i, int64_t di, double h)
{
    const double oo2h = 1.0 / (2 * h);
    return (f[i - 2 * di] - 4 * f[i - di] + 3 * f[i]) * oo2h;
}

/* Fourth order accurate first-derivative operators.
 * Reads  f[i - 2*di], f[i - di], f[i + di], f[i + 2*di]  — i.e. half-width 2. */
#define D4CEN_HALF 2
KOKKOS_INLINE_FUNCTION double D4CEN(const double *f, int64_t i, int64_t di, double h)
{
    const double oo12h = 1.0 / (12 * h);
    return (f[-2 * di + i] - 8 * f[-di + i] + 8 * f[di + i] - f[2 * di + i]) *
           oo12h;
}

/* Sixth order accurate first-derivative operators */
KOKKOS_INLINE_FUNCTION double D6CEN(const double *f, int64_t i, int64_t di, double h)
{
    const double oo60h = 1.0 / (60 * h);
    return (-f[-3 * di + i] + 9 * f[-2 * di + i] -
            9 * (5 * f[-di + i] - 5 * f[di + i] + f[2 * di + i]) +
            f[3 * di + i]) *
           oo60h;
}

/* Eighth order accurate first-derivative operators */
KOKKOS_INLINE_FUNCTION double D8CEN(const double *f, int64_t i, int64_t di, double h)
{
    const double oo840h = 1.0 / (840 * h);
    return (3 * f[-4 * di + i] - 32 * f[-3 * di + i] +
            168 * (f[-2 * di + i] - 4 * f[-di + i] + 4 * f[di + i] -
                   f[2 * di + i]) +
            32 * f[3 * di + i] - 3 * f[4 * di + i]) *
           oo840h;
}

/* Second order accurate second-derivative operators */
KOKKOS_INLINE_FUNCTION double D2xy(const double *f, int64_t i, int64_t di, int64_t dj, double dx, double dy)
{
    const double dxdy = dx * dy;
    const double oo4dxdy = 1.0 / (4 * dxdy);
    return (f[-di - dj + i] - f[di - dj + i] - f[-di + dj + i] +
            f[di + dj + i]) *
           oo4dxdy;
}

KOKKOS_INLINE_FUNCTION double D2xx(const double *f, int64_t i, int64_t di, double dx)
{
    const double dx2 = dx * dx;
    const double oodx2 = 1.0 / dx2;
    return (-2 * f[i] + f[-di + i] + f[di + i]) * oodx2;
}

/* Fourth order accurate second-derivative operators */

KOKKOS_INLINE_FUNCTION double D4xy(const double *f, int64_t i, int64_t di, int64_t dj, double dx, double dy)
{
    const double dxdy = dx * dy;
    const double oo144dxdy = 1.0 / (144 * dxdy);
    return (-8 * f[-di - 2 * dj + i] + 8 * f[di - 2 * dj + i] -
            f[2 * di - 2 * dj + i] - 8 * f[-2 * di - dj + i] +
            64 * f[-di - dj + i] - 64 * f[di - dj + i] +
            8 * f[2 * di - dj + i] + 8 * f[-2 * di + dj + i] -
            64 * f[-di + dj + i] + 64 * f[di + dj + i] -
            8 * f[2 * di + dj + i] - f[-2 * di + 2 * dj + i] +
            8 * f[-di + 2 * dj + i] - 8 * f[di + 2 * dj + i] +
            f[-2 * (di + dj) + i] + f[2 * (di + dj) + i]) *
           oo144dxdy;
}

KOKKOS_INLINE_FUNCTION double D4xx(const double *f, int64_t i, int64_t di, double dx)
{
    const double dx2 = dx * dx;
    const double oo12dx2 = 1.0 / (12 * dx2);
    return (-30 * f[i] - f[-2 * di + i] + 16 * (f[-di + i] + f[di + i]) -
            f[2 * di + i]) *
           oo12dx2;
}

/* Sixth order accurate second-derivative operators */
KOKKOS_INLINE_FUNCTION double D6xy(const double *f, int64_t i, int64_t di, int64_t dj, double dx, double dy)
{
    const double dxdy = dx * dy;
    const double oo3600dxdy = 1.0 / (3600 * dxdy);
    return (-9 * f[-2 * di - 3 * dj + i] + 45 * f[-di - 3 * dj + i] -
            45 * f[di - 3 * dj + i] + 9 * f[2 * di - 3 * dj + i] -
            f[3 * di - 3 * dj + i] - 9 * f[-3 * di - 2 * dj + i] -
            405 * f[-di - 2 * dj + i] + 405 * f[di - 2 * dj + i] -
            81 * f[2 * di - 2 * dj + i] + 9 * f[3 * di - 2 * dj + i] +
            45 * f[-3 * di - dj + i] - 405 * f[-2 * di - dj + i] +
            2025 * f[-di - dj + i] - 2025 * f[di - dj + i] +
            405 * f[2 * di - dj + i] - 45 * f[3 * di - dj + i] -
            45 * f[-3 * di + dj + i] + 405 * f[-2 * di + dj + i] -
            2025 * f[-di + dj + i] + 2025 * f[di + dj + i] -
            405 * f[2 * di + dj + i] + 45 * f[3 * di + dj + i] +
            9 * f[-3 * di + 2 * dj + i] - 81 * f[-2 * di + 2 * dj + i] +
            405 * f[-di + 2 * dj + i] - 405 * f[di + 2 * dj + i] -
            9 * f[3 * di + 2 * dj + i] - f[-3 * di + 3 * dj + i] +
            9 * f[-2 * di + 3 * dj + i] - 45 * f[-di + 3 * dj + i] +
            45 * f[di + 3 * dj + i] - 9 * f[2 * di + 3 * dj + i] +
            f[-3 * (di + dj) + i] +
            81 * (f[-2 * (di + dj) + i] + f[2 * (di + dj) + i]) +
            f[3 * (di + dj) + i]) *
           oo3600dxdy;
}

KOKKOS_INLINE_FUNCTION double D6xx(const double *f, int64_t i, int64_t di, double dx)
{
    const double dx2 = dx * dx;
    const double oo180dx2 = 1.0 / (180 * dx2);
    return (-490 * f[i] + 2 * f[-3 * di + i] -
            27 * (f[-2 * di + i] - 10 * (f[-di + i] + f[di + i]) +
                  f[2 * di + i]) +
            2 * f[3 * di + i]) *
           oo180dx2;
}

/* Eighth order accurate second-derivative operators */
KOKKOS_INLINE_FUNCTION double D8xy(const double *f, int64_t i, int64_t di, int64_t dj, double dx, double dy)
{
    const double dxdy = dx * dy;
    const double oo705600dxdy = 1.0 / (705600 * dxdy);
    return (-96 * f[-3 * di - 4 * dj + i] + 504 * f[-2 * di - 4 * dj + i] -
            2016 * f[-di - 4 * dj + i] + 2016 * f[di - 4 * dj + i] -
            504 * f[2 * di - 4 * dj + i] + 96 * f[3 * di - 4 * dj + i] -
            9 * f[4 * di - 4 * dj + i] - 96 * f[-4 * di - 3 * dj + i] -
            5376 * f[-2 * di - 3 * dj + i] + 21504 * f[-di - 3 * dj + i] -
            21504 * f[di - 3 * dj + i] + 5376 * f[2 * di - 3 * dj + i] -
            1024 * f[3 * di - 3 * dj + i] + 96 * f[4 * di - 3 * dj + i] +
            504 * f[-4 * di - 2 * dj + i] - 5376 * f[-3 * di - 2 * dj + i] -
            112896 * f[-di - 2 * dj + i] + 112896 * f[di - 2 * dj + i] -
            28224 * f[2 * di - 2 * dj + i] + 5376 * f[3 * di - 2 * dj + i] -
            504 * f[4 * di - 2 * dj + i] - 2016 * f[-4 * di - dj + i] +
            21504 * f[-3 * di - dj + i] - 112896 * f[-2 * di - dj + i] +
            451584 * f[-di - dj + i] - 451584 * f[di - dj + i] +
            112896 * f[2 * di - dj + i] - 21504 * f[3 * di - dj + i] +
            2016 * f[4 * di - dj + i] + 2016 * f[-4 * di + dj + i] -
            21504 * f[-3 * di + dj + i] + 112896 * f[-2 * di + dj + i] -
            451584 * f[-di + dj + i] + 451584 * f[di + dj + i] -
            112896 * f[2 * di + dj + i] + 21504 * f[3 * di + dj + i] -
            2016 * f[4 * di + dj + i] - 504 * f[-4 * di + 2 * dj + i] +
            5376 * f[-3 * di + 2 * dj + i] - 28224 * f[-2 * di + 2 * dj + i] +
            112896 * f[-di + 2 * dj + i] - 112896 * f[di + 2 * dj + i] -
            5376 * f[3 * di + 2 * dj + i] + 504 * f[4 * di + 2 * dj + i] +
            96 * f[-4 * di + 3 * dj + i] - 1024 * f[-3 * di + 3 * dj + i] +
            5376 * f[-2 * di + 3 * dj + i] - 21504 * f[-di + 3 * dj + i] +
            21504 * f[di + 3 * dj + i] - 5376 * f[2 * di + 3 * dj + i] -
            96 * f[4 * di + 3 * dj + i] - 9 * f[-4 * di + 4 * dj + i] +
            96 * f[-3 * di + 4 * dj + i] - 504 * f[-2 * di + 4 * dj + i] +
            2016 * f[-di + 4 * dj + i] - 2016 * f[di + 4 * dj + i] +
            504 * f[2 * di + 4 * dj + i] - 96 * f[3 * di + 4 * dj + i] +
            9 * f[-4 * (di + dj) + i] +
            64 * (16 * f[-3 * (di + dj) + i] +
                  441 * (f[-2 * (di + dj) + i] + f[2 * (di + dj) + i]) +
                  16 * f[3 * (di + dj) + i]) +
            9 * f[4 * (di + dj) + i]) *
           oo705600dxdy;
}

KOKKOS_INLINE_FUNCTION double D8xx(const double *f, int64_t i, int64_t di, double dx)
{
    const double dx2 = dx * dx;
    const double oo5040dx2 = 1.0 / (5040 * dx2);
    return (-14350 * f[i] - 9 * f[-4 * di + i] +
            16 * (8 * f[-3 * di + i] -
                  63 * (f[-2 * di + i] - 8 * (f[-di + i] + f[di + i]) +
                        f[2 * di + i]) +
                  8 * f[3 * di + i]) -
            9 * f[4 * di + i]) *
           oo5040dx2;
}

/***********************************************************
** SBP 4-2 BOUNDARY CLOSURE OPERATORS                     **
** (4th-order interior, 2nd-order boundary, Strand 1994)  **
** D = H^{-1} Q,  H = h*diag(17/48, 59/48, 43/48, 49/48, 1, ...) **
************************************************************/

/* The SBP 4-2 scheme uses one-sided closures on the first/last
 * SBP42_CLOSURE_ROWS rows at each physical boundary (rows 0..3 and
 * N-4..N-1). The interior rows between them use the 4th-order centred
 * stencil D4CEN. */
#define SBP42_CLOSURE_ROWS 4

/* SBP 4-2 norm-inverse diagonal entries (H_ii/h)^{-1}. The SAT penalty
 * applied at the exact boundary point scales as tau * SBP42_HINV_0 / h.
 * (Index 0 is the only one the SAT uses; the interior rows 1..3 receive
 * the SBP closure in their derivative but no SAT term.) */
#define SBP42_HINV_0  (48.0 / 17.0)   /* point 0 and N   */
#define SBP42_HINV_1  (48.0 / 59.0)   /* point 1 and N-1 */
#define SBP42_HINV_2  (48.0 / 43.0)   /* point 2 and N-2 */
#define SBP42_HINV_3  (48.0 / 49.0)   /* point 3 and N-3 */


/* Left boundary row 0: [-24/17, 59/34, -4/17, -3/34] */
KOKKOS_INLINE_FUNCTION double SBP42_L0(const double *f, int64_t i, int64_t dk, double h)
{
    const double ooh = 1.0 / h;
    return (-24.0/17.0 * f[i]
          + 59.0/34.0 * f[i +     dk]
          -  4.0/17.0 * f[i + 2 * dk]
          -  3.0/34.0 * f[i + 3 * dk]) * ooh;
}

/* Left boundary row 1: [-1/2, 0, 1/2] */
KOKKOS_INLINE_FUNCTION double SBP42_L1(const double *f, int64_t i, int64_t dk, double h)
{
    const double oo2h = 1.0 / (2.0 * h);
    return (-f[i - dk] + f[i + dk]) * oo2h;
}

/* Left boundary row 2: [4/43, -59/86, 0, 59/86, -4/43] */
KOKKOS_INLINE_FUNCTION double SBP42_L2(const double *f, int64_t i, int64_t dk, double h)
{
    const double ooh = 1.0 / h;
    return ( 4.0/43.0 * f[i - 2 * dk]
          - 59.0/86.0 * f[i -     dk]
          + 59.0/86.0 * f[i +     dk]
          -  4.0/43.0 * f[i + 2 * dk]) * ooh;
}

/* Left boundary row 3: [3/98, 0, -59/98, 0, 32/49, -4/49] */
KOKKOS_INLINE_FUNCTION double SBP42_L3(const double *f, int64_t i, int64_t dk, double h)
{
    const double ooh = 1.0 / h;
    return ( 3.0/98.0 * f[i - 3 * dk]
          - 59.0/98.0 * f[i -     dk]
          + 32.0/49.0 * f[i +     dk]
          -  4.0/49.0 * f[i + 2 * dk]) * ooh;
}

/* Right boundary row N: [3/34, 4/17, -59/34, 24/17] */
KOKKOS_INLINE_FUNCTION double SBP42_RN(const double *f, int64_t i, int64_t dk, double h)
{
    const double ooh = 1.0 / h;
    return ( 3.0/34.0 * f[i - 3 * dk]
          +  4.0/17.0 * f[i - 2 * dk]
          - 59.0/34.0 * f[i -     dk]
          + 24.0/17.0 * f[i]        ) * ooh;
}

/* Right boundary row N-1: [-1/2, 0, 1/2] */
KOKKOS_INLINE_FUNCTION double SBP42_RNm1(const double *f, int64_t i, int64_t dk, double h)
{
    const double oo2h = 1.0 / (2.0 * h);
    return (-f[i - dk] + f[i + dk]) * oo2h;
}

/* Right boundary row N-2: [4/43, -59/86, 0, 59/86, -4/43] */
KOKKOS_INLINE_FUNCTION double SBP42_RNm2(const double *f, int64_t i, int64_t dk, double h)
{
    const double ooh = 1.0 / h;
    return ( 4.0/43.0 * f[i - 2 * dk]
          - 59.0/86.0 * f[i -     dk]
          + 59.0/86.0 * f[i +     dk]
          -  4.0/43.0 * f[i + 2 * dk]) * ooh;
}

/* Right boundary row N-3: [4/49, -32/49, 0, 59/98, 0, -3/98] */
KOKKOS_INLINE_FUNCTION double SBP42_RNm3(const double *f, int64_t i, int64_t dk, double h)
{
    const double ooh = 1.0 / h;
    return ( 4.0/49.0 * f[i - 2 * dk]
          - 32.0/49.0 * f[i -     dk]
          + 59.0/98.0 * f[i +     dk]
          -  3.0/98.0 * f[i + 3 * dk]) * ooh;
}

/***********************************************************
** Kreiss-Oliger DISIPATION OPERATORS                     **
************************************************************/

/* First order dissipation */
KOKKOS_INLINE_FUNCTION double DISSIP_any_1(const double *f, int64_t i, int64_t di)
{
    return (-2 * f[i] + f[-di + i] + f[di + i]) / 4;
}

/* Third order dissipation */
KOKKOS_INLINE_FUNCTION double DISSIP_any_3(const double *f, int64_t i, int64_t di)
{
    return (6 * f[i] + f[-2 * di + i] - 4 * f[-di + i] - 4 * f[di + i] +
            f[2 * di + i]) /
           16;
}

/* Fifth order dissipation */
KOKKOS_INLINE_FUNCTION double DISSIP_any_5(const double *f, int64_t i, int64_t di)
{
    return (-20 * f[i] + f[-3 * di + i] - 6 * f[-2 * di + i] + 15 * f[-di + i] +
            15 * f[di + i] - 6 * f[2 * di + i] + f[3 * di + i]) /
           64;
}

/* Seventh order dissipation */
KOKKOS_INLINE_FUNCTION double DISSIP_any_7(const double *f, int64_t i, int64_t di)
{
    return (70 * f[i] + f[-4 * di + i] - 8 * f[-3 * di + i] +
            28 * f[-2 * di + i] - 56 * f[-di + i] - 56 * f[di + i] +
            28 * f[2 * di + i] - 8 * f[3 * di + i] + f[4 * di + i]) /
           256;
}

/* Ninth order dissipation */
KOKKOS_INLINE_FUNCTION double DISSIP_any_9(const double *f, int64_t i, int64_t di)
{
    return (-252 * f[i] + f[-5 * di + i] - 10 * f[-4 * di + i] +
            45 * f[-3 * di + i] - 120 * f[-2 * di + i] + 210 * f[-di + i] +
            210 * f[di + i] - 120 * f[2 * di + i] + 45 * f[3 * di + i] -
            10 * f[4 * di + i] + f[5 * di + i]) /
           1024;
}

#define DERIVATIVES_H
#endif
