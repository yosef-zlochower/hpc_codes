#ifndef ANALYTIC_PARAMETERS_H
#define ANALYTIC_PARAMETERS_H
/* C-compatible POD parameter structs for the analytic source models.
 * Shared by maxwell_parameters.h (where they sit inside maxwell_param_st
 * as per-source sub-tables) and by analytic_solutions.hpp (where the
 * device-callable evaluators consume them).
 *
 * Keep this header free of Kokkos / C++ so that the parameter-parsing
 * layer (parameter.cpp, maxwell_parameters.cpp) can include the public
 * parameter ABI without dragging in Kokkos headers. */

struct eb_st
{
    double Dx;
    double Dy;
    double Dz;
    double Bx;
    double By;
    double Bz;
    double PsiD;
    double PsiB;
    double rho;
};

struct plane_wave_params
{
    double ax;
    double ay;
    double k;
    double bump_a;
    double bump_b;
};

struct gaussian_beam_params
{
    double w0;
    double z_waist;
    double k;
    double amplitude;
    double ramp_a;
    double ramp_b;
};

struct te_waveguide_mode_params
{
    int l;
    int m;
    int n;
};

struct analytic_params_st
{
    double                          t0;
    struct plane_wave_params        plane_wave;
    struct gaussian_beam_params     gaussian_beam;
    struct te_waveguide_mode_params te_waveguide_mode;
};

#endif /* ANALYTIC_PARAMETERS_H */
