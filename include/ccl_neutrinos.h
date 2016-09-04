#pragma once
#include "gsl/gsl_spline.h"

// maximum number of species
#define CCL_MAX_NU_SPECIES 3
// limits for the precomputed spline of phase
// space diagram in MNU/T
#define CCL_NU_MNUT_MIN 1e-4
#define CCL_NU_MNUT_MAX 1e4
// and number of points
#define CCL_NU_MNUT_N 100

// precalculate the phase space integral
gsl_spline* ccl_calculate_nu_phasespace_spline();

// returns the value of the neutrino phase-space integral * 7/8 fermion factor
double ccl_nu_phasespace_intg(gsl_spline* spl, double mnuOT);

  
// returns density if one neutrino species at a scale factor a, given this particular
// species' Neff and sum_mnu and TCMB
// the output is neutrino density in the units of Omega_nu_h^2 today
double OmegaNuh2 (double a, double Neff, double mnu, double TCMB, gsl_spline* psi);

