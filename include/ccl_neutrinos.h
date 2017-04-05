//#pragma once
#include "ccl_core.h"
#include "gsl/gsl_spline.h"

// maximum number of species
#define CCL_MAX_NU_SPECIES 3
// limits for the precomputed spline of phase
// space diagram in MNU/T
#define CCL_NU_MNUT_MIN 1e-4
#define CCL_NU_MNUT_MAX 1e4
// and number of points
#define CCL_NU_MNUT_N 1000

// precalculate the phase space integral
gsl_spline* calculate_nu_phasespace_spline(int *status);

// Returns density of one neutrino species at a scale factor a.
double Omeganuh2 (double a, double Neff, double mnu, double TCMB, gsl_spline* psi);

