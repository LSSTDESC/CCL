//#pragma once
#include "ccl_core.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_const_mksa.h"

// maximum number of species
#define CCL_MAX_NU_SPECIES 3
// limits for the precomputed spline of phase
// space diagram in MNU/T
#define CCL_NU_MNUT_MIN 1e-4
#define CCL_NU_MNUT_MAX 500
// and number of points
#define CCL_NU_MNUT_N 1000

// The combination of constants required in Omeganuh2
#define NU_CONST 8. * pow(M_PI,5) *pow((KBOLTZ/ HPLANCK),3)* KBOLTZ/(15. *pow( CLIGHT,3))* (8. * M_PI * GNEWT) / (3. * 100.*100.*1000.*1000. /MPC_TO_METER /MPC_TO_METER  * CLIGHT * CLIGHT)

// precalculate the phase space integral
gsl_spline* calculate_nu_phasespace_spline(int *status);

// Returns density of one neutrino species at a scale factor a.
double Omeganuh2 (double a, double Neff, double mnu, double TCMB, gsl_interp_accel* accel, int * status);

