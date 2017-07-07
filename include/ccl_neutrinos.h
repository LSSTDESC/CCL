/** @file */

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
#define NU_CONST (8. * pow(M_PI,5) *pow((KBOLTZ/ HPLANCK),3)* KBOLTZ/(15. *pow( CLIGHT,3))* (8. * M_PI * GNEWT) / (3. * 100.*100.*1000.*1000. /MPC_TO_METER /MPC_TO_METER  * CLIGHT * CLIGHT))

/**
 * Spline for the phasespace integral required for getting the fractional energy density of massive neutrinos.
 * Returns a gsl spline for the phase space integral needed for massive neutrinos.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return spl, the gsl spline for the phasespace integral required for massive neutrino calculations.
 */
gsl_spline* calculate_nu_phasespace_spline(int *status);

/** 
 * Returns density of one neutrino species at a scale factor a. 
 * Users are encouraged to access this quantity via the function ccl_omega_x.
 * @param a Scale factor
 * @param Neff The effective number of species with neutrino mass mnu.
 * @param mnu Neutrino mass (can be 0).
 * @param TCMB Temperature of the CMB
 * @param accel - Interpolation accelerator to be used with phasespace spline. If not set yet, pass NULL.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return OmNuh2 Fractional energy density of neutrions with mass mnu, multiplied by h squared. 
 */
double Omeganuh2 (double a, double Neff, double mnu, double TCMB, gsl_interp_accel* accel, int * status);
