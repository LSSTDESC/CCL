/** @file */
#ifndef __CCL_NEUTRINOS_H_INCLUDED__
#define __CCL_NEUTRINOS_H_INCLUDED__

#include <gsl/gsl_spline.h>
#include <gsl/gsl_const_mksa.h>

// maximum number of species
#define CCL_MAX_NU_SPECIES 3
// limits for the precomputed spline of phase
// space diagram in MNU/T
#define CCL_NU_MNUT_MIN 1e-4
#define CCL_NU_MNUT_MAX 500
// and number of points
#define CCL_NU_MNUT_N 1000

// The combination of constants required in Omeganuh2
#define NU_CONST ( \
  8. * pow(M_PI,5) *pow((ccl_constants.KBOLTZ/ ccl_constants.HPLANCK),3)* \
  ccl_constants.KBOLTZ/(15. *pow( ccl_constants.CLIGHT,3))* \
  (8. * M_PI * ccl_constants.GNEWT) / \
  (3. * 100.*100.*1000.*1000. /ccl_constants.MPC_TO_METER /ccl_constants.MPC_TO_METER  * ccl_constants.CLIGHT * ccl_constants.CLIGHT))

CCL_BEGIN_DECLS

typedef enum ccl_neutrino_mass_splits{
  ccl_nu_normal=0,
  ccl_nu_inverted=1,
  ccl_nu_equal=2,
  ccl_nu_sum=3,
  ccl_nu_single=4
} ccl_neutrino_mass_splits;

/**
 * Returns density of one neutrino species at a scale factor a.
 * Users are encouraged to access this quantity via the function ccl_omega_x.
 * @param a Scale factor
 * @param Neff The effective number of species with neutrino mass mnu.
 * @param mnu Pointer to array containing neutrino mass (can be 0).
 * @param T_CMB Temperature of the CMB
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return OmNuh2 Fractional energy density of neutrions with mass mnu, multiplied by h squared.
 */
double ccl_Omeganuh2(double a, int N_nu_mass, double* mnu, double T_CMB, int * status);

/**
 * Returns mass of one neutrino species at a scale factor a.
 * @param a Scale factor
 * @param Neff The effective number of species with neutrino mass mnu.
 * @param OmNuh2 Fractional energy density of neutrions with mass mnu, multiplied by h squared. (can be 0).
 * @param T_CMB Temperature of the CMB
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return Mnu Neutrino mass [eV].
 */
double* ccl_nu_masses(double OmNuh2, ccl_neutrino_mass_splits mass_split, double T_CMB, int * status);

CCL_END_DECLS
#endif
