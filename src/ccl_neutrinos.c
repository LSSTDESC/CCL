#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_const_mksa.h>
#include <gsl/gsl_roots.h>

#include "ccl.h"

// Global variable to hold the neutrino phase-space spline
gsl_spline* nu_spline = NULL;

// these are NOT adjustable
// this phase space integral is only done once and the following is accurate
// enough according to tests done by the devs
/**
 * Absolute precision in neutrino root finding
 */
#define GSL_EPSABS_NU 1E-7

/**
 * Relative precision in neutrino root finding
 */
#define GSL_EPSREL_NU 1E-7

/**
 * Number of iterations for neutrino root finding
 */
#define GSL_N_ITERATION_NU 1000



/* ------- ROUTINE: nu_integrand ------
INPUTS: x: dimensionless momentum, *r: pointer to a dimensionless mass / temperature
TASK: Integrand of phase-space massive neutrino integral
*/
static double nu_integrand(double x, void *r) {
  double rat = *((double*)(r));
  double x2 = x*x;
  return sqrt(x2 + rat*rat) / (exp(x)+1.0) * x2;
}

/* ------- ROUTINE: ccl_calculate_nu_phasespace_spline ------
TASK: Get the spline of the result of the phase-space integral required for massive neutrinos.
*/

static gsl_spline* calculate_nu_phasespace_spline(int *status) {
  int N = CCL_NU_MNUT_N;
  double *mnut = NULL;
  double *y = NULL;
  gsl_spline* spl = NULL;
  gsl_integration_cquad_workspace * workspace = NULL;
  int stat = 0, gslstatus;
  gsl_function F;

  mnut = ccl_linear_spacing(log(CCL_NU_MNUT_MIN), log(CCL_NU_MNUT_MAX), N);
  y = malloc(sizeof(double)*CCL_NU_MNUT_N);
  if ((y == NULL) || (mnut == NULL)) {
    // Not setting a status_message here because we can't easily pass a
    // cosmology to this function - message printed in ccl_error.c.
    *status = CCL_ERROR_NU_INT;
  }

  if (*status == 0) {
    workspace = gsl_integration_cquad_workspace_alloc(GSL_N_ITERATION_NU);
    if (workspace == NULL)
      *status = CCL_ERROR_NU_INT;
  }

  if (*status == 0) {
    F.function = &nu_integrand;
    for (int i=0; i < CCL_NU_MNUT_N; i++) {
      double mnut_ = exp(mnut[i]);
      F.params = &(mnut_);
      gslstatus = gsl_integration_cquad(&F, 0, 1000.0,
                                        GSL_EPSABS_NU,
                                        GSL_EPSREL_NU,
                                        workspace, &y[i], NULL, NULL);
      if (gslstatus != GSL_SUCCESS) {
        ccl_raise_gsl_warning(gslstatus, "ccl_neutrinos.c: calculate_nu_phasespace_spline():");
        stat |= gslstatus;
      }
    }

    double renorm = 1./y[0];
    for (int i=0; i < CCL_NU_MNUT_N; i++)
      y[i] *= renorm;

    if (stat) {
      *status = CCL_ERROR_NU_INT;
    }
  }

  if (*status == 0) {
    spl = gsl_spline_alloc(gsl_interp_akima, CCL_NU_MNUT_N);
    if (spl == NULL)
      *status = CCL_ERROR_NU_INT;
  }

  if (*status == 0) {
    stat |= gsl_spline_init(spl, mnut, y, CCL_NU_MNUT_N);
    if (stat) {
      ccl_raise_gsl_warning(gslstatus, "ccl_neutrinos.c: calculate_nu_phasespace_spline():");
      *status = CCL_ERROR_NU_INT;
    }
  }

  // Check for errors in creating the spline
  if (stat || (*status)) {
    // Not setting a status_message here because we can't easily pass a
    // cosmology to this function - message printed in ccl_error.c.
    *status = CCL_ERROR_NU_INT;
    gsl_spline_free(spl);
  }

  gsl_integration_cquad_workspace_free(workspace);
  free(mnut);
  free(y);

  return spl;
}

/* ------- ROUTINE: ccl_nu_phasespace_intg ------
INPUTS: mnuOT: the dimensionless mass / temperature of a single massive neutrino
TASK: Get the value of the phase space integral at mnuOT
*/
static double nu_phasespace_intg(double mnuOT, int* status)
{
  // Check if the global variable for the phasespace spline has been defined yet:
  if (nu_spline == NULL)
    nu_spline = calculate_nu_phasespace_spline(status);

  if (*status) {
    return NAN;
  }

  double integral_value = 0.;

  // First check the cases where we are in the limits.
  if (mnuOT < CCL_NU_MNUT_MIN)
    return 7./8.;
  else if (mnuOT > CCL_NU_MNUT_MAX)
    return 0.2776566337 * mnuOT;

  int gslstatus = gsl_spline_eval_e(nu_spline, log(mnuOT), NULL, &integral_value);
  if (gslstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_neutrinos.c: nu_phasespace_intg():");
    *status |= gslstatus;
  }
  return integral_value * 7./8.;
}

/* -------- ROUTINE: Omeganuh2 ---------
INPUTS: a: scale factor, Nnumass: number of massive neutrino species,
        mnu: total mass in eV of neutrinos, T_CMB: CMB temperature,
        T_ncdm: non-CDM temperature in units of photon temperature,
        status: pointer to status integer.
TASK: Compute Omeganu * h^2 as a function of time.
!! To all practical purposes, Neff is simply N_nu_mass !!
*/
double ccl_Omeganuh2(double a, int N_nu_mass, double* mnu, double T_CMB, double T_ncdm, int* status) {
  double Tnu, a4, prefix_massless, OmNuh2;
  double Tnu_eff, mnuOT, intval, prefix_massive;

  // First check if N_nu_mass is 0
  if (N_nu_mass == 0) return 0.0;

  Tnu = T_CMB*pow(4./11.,1./3.);
  a4 = a*a*a*a;

  // Tnu_eff is used in the massive case because CLASS uses an effective
  // temperature of nonLCDM components to match to mnu / Omeganu =93.14eV. Tnu_eff = T_ncdm * T_CMB = 0.71611 * T_CMB
  Tnu_eff = Tnu * T_ncdm / (pow(4./11.,1./3.));

  // Define the prefix using the effective temperature (to get mnu / Omega = 93.14 eV) for the massive case:
  prefix_massive = NU_CONST * Tnu_eff * Tnu_eff * Tnu_eff * Tnu_eff;

  OmNuh2 = 0.; // Initialize to 0 - we add to this for each massive neutrino species.
  for(int i=0; i < N_nu_mass; i++) {

    // Check whether this species is effectively massless
    // In this case, invoke the analytic massless limit:
    if (mnu[i] < 0.00017) {  // Limit taken from Lesgourges et al. 2012
      prefix_massless = NU_CONST  * Tnu * Tnu * Tnu * Tnu;
      OmNuh2 = N_nu_mass*prefix_massless*7./8./a4 + OmNuh2;
    } else {
       // For the true massive case:
       // Get mass over T (mass (eV) / ((kb eV/s/K) Tnu_eff (K))
       // This returns the density normalized so that we get nuh2 at a=0
       mnuOT = mnu[i] / (Tnu_eff/a) * (ccl_constants.EV_IN_J / (ccl_constants.KBOLTZ));

       // Get the value of the phase-space integral
       intval = nu_phasespace_intg(mnuOT, status);
       OmNuh2 = intval*prefix_massive/a4 + OmNuh2;
    }
  }

  return OmNuh2;
}

#undef GSL_EPSABS_NU
#undef GSL_EPSREL_NU
#undef GSL_N_ITERATION_NU
