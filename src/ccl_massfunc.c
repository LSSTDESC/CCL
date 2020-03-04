#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include "ccl.h"


/*----- ROUTINE: dc_NakamuraSuto -----
INPUT: cosmology, scale factor
TASK: Computes the peak threshold: delta_c(z) assuming LCDM.
Cosmology dependence of the critical linear density according to the spherical-collapse model.
Fitting function from Nakamura & Suto (1997; arXiv:astro-ph/9710107).
*/
double dc_NakamuraSuto(ccl_cosmology *cosmo, double a, int *status){

  double Om_mz = ccl_omega_x(cosmo, a, ccl_species_m_label, status);
  double dc0 = (3./20.)*pow(12.*M_PI,2./3.);
  double dc = dc0*(1.+0.012299*log10(Om_mz));

  return dc;

}

/*----- ROUTINE: Dv_BryanNorman -----
INPUT: cosmology, scale factor
TASK: Computes the virial collapse density contrast with respect to the matter density assuming LCDM.
Cosmology dependence of the virial collapse density according to the spherical-collapse model
Fitting function from Bryan & Norman (1998; arXiv:astro-ph/9710107)
*/
double Dv_BryanNorman(ccl_cosmology *cosmo, double a, int *status){

  double Om_mz = ccl_omega_x(cosmo, a, ccl_species_m_label, status);
  double x = Om_mz-1.;
  double Dv0 = 18.*pow(M_PI,2);
  double Dv = (Dv0+82.*x-39.*pow(x,2))/Om_mz;

  return Dv;
}

static double sigmaM_m2r(ccl_cosmology *cosmo, double halomass, int *status)
{
  double rho_m, smooth_radius;

  // Comoving matter density
  //rho_m = ccl_constants.RHO_CRITICAL*cosmo->params.Omega_m*cosmo->params.h*cosmo->params.h;
  rho_m = ccl_rho_x(cosmo, 1., ccl_species_m_label, 1, status);

  smooth_radius = pow((3.0*halomass) / (4*M_PI*rho_m), (1.0/3.0));

  return smooth_radius;
}

void ccl_cosmology_compute_sigma(ccl_cosmology *cosmo, int *status)
{
  if(cosmo->computed_sigma)
    return;

  // create linearly-spaced values of the mass.
  int nm = cosmo->spline_params.LOGM_SPLINE_NM;
  double *m = NULL;
  double *y = NULL;
  double smooth_radius;
  double na, nb;

  m = ccl_linear_spacing(cosmo->spline_params.LOGM_SPLINE_MIN, cosmo->spline_params.LOGM_SPLINE_MAX, nm);
  if (m == NULL ||
      (fabs(m[0]-cosmo->spline_params.LOGM_SPLINE_MIN)>1e-5) ||
      (fabs(m[nm-1]-cosmo->spline_params.LOGM_SPLINE_MAX)>1e-5) ||
      (m[nm-1]>10E17)) {
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,"ccl_cosmology_compute_sigmas(): Error creating linear spacing in m\n");
  }

  if (*status == 0) {
    // create space for y, to be filled with sigma and dlnsigma_dlogm
    y = malloc(sizeof(double)*nm);
    if (y == NULL) {
      *status = CCL_ERROR_MEMORY;
    }
  }

  // start up of GSL pointers
  int gslstatus = 0;
  gsl_spline *logsigma = NULL;
  gsl_spline *dlnsigma_dlogm = NULL;

  // fill in sigma, if no errors have been triggered at this time.
  if (*status == 0) {
    for (int i=0; i<nm; i++) {
      smooth_radius = sigmaM_m2r(cosmo, pow(10,m[i]), status);
      y[i] = log(ccl_sigmaR(cosmo, smooth_radius, 1., status));
    }
    logsigma = gsl_spline_alloc(cosmo->spline_params.M_SPLINE_TYPE, nm);
    if (logsigma == NULL) {
      *status = CCL_ERROR_MEMORY;
    }
  }

  if (*status == 0) {
    gslstatus = gsl_spline_init(logsigma, m, y, nm);
    if (gslstatus != GSL_SUCCESS) {
      *status = CCL_ERROR_SPLINE ;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_massfunc.c: ccl_cosmology_compute_sigma(): Error creating sigma(M) spline\n");
    }
  }

  // again, making splines assuming nothing bad has happened to this point
  if (*status == 0 ) {
    for (int i=0; i<nm; i++) {
      if(i==0) {
        gslstatus |= gsl_spline_eval_e(logsigma, m[i], NULL,&na);
        gslstatus |= gsl_spline_eval_e(logsigma, m[i]+cosmo->spline_params.LOGM_SPLINE_DELTA/2., NULL,&nb);
        y[i] = 2.*(na-nb)*y[i] / cosmo->spline_params.LOGM_SPLINE_DELTA;
      }
      else if (i==nm-1) {
        gslstatus |= gsl_spline_eval_e(logsigma, m[i]-cosmo->spline_params.LOGM_SPLINE_DELTA/2., NULL,&na);
        gslstatus |= gsl_spline_eval_e(logsigma, m[i], NULL,&nb);
        y[i] = 2.*(na-nb)*y[i] / cosmo->spline_params.LOGM_SPLINE_DELTA;
      }
      else {
        gslstatus |= gsl_spline_eval_e(logsigma, m[i]-cosmo->spline_params.LOGM_SPLINE_DELTA/2., NULL,&na);
        gslstatus |= gsl_spline_eval_e(logsigma, m[i]+cosmo->spline_params.LOGM_SPLINE_DELTA/2., NULL,&nb);
        y[i] = (na-nb) / cosmo->spline_params.LOGM_SPLINE_DELTA;
      }
    }

    if(gslstatus != GSL_SUCCESS ) {
      ccl_raise_gsl_warning(
        gslstatus, "ccl_massfunc.c: ccl_cosmology_compute_sigma():");
      *status = CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_massfunc.c: ccl_cosmology_compute_sigma(): "
        "Error evaluating grid points for dlnsigma/dlogM spline\n");
    }
  }

  if (*status == 0) {
    dlnsigma_dlogm = gsl_spline_alloc(cosmo->spline_params.M_SPLINE_TYPE, nm);
    if (dlnsigma_dlogm == NULL) {
      *status = CCL_ERROR_MEMORY;
    }
  }

  if (*status == 0) {
    gslstatus = gsl_spline_init(dlnsigma_dlogm, m, y, nm);
    if (gslstatus != GSL_SUCCESS) {
      *status = CCL_ERROR_SPLINE ;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_massfunc.c: ccl_cosmology_compute_sigma(): Error creating dlnsigma/dlogM spline\n");
    }
  }

  if (*status == 0) {
    cosmo->data.logsigma = logsigma;
    cosmo->data.dlnsigma_dlogm = dlnsigma_dlogm;
    cosmo->computed_sigma = true;
  } else {
    gsl_spline_free(logsigma);
    gsl_spline_free(dlnsigma_dlogm);
  }

  free(m);
  free(y);
}

/*----- ROUTINE: ccl_sigma_M -----
INPUT: ccl_cosmology * cosmo, double halo mass in units of Msun, double scale factor
TASK: returns sigma from the sigmaM interpolation. Also computes the sigma interpolation if
necessary.
*/
double ccl_sigmaM(ccl_cosmology *cosmo, double log_halomass, double a, int *status)
{
  double sigmaM;
  // Check if sigma has already been calculated
  if (!cosmo->computed_sigma) {
    *status = CCL_ERROR_SIGMA_INIT;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_massfunc.c: ccl_sigmaM(): linear power spctrum has not been computed!");
    return NAN;
  }

  double lgsigmaM;

  int gslstatus = gsl_spline_eval_e(cosmo->data.logsigma, log_halomass, NULL, &lgsigmaM);

  if(gslstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_massfunc.c: ccl_sigmaM():");
    *status |= gslstatus;
  }

  // Interpolate to get sigma
  sigmaM = exp(lgsigmaM)*ccl_growth_factor(cosmo, a, status);
  return sigmaM;
}

/*----- ROUTINE: ccl_dlnsigM_dlogM -----
INPUT: ccl_cosmology *cosmo, double halo mass in units of Msun
TASK: returns the value of the derivative of ln(sigma^-1) with respect to log10 in halo mass.
*/
double ccl_dlnsigM_dlogM(ccl_cosmology *cosmo, double log_halomass, int *status)
{
  // Check if sigma has already been calculated
  if (!cosmo->computed_sigma) {
    *status = CCL_ERROR_SIGMA_INIT;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_massfunc.c: ccl_sigmaM(): linear power spctrum has not been computed!");
    return NAN;
  }
  
  double dlsdlgm;
  int gslstatus = gsl_spline_eval_e(cosmo->data.dlnsigma_dlogm,
				    log_halomass, NULL, &dlsdlgm);
  if(gslstatus) { 
    ccl_raise_gsl_warning(gslstatus, "ccl_massfunc.c: ccl_dlnsigM_dlogM():");
    *status |= gslstatus;
  }
  return dlsdlgm;
}
