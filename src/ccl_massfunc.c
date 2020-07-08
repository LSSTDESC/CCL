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

  int na = cosmo->spline_params.A_SPLINE_NA_PK+cosmo->spline_params.A_SPLINE_NLOG_PK-1;
  double amin=cosmo->spline_params.A_SPLINE_MINLOG_PK;
  double amax=cosmo->spline_params.A_SPLINE_MAX;
  int nm = cosmo->spline_params.LOGM_SPLINE_NM;
  double *m = NULL;
  double *y = NULL;
  double *aa = NULL;
  double smooth_radius;

  // create linearly-spaced values of the mass.
  m = ccl_linear_spacing(cosmo->spline_params.LOGM_SPLINE_MIN, cosmo->spline_params.LOGM_SPLINE_MAX, nm);
  if (m == NULL ||
      (fabs(m[0]-cosmo->spline_params.LOGM_SPLINE_MIN)>1e-5) ||
      (fabs(m[nm-1]-cosmo->spline_params.LOGM_SPLINE_MAX)>1e-5) ||
      (m[nm-1]>10E17)) {
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,"ccl_cosmology_compute_sigmas(): "
                                     "Error creating linear spacing in m\n");
  }

  // create scale factor array
  if (*status == 0) {
    aa = ccl_linlog_spacing(amin, cosmo->spline_params.A_SPLINE_MIN_PK,
                            amax, cosmo->spline_params.A_SPLINE_NLOG_PK,
                            cosmo->spline_params.A_SPLINE_NA_PK);
    if (aa == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
       "ccl_cosmology_compute_sigmas(): Error creating scale factor array\n");
    }
  }

  // create space for y, to be filled with sigma and dlnsigma_dlogm
  if (*status == 0) {
    y = malloc(sizeof(double)*nm*na);
    if (y == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_cosmology_compute_sigmas(): memory allocation\n");
    }
  }

  // fill in sigma, if no errors have been triggered at this time.
  if (*status == 0) {
    for (int j=0; j<na; j++) {
      double a_sf = aa[j];
      for (int i=0; i<nm; i++) {
        smooth_radius = sigmaM_m2r(cosmo, pow(10,m[i]), status);
        y[j*nm + i] = log(ccl_sigmaR(cosmo, smooth_radius, a_sf, status));
      }
    }
  }

  // If all went well, create sigma_M spline
  if (*status == 0) {
    cosmo->data.sigma_M = ccl_f2d_t_new(
      na, aa, nm, m, y, NULL, NULL,
      0, 1, 2, ccl_f2d_cclgrowth,
      1, NULL, 0, 1,
      ccl_f2d_3, status);
  }

  // Now we compute dlogsigma/dlogM
  int gslstatus = 0;
  if (*status == 0) {
    for (int j=0; j<na; j++) {
      double *y_row = &(y[j*nm]);
      for (int i=0; i<nm; i++) {
        gslstatus |=  gsl_spline2d_eval_deriv_x_e(cosmo->data.sigma_M->fka,
                                                  m[i], aa[j], NULL, NULL, y_row + i);
        y_row[i] = -y_row[i];
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

  // If all went well, create spline
  if (*status == 0) {
    cosmo->data.dlnsigma_dlogm = ccl_f2d_t_new(
      na, aa, nm, m, y, NULL, NULL,
      0, 1, 2, ccl_f2d_constantgrowth,
      0, NULL, 1., 1,
      ccl_f2d_3, status);
  }

  if (*status == 0)
    cosmo->computed_sigma = true;

  free(aa);
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
  // Check if sigma has already been calculated
  if (!cosmo->computed_sigma) {
    *status = CCL_ERROR_SIGMA_INIT;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_massfunc.c: ccl_sigmaM(): sigma(M) has not been computed!");
    return NAN;
  }

  return ccl_f2d_t_eval(cosmo->data.sigma_M, log_halomass, a, cosmo, status);
}

/*----- ROUTINE: ccl_dlnsigM_dlogM -----
INPUT: ccl_cosmology *cosmo, double halo mass in units of Msun
TASK: returns the value of the derivative of ln(sigma^-1) with respect to log10 in halo mass.
*/
double ccl_dlnsigM_dlogM(ccl_cosmology *cosmo, double log_halomass, double a, int *status)
{
  // Check if sigma has already been calculated
  if (!cosmo->computed_sigma) {
    *status = CCL_ERROR_SIGMA_INIT;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_massfunc.c: ccl_sigmaM(): linear power spctrum has not been computed!");
    return NAN;
  }

  return ccl_f2d_t_eval(cosmo->data.dlnsigma_dlogm, log_halomass, a, cosmo, status);
}
