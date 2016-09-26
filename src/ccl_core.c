#include "ccl_core.h"
#include "ccl_utils.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "gsl/gsl_errno.h"
#include "gsl/gsl_odeiv.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_integration.h"

const ccl_configuration default_config = {ccl_fitting_function, ccl_halofit, ccl_tinker};

ccl_cosmology * ccl_cosmology_create(ccl_parameters params, ccl_configuration config)
{
  ccl_cosmology * cosmo = malloc(sizeof(ccl_cosmology));
  cosmo->params = params;
  cosmo->config = config;

  cosmo->data.chi = NULL;
  cosmo->data.growth = NULL;
  cosmo->data.fgrowth = NULL;
  cosmo->data.E = NULL;
  cosmo->data.accelerator=NULL;
  cosmo->data.growth0 = 1.;

  cosmo->data.sigma = NULL;
  
  cosmo->data.p_lin = NULL;
  cosmo->data.p_nl = NULL;
  
  cosmo->computed_distances = false;
  cosmo->computed_growth = false;
  cosmo->computed_power = false;
  cosmo->computed_sigma = false;
  cosmo->status = 0;
  
  return cosmo;
}

void ccl_parameters_fill_initial(ccl_parameters *params)
{
  // Fixed radiation parameters
  // Omega_g * h**2 is known from T_CMB
  double omega_g = 1.71e-5;
  params->Omega_g = omega_g/params->h/params->h;
  params->T_CMB =  2.726;

  // Derived parameters
  params->Omega_l = 1.0 - params->Omega_m - params->Omega_g - params->Omega_n - params->Omega_k;
    // Initially undetermined parameters - set to nan to trigger
  // problems if they are mistakenly used.
  if (isfinite(params->A_s)){params->sigma_8 = NAN;}
  if (isfinite(params->sigma_8)){params->A_s = NAN;}
  params->z_star = NAN;

}


ccl_parameters ccl_parameters_create(double Omega_c, double Omega_b, double Omega_k, double Omega_n, double w0, double wa, double h, double A_s, double n_s){
  ccl_parameters params;
  params.sigma_8 = NAN;
  params.A_s = NAN;
  params.Omega_c = Omega_c;
  params.Omega_b = Omega_b;
  params.Omega_m = Omega_b + Omega_c;
  params.Omega_n = Omega_n;
  params.Omega_k = Omega_k;

  // Dark Energy
  params.w0 = w0;
  params.wa = wa;

  // Hubble parameters
  params.h = h;
  params.H0 = h*100;

  // Primordial power spectra
  params.A_s = A_s;
  params.n_s = n_s;

  // Set remaining standard and easily derived parameters
  ccl_parameters_fill_initial(&params);
  
  return params;  
}

ccl_parameters ccl_parameters_create_flat_lcdm(double Omega_c, double Omega_b, double h, double A_s, double n_s)
{
  double Omega_k = 0.0;
  double Omega_n = 0.0;
  double w0 = -1.0;
  double wa = 0.0;
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Omega_n, w0, wa, h, A_s, n_s);
  return params;


}

ccl_parameters ccl_parameters_create_lcdm(double Omega_c, double Omega_b, double Omega_k, double h, double A_s, double n_s)
{
  double Omega_n = 0.0;
  double w0 = -1.0;
  double wa = 0.0;
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Omega_n, w0, wa, h, A_s, n_s);
  return params;


}

ccl_parameters ccl_parameters_create_flat_wcdm(double Omega_c, double Omega_b, double w0, double h, double A_s, double n_s)
{

  double Omega_k = 0.0;
  double Omega_n = 0.0;
  double wa = 0.0;
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Omega_n, w0, wa, h, A_s, n_s);
  return params;
}

ccl_parameters ccl_parameters_create_flat_wacdm(double Omega_c, double Omega_b, double w0, double wa, double h, double A_s, double n_s)
{

  double Omega_k = 0.0;
  double Omega_n = 0.0;
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Omega_n, w0, wa, h, A_s, n_s);
  return params;
}

void ccl_data_free(ccl_data * data)
{
  gsl_spline_free(data->chi);
  gsl_spline_free(data->growth);
  gsl_spline_free(data->fgrowth);
  gsl_interp_accel_free(data->accelerator);
  gsl_spline_free(data->E);
  gsl_spline_free(data->sigma);
  gsl_spline_free(data->p_lin);
  gsl_spline_free(data->p_nl);
}

void ccl_cosmology_free(ccl_cosmology * cosmo)
{
  ccl_data_free(&cosmo->data);
  free(cosmo);
}
