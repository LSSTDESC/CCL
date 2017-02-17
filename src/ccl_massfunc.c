#include "ccl.h"
#include "ccl_core.h"
#include "ccl_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gsl/gsl_integration.h"
#include "gsl/gsl_interp.h"
#include "gsl/gsl_spline.h"
#include "ccl_power.h"
#include "ccl_massfunc.h"
#include "ccl_error.h"

/*----- ROUTINE: ccl_massfunc_f -----
INPUT: cosmology+parameters, a smoothing mass, and a redshift
TASK: Outputs fitting function for use in halo mass function calculation;
  currently only supports:
    ccl_tinker (arxiv 0803.2706 )
    ccl_angulo (arxiv 1203.3216 ) 
    ccl_watson (arxiv 1212.0095 )
*/

static double massfunc_f(ccl_cosmology *cosmo, double smooth_mass,double redshift, int * status)
{
  double fit_A, fit_a, fit_b, fit_c, fit_d, overdensity_delta;
  double scale, Omega_m_a;
  double delta_c_Tinker, nu;

  double sigma=ccl_sigmaM(cosmo,smooth_mass,redshift, status);

  switch(cosmo->config.mass_function_method){
  case ccl_tinker:
    
    //TODO: maybe use macros for numbers
    overdensity_delta = 200.0;
    fit_A = 0.186*pow(1+redshift, -0.14);
    fit_a = 1.47*pow(1+redshift, -0.06);
    fit_d = pow(10, -1.0*pow(0.75 / log10(overdensity_delta / 75.0), 1.2 ));
    fit_b = 2.57*pow(1+redshift, -1.0*fit_d);
    fit_c = 1.19;

    return fit_A*(pow(sigma/fit_b,-fit_a)+1.0)*exp(-fit_c/sigma/sigma);
    break;
    //this version uses f(nu) parameterization from Eq. 8 in Tinker et al. 2010
    // use this for consistency with Tinker et al. 2010 fitting function for halo bias
  case ccl_tinker10:
    
    overdensity_delta = 200.0;
    //critical collapse overdensity assumed in this model
    delta_c_Tinker = 1.686;
    nu = delta_c_Tinker/(sigma);

    fit_A = 0.368; //alpha in Eq. 8
    fit_a = -0.243*pow(1+redshift, 0.27); //eta in Eq. 8
    fit_b = 0.589*pow(1+redshift, 0.20); //beta in Eq. 8
    fit_c = 0.864*pow(1+redshift, -0.01); //gamma in Eq. 8
    fit_d = -0.729*pow(1+redshift, -0.08); //phi in Eq. 8;

    return nu*fit_A*(1.+pow(fit_b*nu,-2.*fit_d))*pow(nu, 2.*fit_a)*exp(-0.5*fit_c*nu*nu);
    break;

  case ccl_watson:
    scale = 1.0/(1.0+redshift);
    Omega_m_a = ccl_omega_m_a(cosmo, scale);
    
    fit_A = Omega_m_a*(0.990*pow(1+redshift,-3.216)+0.074);
    fit_a = Omega_m_a*(5.907*pow(1+redshift,-3.599)+2.344);
    fit_b = Omega_m_a*(3.136*pow(1+redshift,-3.058)+2.349);
    fit_c = 1.318;

    return fit_A*(pow(sigma/fit_b,-fit_a)+1.0)*exp(-fit_c/sigma/sigma);

  case ccl_angulo:
    fit_A = 0.201;
    fit_a = 2.08;
    fit_b = 1.7;
    fit_c = 1.172;

    return fit_A*pow( (fit_a/sigma)+1.0, fit_b)*exp(-fit_c/sigma/sigma);

  default:
    *status = CCL_ERROR_MF;
    sprintf(cosmo->status_message ,
	    "ccl_massfunc.c: ccl_massfunc(): Unknown or non-implemented mass function method: %d \n",
	    cosmo->config.mass_function_method);
    return 0;
  }
}
static double ccl_halo_b1(ccl_cosmology *cosmo, double smooth_mass,double redshift, int * status)
{
  double fit_A, fit_B, fit_C, fit_a, fit_b, fit_c, overdensity_delta, y;
  double scale, Omega_m_a;
  double delta_c_Tinker, nu;
  double sigma=ccl_sigmaM(cosmo,smooth_mass,redshift, status);
  switch(cosmo->config.mass_function_method){

    //this version uses b(nu) parameterization, Eq. 6 in Tinker et al. 2010
    // use this for consistency with Tinker et al. 2010 fitting function for halo bias
  case ccl_tinker10:
    
    overdensity_delta = 200.0;
    y = log10(overdensity_delta);
    //critical collapse overdensity assumed in this model
    delta_c_Tinker = 1.686;
    //peak height - note that this factorization is incorrect for e.g. massive neutrino cosmologies
    nu = delta_c_Tinker/(sigma);
    // Table 2 in https://arxiv.org/pdf/1001.3162.pdf
    fit_A = 1.0 + 0.24*y*exp(-pow(4./y,4.)); 
    fit_a = 0.44*y-0.88; 
    fit_B = 0.183; 
    fit_b = 1.5; 
    fit_C = 0.019+0.107*y+0.19*exp(-pow(4./y,4.)); 
    fit_c = 2.4; 

    return 1.-fit_A*pow(nu,fit_a)/(pow(nu,fit_a)+pow(delta_c_Tinker,fit_a))+fit_B*pow(nu,fit_b)+fit_C*pow(nu,fit_c);
    break;

  default:
    *status = CCL_ERROR_MF;
    cosmo->status = 11;
    sprintf(cosmo->status_message ,
      "ccl_massfunc.c: ccl_halo_b1(): No b(M) fitting function implemented for mass_function_method: %d \n",
      cosmo->config.mass_function_method);
    return 0;
  }
}

void ccl_cosmology_compute_sigma(ccl_cosmology * cosmo, int *status)
{
    if(cosmo->computed_sigma)
        return;

    // create linearly-spaced values of the mass.
    int nm=LOGM_SPLINE_NM;
    double * m = ccl_linear_spacing(LOGM_SPLINE_MIN, LOGM_SPLINE_MAX, nm);
    if (m==NULL ||
        (fabs(m[0]-LOGM_SPLINE_MIN)>1e-5) ||
        (fabs(m[nm-1]-LOGM_SPLINE_MAX)>1e-5) ||
        (m[nm-1]>10E17)
        ) {
       *status =CCL_ERROR_LINSPACE;
       strcpy(cosmo->status_message,"ccl_cosmology_compute_sigmas(): Error creating linear spacing in m\n");
       return;
    }
    
    // allocate space for y, to be filled with sigma and dlnsigma_dlogm
    double *y = malloc(sizeof(double)*nm);
    double smooth_radius; 
   
   // fill in sigma
   for (int i=0; i<nm; i++){
     smooth_radius = ccl_massfunc_m2r(cosmo, pow(10,m[i]), status);
     y[i] = log10(ccl_sigmaR(cosmo, smooth_radius));
   }
   gsl_spline * logsigma = gsl_spline_alloc(M_SPLINE_TYPE, nm);
   *status = gsl_spline_init(logsigma, m, y, nm);
   if (*status){
     free(m);
     free(y);
     gsl_spline_free(logsigma);
     *status = CCL_ERROR_SPLINE ;
     strcpy(cosmo->status_message, "ccl_massfunc.c: ccl_cosmology_compute_sigma(): Error creating sigma(M) spline\n");
     return;
   }
   for (int i=0; i<nm; i++){
     if(i==0){
       y[i] = log(pow(10, gsl_spline_eval(logsigma, m[i], NULL)))-log(pow(10,gsl_spline_eval(logsigma, m[i]+LOGM_SPLINE_DELTA/2., NULL)));
       y[i] = 2.*y[i] / LOGM_SPLINE_DELTA;
     }
     else if (i==nm-1){
       y[i] = log(pow(10, gsl_spline_eval(logsigma, m[i]-LOGM_SPLINE_DELTA/2., NULL)))-log(pow(10,gsl_spline_eval(logsigma, m[i], NULL)));
       y[i] = 2.*y[i] / LOGM_SPLINE_DELTA;
     }
     else{
       y[i] = (log(pow(10,gsl_spline_eval(logsigma, m[i]-LOGM_SPLINE_DELTA/2., NULL)))-log(pow(10,gsl_spline_eval(logsigma, m[i]+LOGM_SPLINE_DELTA/2., NULL))));
       y[i] = y[i] / LOGM_SPLINE_DELTA;
     }
   }

   gsl_spline * dlnsigma_dlogm = gsl_spline_alloc(M_SPLINE_TYPE, nm);
   *status = gsl_spline_init(dlnsigma_dlogm, m, y, nm);
   if (*status){
     free(m);
     free(y);
     gsl_spline_free(logsigma);
     *status = CCL_ERROR_SPLINE ;
     strcpy(cosmo->status_message, "ccl_massfunc.c: ccl_cosmology_compute_sigma(): Error creating dlnsigma/dlogM spline\n");
     return;
   }

   if(cosmo->data.accelerator_m==NULL)
     cosmo->data.accelerator_m=gsl_interp_accel_alloc();
   cosmo->data.logsigma = logsigma;
   cosmo->data.dlnsigma_dlogm = dlnsigma_dlogm;
   cosmo->computed_sigma = true;

   free(m);
   free(y);
}

/*----- ROUTINE: ccl_massfunc -----
INPUT: ccl_cosmology * cosmo, double smoothing mass in units of Msun, double redshift
TASK: returns halo mass function as dn / dlog10 m
*/

double ccl_massfunc(ccl_cosmology *cosmo, double smooth_mass, double redshift, int * status)
{
  if (!cosmo->computed_sigma){
    ccl_cosmology_compute_sigma(cosmo, status);
    ccl_check_status(cosmo, status);
  }

  double f,deriv,rho_m,logmass;
  
  logmass = log10(smooth_mass);
  rho_m = RHO_CRITICAL*cosmo->params.Omega_m*cosmo->params.h*cosmo->params.h;
  f=massfunc_f(cosmo,smooth_mass,redshift, status);
  deriv = gsl_spline_eval(cosmo->data.dlnsigma_dlogm, logmass, cosmo->data.accelerator_m);
  return f*rho_m*deriv/smooth_mass;
}

/*----- ROUTINE: ccl_halob1 -----
INPUT: ccl_cosmology * cosmo, double smoothing mass in units of Msun, double redshift
TASK: returns linear halo bias
*/

double ccl_halo_bias(ccl_cosmology *cosmo, double smooth_mass, double redshift, int * status)
{
  if (!cosmo->computed_sigma){
    ccl_cosmology_compute_sigma(cosmo, status);
    ccl_check_status(cosmo, status);
  }

  double f;
  f = ccl_halo_b1(cosmo,smooth_mass,redshift, status);  
  ccl_check_status(cosmo, status);  
  return f;
}
/*---- ROUTINE: ccl_massfunc_m2r -----
INPUT: ccl_cosmology * cosmo, smooth_mass in units of Msun
TASK: takes smoothing halo mass and converts to smoothing halo radius
  in units of Mpc.
*/
double ccl_massfunc_m2r(ccl_cosmology * cosmo, double smooth_mass, int * status)
{
    double rho_m, smooth_radius;

    //TODO: make this neater
    rho_m = RHO_CRITICAL*cosmo->params.Omega_m*cosmo->params.h*cosmo->params.h;

    smooth_radius = pow((3.0*smooth_mass) / (4*M_PI*rho_m), (1.0/3.0));

    return smooth_radius;
}

/*----- ROUTINE: ccl_sigma_M -----
INPUT: ccl_cosmology * cosmo, double smoothing mass in units of Msun, double redshift
TASK: returns sigma from the sigmaM interpolation. Also computes the sigma interpolation if
necessary.
*/

double ccl_sigmaM(ccl_cosmology * cosmo, double smooth_mass, double redshift, int * status)
{
    double sigmaM;

    if (!cosmo->computed_sigma){
        ccl_cosmology_compute_sigma(cosmo, status);
        ccl_check_status(cosmo, status);
    }

    sigmaM = pow(10,gsl_spline_eval(cosmo->data.logsigma, log10(smooth_mass), cosmo->data.accelerator_m));
    sigmaM = sigmaM*ccl_growth_factor(cosmo, 1.0/(1.0+redshift), status);

    return sigmaM;
}
