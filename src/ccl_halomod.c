#include "ccl_utils.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_sf_expint.h"
#include "gsl/gsl_roots.h"
#include "ccl_background.h"
#include "ccl_power.h"
#include "ccl_massfunc.h"
#include "ccl_error.h"
#include "ccl_halomod.h"

// Analytic FT of NFW profile, from Cooray & Sheth (2002; Section 3 of https://arxiv.org/abs/astro-ph/0206508)
// Normalised such that U(k=0)=1
static double u_nfw_c(ccl_cosmology *cosmo, double rv, double c, double k, int *status){
   
  double rs, ks;
  double f1, f2, f3, fc;

  // Special case to prevent numerical problems if k=0,
  // the result should be unity here because of the normalisation
  if (k==0.) {    
    return 1.;    
  }

  // The general k case
  else{
    
    // Scale radius for NFW (rs=rv/c)
    rs = rv/c;

    // Dimensionless wave-number variable
    ks = k*rs;

    // Various bits for summing together to get final result
    f1 = sin(ks)*(gsl_sf_Si(ks*(1.+c))-gsl_sf_Si(ks));
    f2 = cos(ks)*(gsl_sf_Ci(ks*(1.+c))-gsl_sf_Ci(ks));
    f3 = sin(c*ks)/(ks*(1.+c));
    fc = log(1.+c)-c/(1.+c);
  
    return (f1+f2-f3)/fc;
    
  }
}

/*----- ROUTINE: ccl_halo_concentration -----
INPUT: cosmology, a halo mass [Msun], scale factor, halo definition, concentration model label
TASK: Computes halo concentration; the ratio of virial raidus to scale radius for an NFW halo.
*/
double ccl_halo_concentration(ccl_cosmology *cosmo, double halomass, double a, double odelta, int *status){

  double gz, g0, nu, delta_c, a_form;
  double Mpiv, A, B, C;

  switch(cosmo->config.halo_concentration_method){

    // Bhattacharya et al. (2011; 1005.2239; Delta = 200rho_m; Table 2)
  case ccl_bhattacharya2011:

    if (odelta != 200.) {
      *status = CCL_ERROR_CONC_DV;
      ccl_cosmology_set_status_message(cosmo, "ccl_halomod.c: halo_concentration(): Bhattacharya (2011) concentration relation only valid for Delta_v = 200 \n");
      return NAN;
    }
    
    gz = ccl_growth_factor(cosmo,a,status);
    g0 = ccl_growth_factor(cosmo,1.0,status);
    delta_c = 1.686;
    nu = delta_c/ccl_sigmaM(cosmo, halomass, a, status);
    return 9.*pow(nu,-0.29)*pow(gz/g0,1.15);

    // Duffy et al. (2008; 0804.2486; Table 1)
  case ccl_duffy2008:

    Mpiv = 2e12/cosmo->params.h; // Pivot mass in Msun (note in the paper units are Msun/h)

    if (odelta == Dv_BryanNorman(cosmo, a, status)) {

      // Duffy et al. (2008) for virial density haloes (second section in Table 1)
      A = 7.85;
      B = -0.081;
      C = -0.71;
      return A*pow(halomass/Mpiv,B)*pow(a,-C);

    } else if (odelta == 200.) {

      // Duffy et al. (2008) for x200 mean-matter-density haloes (third section in Table 1)
      A = 10.14;
      B = -0.081;
      C = -1.01;
      return A*pow(halomass/Mpiv,B)*pow(a,-C);

    } else {

      *status = CCL_ERROR_CONC_DV;
      ccl_cosmology_set_status_message(cosmo, "ccl_halomod.c: halo_concentration(): Duffy (2008) virial concentration only valid for virial Delta_v or 200\n");
      return NAN;
      
    }

    // Constant concentration (good for tests)
  case ccl_constant_concentration:
    
    return 4.;

    // Something went wrong
  default:
    
    *status = CCL_ERROR_HALOCONC;
    ccl_raise_exception(*status, "ccl_halomod.c: concentration-mass relation specified incorrectly");
    return NAN;
    	  
  }
}

// Fourier Transforms of halo profiles
static double window_function(ccl_cosmology *cosmo, double m, double k, double a, double odelta, ccl_win_label label, int *status){

  double rho_matter, c, rv;
  
  switch(label){

  case ccl_nfw:

    // The mean background matter density in Msun/Mpc^3
    rho_matter = ccl_rho_x(cosmo, 1., ccl_species_m_label, 1, status);

    // The halo virial radius
    rv = r_delta(cosmo, m, a, odelta, status);

    // The halo concentration for this halo mass and at this scale factor  
    c = ccl_halo_concentration(cosmo, m, a, odelta, status);    

    // The function U is normalised to 1 for k<<1 so multiplying by M/rho turns units to overdensity
    return m*u_nfw_c(cosmo, rv, c, k, status)/rho_matter;

    // Something went wrong
  default:
    
    *status = CCL_ERROR_HALOWIN;
    ccl_raise_exception(*status, "ccl_halomod.c: Window function specified incorrectly");
    return NAN;
    
  }
  
}

// Parameters structure for the one-halo integrand
typedef struct{  
  ccl_cosmology *cosmo;
  double k, a;
  int *status;
} Int_one_halo_Par;

// Integrand for the one-halo integral
static double one_halo_integrand(double log10mass, void *params){  
  
  Int_one_halo_Par *p = (Int_one_halo_Par *)params;;
  double halomass = pow(10,log10mass);
  double odelta = Dv_BryanNorman(p->cosmo, p->a, p->status); // Virial density for haloes

  // The normalised Fourier Transform of a halo density profile
  double wk = window_function(p->cosmo,halomass, p->k, p->a, odelta, ccl_nfw, p->status);

  // Fairly sure that there should be no ln(10) factor should be here since the integration is being specified in log10 range
  double dn_dlogM = ccl_massfunc(p->cosmo, halomass, p->a, odelta, p->status);
    
  return dn_dlogM*pow(wk,2);
}

// The one-halo term integral using gsl
static double one_halo_integral(ccl_cosmology *cosmo, double k, double a, int *status){

  int one_halo_integral_status = 0, qagstatus;
  double result = 0, eresult;
  double log10mmin = log10(HM_MMIN);
  double log10mmax = log10(HM_MMAX);
  Int_one_halo_Par ipar;
  gsl_function F;
  gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);

  // Structure required for the gsl integration
  ipar.cosmo = cosmo;
  ipar.k = k;
  ipar.a = a;
  ipar.status = &one_halo_integral_status;
  F.function = &one_halo_integrand;
  F.params = &ipar;

  // Actually does the integration
  qagstatus = gsl_integration_qag(&F, log10mmin, log10mmax, HM_EPSABS, HM_EPSREL, HM_LIMIT, HM_INT_METHOD, w, &result, &eresult);

  // Clean up
  gsl_integration_workspace_free(w);

  // Check for errors
  if (qagstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(qagstatus, "ccl_halomod.c: one_halo_integral():");
    *status = CCL_ERROR_ONE_HALO_INT;
    ccl_cosmology_set_status_message(cosmo, "ccl_halomod.c: one_halo_integral(): Integration failure\n");
    return NAN;      
  } else {
    return result;
  }
  
}

// Parameters structure for the two-halo integrand
typedef struct{  
  ccl_cosmology *cosmo;
  double k, a;
  int *status;
} Int_two_halo_Par;

// Integrand for the two-halo integral
static double two_halo_integrand(double log10mass, void *params){  
  
  Int_two_halo_Par *p = (Int_two_halo_Par *)params;
  double halomass = pow(10,log10mass);
  double odelta = Dv_BryanNorman(p->cosmo, p->a, p->status); // Virial density for haloes

  // The normalised Fourier Transform of a halo density profile
  double wk = window_function(p->cosmo,halomass, p->k, p->a, odelta, ccl_nfw, p->status);

  // Fairly sure that there should be no ln(10) factor should be here since the integration is being specified in log10 range
  double dn_dlogM = ccl_massfunc(p->cosmo, halomass, p->a, odelta, p->status);

  // Halo bias
  double b = ccl_halo_bias(p->cosmo, halomass, p->a, odelta, p->status);
    
  return b*dn_dlogM*wk;
}

// The two-halo term integral using gsl
static double two_halo_integral(ccl_cosmology *cosmo, double k, double a, int *status){

  int two_halo_integral_status = 0, qagstatus;
  double result = 0, eresult;
  double log10mmin = log10(HM_MMIN);
  double log10mmax = log10(HM_MMAX);
  Int_two_halo_Par ipar;
  gsl_function F;
  gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);

  // Structure required for the gsl integration
  ipar.cosmo = cosmo;
  ipar.k = k;
  ipar.a = a;
  ipar.status = &two_halo_integral_status;
  F.function = &two_halo_integrand;
  F.params = &ipar;

  // Actually does the integration
  qagstatus = gsl_integration_qag(&F, log10mmin, log10mmax, HM_EPSABS, HM_EPSREL, HM_LIMIT, HM_INT_METHOD, w, &result, &eresult);

  // Clean up
  gsl_integration_workspace_free(w);

  // Check for errors
  if (qagstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(qagstatus, "ccl_halomod.c: two_halo_integral():");
    *status = CCL_ERROR_TWO_HALO_INT;
    ccl_cosmology_set_status_message(cosmo, "ccl_halomod.c: two_halo_integral(): Integration failure\n");
    return NAN;      
  } else {
    return result;
  }
  
}

/*----- ROUTINE: ccl_twohalo_matter_power -----
INPUT: cosmology, wavenumber [Mpc^-1], scale factor
TASK: Computes the two-halo power spectrum term in the halo model assuming NFW haloes
*/
double ccl_twohalo_matter_power(ccl_cosmology *cosmo, double k, double a, int *status){
    
  // Get the integral
  double I2h = two_halo_integral(cosmo, k, a, status);
    
  // The addative correction is the missing part of the integral below the lower-mass limit
  double A = 1.-two_halo_integral(cosmo, 0., a, status);

  // Virial overdensity for haloes
  double odelta = Dv_BryanNorman(cosmo, a, status);

  // ...multiplied by the ratio of window functions
  double W1 = window_function(cosmo, HM_MMIN, k,  a, odelta, ccl_nfw, status);
  double W2 = window_function(cosmo, HM_MMIN, 0., a, odelta, ccl_nfw, status);
  A = A*W1/W2;    

  // Add the additive correction to the calculated integral
  I2h = I2h+A;
      
  return ccl_linear_matter_power(cosmo, k, a, status)*I2h*I2h;
  
}

/*----- ROUTINE: ccl_onehalo_matter_power -----
INPUT: cosmology, wavenumber [Mpc^-1], scale factor
TASK: Computes the one-halo power spectrum term in the halo model assuming NFW haloes
*/
double ccl_onehalo_matter_power(ccl_cosmology *cosmo, double k, double a, int *status){
  
  return one_halo_integral(cosmo, k, a, status);
  
}

/*----- ROUTINE: ccl_onehalo_matter_power -----
INPUT: cosmology, wavenumber [Mpc^-1], scale factor
TASK: Computes the halo model power spectrum by summing the two- and one-halo terms
*/
double ccl_halomodel_matter_power(ccl_cosmology *cosmo, double k, double a, int *status){

  // Standard sum of two- and one-halo terms
  return ccl_twohalo_matter_power(cosmo, k, a, status)+ccl_onehalo_matter_power(cosmo, k, a, status);
  
}
