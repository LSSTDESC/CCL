#include "ccl_utils.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_sf_expint.h"
#include "ccl_background.h"
#include "ccl_power.h"
#include "ccl_massfunc.h"

// Set two-halo term
// 1 - Standard two-halo term
// 2 - Linear theory
int i2h=1;

// Set full halo model pwoer spectrum method
// 1 - Standard sum of two- and one-halo terms
// 2 - Smooth transition using alpha parameter from Mead et al. (2015)
int ipow=1;

// Set concentration-mass relation
// 1 - Bhattaharya et al. (2011)
// 2 - Full Bullock et al. (2001)
// 3 - Duffy et al. (2008)
// 4 - Constant concentration (for testing)
// 5 - Simple Bullock et al. (2001)
int iconc=3;

// Select window function
// 1 - NFW profile, appropriate for matter power spectrum
int iwin=1;

// Set the mass range for the halo-model integration
double mmin=1e7;
double mmax=1e17;

// Virial density of haloes
double Delta_v=200.;

// analytic FT of NFW profile, from Cooray & Sheth (2002; Section 3 of https://arxiv.org/abs/astro-ph/0206508)
// Normalised such that U(k=0)=1
double u_nfw_c(ccl_cosmology *cosmo, double c, double halomass, double k, double a, int * status){
   
  double rv, rs, ks, Dv;
  double f1, f2, f3, fc;

  // Special case to prevent numerical problems if k=0,
  // the result should be unity here because of the normalisation
  if(k==0.){    
    return 1.;    
  }

  // The general k case
  else{
    //Scale radius for NFW (rs=rv/c)
    rv = ccl_r_delta(cosmo, halomass, a, Delta_v, status);
    rs = rv/c;

    //Dimensionless wave-number variable
    ks = k*rs;

    // Various bits for summing together to get final result
    f1 = sin(ks)*(gsl_sf_Si(ks*(1.+c))-gsl_sf_Si(ks));
    f2 = cos(ks)*(gsl_sf_Ci(ks*(1.+c))-gsl_sf_Ci(ks));
    f3 = sin(c*ks)/(ks*(1.+c));
    fc = log(1.+c)-c/(1.+c);
  
    return (f1+f2-f3)/fc;
  }
}

// Halo "formation redshift" calculated according to the Bullock et al. (2001) prescription
// TODO: Actually code this up
double z_formation_Bullock(ccl_cosmology *cosmo, double halomass, double a, int * status){ 
  return 0.;
}

// The non-linear mass (sigma(M*)=delta_c))
// TODO: Actually code this up
double Mstar(){
  return 1e13;
}

// The concentration-mass relation for haloes
// TODO: make consistency check so that ccl_halo_concentration only runs if called with appropriate definition
// TODO: e.g. if Delta != 200 rho_{mean}, should not function (or should it?)
// TODO: should this be moved to the ccl_massfunc.c ?
double ccl_halo_concentration(ccl_cosmology *cosmo, double halomass, double a, int * status)
{

  // Bhattacharya et al. 2011, Delta = 200 rho_{mean} (Table 2)
  if(iconc==1){    
    double gz = ccl_growth_factor(cosmo,a,status);
    double g0 = ccl_growth_factor(cosmo,1.0,status);
    return 9.*pow(ccl_nu(cosmo,halomass,a,status),-0.29)*pow(gz/g0,1.15);
  }

  // Full Bullock et al. (2001)
  else if(iconc==2){    
    double A = 4.;
    double z = -1.+1./a;
    double zf = z_formation_Bullock(cosmo,halomass,a,status);
    return A*(1.+zf)/(1.+z);
  }

  // Duffy et al. (2008; 0804.2486; Table 1, third section: Delta = 200 mean)
  else if(iconc==3){
    double Mpiv=2e12/cosmo->params.h; //Pivot mass in Msun (note in the paper units are Msun/h)
    double A=10.14;
    double B=-0.081;
    double C=-1.01;
    return A*pow(halomass/Mpiv,B)*pow(a,-C); 
  }

  // Constant concentration (good for tests)
  else if(iconc==4){
    return 4.;
  }

  // Simple Bullock et al. (2001) relation
  else if(iconc==5){
    return 9.*pow(halomass/Mstar(),-0.13);
  }

  // Something went wrong
  else{    
    exit(0);
  }
	  
}

// Fourier transforms of halo profiles
double window_function(ccl_cosmology *cosmo, double m, double k, double a, int * status){

  //Window function for matter power spectrum with NFW haloes
  if(iwin==1){
    // The mean background matter density in Msun/Mpc^3
    double rho_matter = ccl_comoving_matter_density(cosmo);

    // The halo concentration for this mass and scale factor  
    double c = ccl_halo_concentration(cosmo,m,a,status);

    // The function U is normalised so multiplying by M/rho turns units to overdensity
    return m*u_nfw_c(cosmo,c,m,k,a,status)/rho_matter;
  }

  // Something went wrong
  else{
    exit(0);
  }
  
}

// Parameters structure for the one-halo integrand
typedef struct{  
  ccl_cosmology *cosmo;
  double k, a;
  int * status;
} Int_one_halo_Par;

// Integrand for the one-halo integral
double one_halo_integrand(double log10mass, void *params){  
  
  Int_one_halo_Par *p=(Int_one_halo_Par *)params;;
  double halomass = pow(10,log10mass);

  // The squared normalised Fourier Transform of a halo profile (W(k->0 = 1)
  double wk = window_function(p->cosmo,halomass,p->k,p->a,p->status);

  // Fairly sure that there should be no ln(10) factor should be here since the integration is being specified in log10 range
  double dn_dlogM = ccl_massfunc(p->cosmo,halomass,p->a,Delta_v,p->status);
    
  return dn_dlogM*pow(wk,2);
}

// The one-halo term integral using gsl
double one_halo_integral(ccl_cosmology *cosmo, double k, double a, int * status){
    
  int one_halo_integral_status = 0, qagstatus;
  double result = 0, eresult;
  double log10massmin = log10(mmin);
  double log10massmax = log10(mmax);
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
  qagstatus = gsl_integration_qag(&F, log10massmin, log10massmax, 0, 1E-4, 1000, GSL_INTEG_GAUSS41, w, &result, &eresult);

  // Clean up
  gsl_integration_workspace_free(w);

  return result;
}

// Parameters structure for the two-halo integrand
typedef struct{  
  ccl_cosmology *cosmo;
  double k, a;
  int * status;
} Int_two_halo_Par;

// Integrand for the two-halo integral
double two_halo_integrand(double log10mass, void *params){  
  
  Int_two_halo_Par *p=(Int_two_halo_Par *)params;
  double halomass = pow(10,log10mass);

  // The window function appropriate for the matter power spectrum
  //double wk = halomass*u_nfw_c(p->cosmo,c,halomass,p->k,p->a,p->status)/rho_matter
  double wk = window_function(p->cosmo,halomass,p->k,p->a,p->status);

  // Fairly sure that there should be no ln(10) factor should be here since the integration is being specified in log10 range
  double dn_dlogM = ccl_massfunc(p->cosmo,halomass,p->a,Delta_v,p->status);

  // Halo bias
  double b = ccl_halo_bias(p->cosmo,halomass,p->a,Delta_v,p->status);
    
  return b*dn_dlogM*wk;
}

// The two-halo term integral using gsl
double two_halo_integral(ccl_cosmology *cosmo, double k, double a, int * status){
    
  int two_halo_integral_status = 0, qagstatus;
  double result = 0, eresult;
  double log10massmin = log10(mmin);
  double log10massmax = log10(mmax);
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
  qagstatus = gsl_integration_qag(&F, log10massmin, log10massmax, 0, 1E-4, 1000, GSL_INTEG_GAUSS41, w, &result, &eresult);

  // Clean up
  gsl_integration_workspace_free(w);

  return result;
}

// Computes the two-halo term
double ccl_p_2h(ccl_cosmology *cosmo, double k, double a, int * status){

  // The standard formation of the two-halo term
  if(i2h==1){
    // Get the integral
    double I2h = two_halo_integral(cosmo, k, a, status);
    
    // The addative correction is the missing part of the integral below the lower-mass limit
    double A = 1.-two_halo_integral(cosmo, 0., a, status);
    //printf("A: %10.5f\n", A);

    // ...multiplied by the ratio of window functions
    A=A*window_function(cosmo, mmin, k, a, status)/window_function(cosmo, mmin, 0., a, status);    

    // Add the additive correction to the calculated integral
    I2h=I2h+A;
      
    return ccl_linear_matter_power(cosmo, k, a, status)*pow(I2h,2);
  }

  // Two-halo term is just linear theory in this case
  // This is okay for the matter spectrum, but not some other things
  else if(i2h==2){
    return ccl_linear_matter_power(cosmo, k, a, status);
  }

  //Something went wrong
  else{
    exit(0);
  }
  
}

// Computes the one-halo term
double ccl_p_1h(ccl_cosmology *cosmo, double k, double a, int * status){  
    return one_halo_integral(cosmo, k, a, status);
}

// Computes the full halo-model power
double ccl_p_halomod(ccl_cosmology *cosmo, double k, double a, int * status){  
  // Standard sum of two- and one-halo terms
  return ccl_p_2h(cosmo, k, a, status)+ccl_p_1h(cosmo, k, a, status);   
}
