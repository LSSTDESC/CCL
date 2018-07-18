#include "ccl_utils.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_sf_expint.h"
#include "ccl_background.h"
#include "ccl_power.h"
#include "ccl_massfunc.h"

// Cosmology dependence of the virial collapse density according to the spherical-collapse model
// Fitting function from Bryan & Norman (1998; arXiv:astro-ph/9710107)
// Here, this is defined relative to the background matter density, not the critical density
static double Dv_BryanNorman(ccl_cosmology *cosmo, double a, int *status){
  double Om_mz = ccl_omega_x(cosmo, a, ccl_species_m_label, status);
  double x = Om_mz-1.;
  double Dv0 = 18.*pow(M_PI,2);
  double Dv = (Dv0+82.*x-39.*pow(x,2))/Om_mz;
  return Dv;
}

// analytic FT of NFW profile, from Cooray & Sheth (2002; Section 3 of https://arxiv.org/abs/astro-ph/0206508)
// Normalised such that U(k=0)=1
static double u_nfw_c(ccl_cosmology *cosmo, double c, double halomass, double k, double a, int *status){
   
  double rv, rs, ks, Dv;
  double f1, f2, f3, fc;
  double Delta_v = Dv_BryanNorman(cosmo, a, status); // Virial density of haloes

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
static double z_formation_Bullock(ccl_cosmology *cosmo, double halomass, double a, int *status){ 
  return 0.;
}

// The non-linear mass (sigma(M*)=delta_c))
// TODO: Actually code this up
static double Mstar(){
  return 1e13;
}

// The concentration-mass relation for haloes
// TODO: make consistency check so that ccl_halo_concentration only runs if called with appropriate definition
// TODO: e.g. if Delta != 200 rho_{mean}, should not function (or should it?)
// TODO: should this be moved to the ccl_massfunc.c ?
double ccl_halo_concentration(ccl_cosmology *cosmo, double halomass, double a, int * status)
{

  // Set concentration-mass relation
  // 1 - Bhattaharya et al. (2011)
  // 2 - Full Bullock et al. (2001)
  // 3 - Virial Duffy et al. (2008)
  // 4 - Constant concentration (useful for testing)
  // 5 - Simple Bullock et al. (2001)
  int iconc=3;
  
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

  // Duffy et al. (2008; 0804.2486; Table 1, second section: Delta = Virial)
  else if(iconc==3){
    double Mpiv = 2e12/cosmo->params.h; // Pivot mass in Msun (note in the paper units are Msun/h)
    double A = 7.85;
    double B = -0.081;
    double C = -0.71;
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

// Fourier Transforms of halo profiles
static double window_function(ccl_cosmology *cosmo, double m, double k, double a, int *status){

  // Select window function
  // 1 - NFW profile, appropriate for matter power spectrum
  int iwin=1;

  //Window function for matter power spectrum with NFW haloes
  if(iwin==1){
    // The mean background matter density in Msun/Mpc^3
    //double rho_matter = ccl_comoving_matter_density(cosmo);
    double rho_matter = ccl_rho_x(cosmo, 1., ccl_species_m_label, 1, status);

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
static double one_halo_integrand(double log10mass, void *params){  
  
  Int_one_halo_Par *p=(Int_one_halo_Par *)params;;
  double halomass = pow(10,log10mass);
  double Delta_v = Dv_BryanNorman(p->cosmo, p->a, p->status); // Virial density of haloes

  // The squared normalised Fourier Transform of a halo profile (W(k->0 = 1)
  double wk = window_function(p->cosmo,halomass,p->k,p->a,p->status);

  // Fairly sure that there should be no ln(10) factor should be here since the integration is being specified in log10 range
  double dn_dlogM = ccl_massfunc(p->cosmo,halomass,p->a,Delta_v,p->status);
    
  return dn_dlogM*pow(wk,2);
}

// The one-halo term integral using gsl
static double one_halo_integral(ccl_cosmology *cosmo, double k, double a, int *status){

  double mmin=1e7; // Minimum mass for the halo-model integration
  double mmax=1e17; // Maximum mass for the halo-model integration
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
static double two_halo_integrand(double log10mass, void *params){  
  
  Int_two_halo_Par *p=(Int_two_halo_Par *)params;
  double halomass = pow(10,log10mass);
  double Delta_v=Dv_BryanNorman(p->cosmo, p->a, p->status);

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
static double two_halo_integral(ccl_cosmology *cosmo, double k, double a, int *status){

  double mmin=1e7; // Minimum mass for the halo-model integration
  double mmax=1e17; // Maximum mass for the halo-model integration
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
double ccl_twohalo_matter_power(ccl_cosmology *cosmo, double k, double a, int *status){

  // Set two-halo term
  // 1 - Standard two-halo term
  // 2 - Linear theory
  int i2h=1;

  double mmin=1e7; //Minimum mass for the integrals

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
double ccl_onehalo_matter_power(ccl_cosmology *cosmo, double k, double a, int *status){  
    return one_halo_integral(cosmo, k, a, status);
}

// Computes the full halo-model power
double ccl_halomodel_matter_power(ccl_cosmology *cosmo, double k, double a, int *status){  
  // Standard sum of two- and one-halo terms
  return ccl_twohalo_matter_power(cosmo, k, a, status)+ccl_onehalo_matter_power(cosmo, k, a, status);   
}
