#include "ccl_utils.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_interp.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_sf_expint.h"
#include "ccl_placeholder.h"
#include "ccl_background.h"
#include "ccl_power.h"
#include "ccl_massfunc.h"
#include "ccl_error.h"
#include "class.h"
#include "ccl_params.h"
#include "ccl_emu17.h"
#include "ccl_emu17_params.h"

double delta_c(){
  // Linear collapse threshold for haloes
  return 1.686;
}

// TODO: referred to as odelta in ccl_massfunc, as a free parameter.
double Delta_v() {
  // Halo mean density
  return 200.;
}

// TODO: Check growth factor is normalised g(a=1)=1
double k_star(ccl_cosmology *cosmo, double a, int * status){
  // HMcode parameter: k*  
  double sigmaVz = ccl_sigmaV(cosmo, 0., status)*ccl_growth_factor(cosmo, a, status);
  return 0.584/sigmaVz;
}

// TODO: Check growth factor is normalised g(a=1)=1
double f_damp(ccl_cosmology *cosmo, double a, int * status){
  // HMcode parameter: f  
  double sigma8z = ccl_sigma8(cosmo, status)*ccl_growth_factor(cosmo, a, status);
  return 0.188*pow(sigma8z, 4.29);
}

//TODO: Actually code this up
double collapse_index(ccl_cosmology *cosmo, double a, int * status){
  // HMcode parameter: spectral index at the collapse scale, neff
  return -2.;
}

double alpha(ccl_cosmology *cosmo, double a, int * status){
  // HMcode parameter: alpha to transition between one- and two-halo regimes
  double neff = collapse_index(cosmo, a, status);
  return 2.93*pow(1.77,neff);
}

double comoving_matter_density(ccl_cosmology *cosmo){
  // The comoving density of matter. This is a constant. Units are Msun/Mpc^3 (no factors of h)
  return RHO_CRITICAL*cosmo->params.Omega_m*cosmo->params.h*cosmo->params.h;
}

// TODO: possible that delta should be passed around for consistency checks
double r_delta(ccl_cosmology *cosmo, double halomass, double a, int * status){  
  // Converts halo mass to rdelta.
  double rho_matter = comoving_matter_density(cosmo);  
  //rho_m = RHO_CRITICAL*cosmo->params.Omega_m*cosmo->params.h*cosmo->params.h;  
  return pow(halomass*3.0/(4.0*M_PI*rho_matter*Delta_v()),1.0/3.0);
}

double r_Lagrangian(ccl_cosmology *cosmo, double halomass, double a, int * status){
  // Calculates the halo Lagrangian radius as a function of halo mass
  double rho_matter = comoving_matter_density(cosmo);  
  return pow(halomass*3.0/(4.0*M_PI*rho_matter),1.0/3.0);
}

// TODO: Why is 'sinl' used in the below routine as well as 'sin'?
double u_nfw_c(ccl_cosmology *cosmo, double c, double halomass, double k, double a, int * status){
  
  // analytic FT of NFW profile, from Cooray & Sheth (2002; Section 3 of https://arxiv.org/abs/astro-ph/0206508)
  double rs, ks;
  double f1, f2, f3, fc;

  //Scale radius for NFW (rs=rv/c)
  rs = r_delta(cosmo, halomass, a, status)/c;

  //Dimensionless wave-number variable
  ks = k*rs;

  // Various bits for summing together to get final result
  f1 = sin(ks)*(gsl_sf_Si(ks*(1.+c))-gsl_sf_Si(ks));
  f2 = cos(ks)*(gsl_sf_Ci(ks*(1.+c))-gsl_sf_Ci(ks));
  f3 = sin(c*ks)/(ks*(1.+c));
  fc = log(1.+c)-c/(1.+c);
  
  return (f1+f2-f3)/fc;
}

// TODO: move this into ccl_massfunc as standard function conversion.
// Alternatively - just use this in ccl_massfunc as necessary
double nu(ccl_cosmology *cosmo, double halomass, double a, int * status) {
  return delta_c()/ccl_sigmaM(cosmo, halomass, a, status);
}

// TODO: Actually write this
double z_formation_Bullock(ccl_cosmology *cosmo, double halomass, double a, int * status){
  // Halo "formation redshift" calculated according to the Bullock et al. (2001) prescription
  return 0.;
}

// TODO: make consistency check so that ccl_halo_concentration only runs if called with appropriate definition
// TODO: should this be moved to the ccl_massfunc.c ?
// TODO: e.g. if Delta != 200 rho_{mean}, should not function (or should it?)
double ccl_halo_concentration(ccl_cosmology *cosmo, double halomass, double a, int * status)
{
  // Set concentration-mass relation
  // 1 - Bhattaharya et al. (2011)
  // 2 - Bullock et al. (2001)
  // 3 - Duffy et al. (2008)
  // 4 - Constant concentration (for testing)
  int iconc=1;

  // Bhattacharya et al. 2011, Delta = 200 rho_{mean} (Table 2)
  if(iconc==1){    
    double gz = ccl_growth_factor(cosmo,a,status);
    double g0 = ccl_growth_factor(cosmo,1.0,status);
    //return 9.*pow(nu(cosmo,halomass,a,status),-.29)*pow(ccl_growth_factor(cosmo, a, status)/ccl_growth_factor(cosmo, 1.0, status),1.15)
    return 9.*pow(nu(cosmo,halomass,a,status),-0.29)*pow(gz/g0,1.15);
  }

  // Bullock et al. (2001)
  else if(iconc==2){    
    double A = 4.;
    double z = -1.+1./a;
    double zf = z_formation_Bullock(cosmo,halomass,a,status);
    return A*(1.+zf)/(1.+z);
  }

  // Duffy et al. 2008 (Delta = 200 mean)
  else if(iconc==3){    
    return 10.14*pow(halomass/2.e+12,-0.081)*pow(a,1.01); 
  }

  // Constant concentration (good for tests)
  else if(iconc==4){
    return 100.;
  }

  // Something went wrong
  else{    
    exit(0);
  }
	  
}

typedef struct{
  // Parameters for the I02 integrand
  ccl_cosmology *cosmo;
  double k, a;
  int * status;
} IntI02Par;

double one_halo_integrand(double log10mass, void *params){  
  // Integrand for the one-halo integral
  IntI02Par *p=(IntI02Par *)params;
  double halomass = pow(10,log10mass);

  // The halo concentration for this mass and scale factor  
  double c = ccl_halo_concentration(p->cosmo,halomass,p->a,p->status); 

  // The squared normalised Fourier Transform of a halo profile (W(k->0 = 1)
  double wk_squared = pow(u_nfw_c(p->cosmo,c,halomass,p->k,p->a,p->status),2); 

  // The mean background matter density in Msun/Mpc^3
  //double rho_matter = RHO_CRITICAL*p->cosmo->params.Omega_m*pow(p->cosmo->params.h,2);
  double rho_matter = comoving_matter_density(p->cosmo);

  // Fairly sure that there should be no ln(10) factor should be here since the integration is being specified in log10 range
  double dn_dlogM = ccl_massfunc(p->cosmo,halomass,p->a,Delta_v(),p->status);
    
  return pow(halomass,2)*dn_dlogM*wk_squared/pow(rho_matter,2); 
}

double one_halo_integral(ccl_cosmology *cosmo, double k, double a, int * status){
  
  // The one-halo term integral using gsl
  int one_halo_integral_status = 0, qagstatus;
  double result = 0, eresult;
  double log10massmin = 7;
  double log10massmax = 17;
  IntI02Par ipar;
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

double p_1h(ccl_cosmology *cosmo, double k, double a, int * status){
  // Computes the one-halo term
  return one_halo_integral(cosmo, k, a, status);
}

double p_2h(ccl_cosmology *cosmo, double k, double a, int * status){
  // Computes the two-halo term (just the linear power at the moment)
  return ccl_linear_matter_power(cosmo, k, a, status);
}

double p_halomod(ccl_cosmology *cosmo, double k, double a, int * status){
  // Computes the full halo-model power
  return p_2h(cosmo, k, a, status)+p_1h(cosmo, k, a, status);
  //double alp = alpha(cosmo, a, status);
  //return pow(pow(p_2h(cosmo, k, a, status),alp)+pow(p_1h(cosmo, k, a, status),alp),1./alp);
}

/*
// TODO: move this into ccl_massfunc - be careful about the units!
// Keep for now for immediate testing.
double massfunc_st(double nu) {
// Sheth Tormen mass function!
// Note that nu=dc/sigma(M) and this Sheth & Tormen (1999) use nu=(dc/sigma)^2
// This accounts for some small differences
double p=0.3;
double q=0.707;
double A=0.21616;
return A*(1.+pow(q*nu*nu,-p))*exp(-q*nu*nu/2.);
}
*/

/*
  TODO: serious simplification of this function.
  double inner_I0j (ccl_cosmology *cosmo, double halomass, double k, double a, void *para, int * status){
  double *array = (double *) para;
  long double u = 1.0; //the number one?
  double arr= array[6]; //Array of ...
  double c = ccl_halo_concentration(cosmo, halomass,a, status); //The halo concentration for this mass and scale factor
  int l;
  int j = (int)(array[5]);
  for (l = 0; l< j; l++){
  u = u*u_nfw_c(cosmo, c, halomass, k, a, status);
  }
  TODO: mass function should be the CCL call - check units due to changes (Msun vs Msun/h, etc)
  return massfunc(nu(cosmo,halomass,a,status))*halomass*pow(halomass/(RHO_CRITICAL*cosmo->params.Omega_m),(double)j)*u;
  }
*/

/*
// QUESTION: Mead - Why is u = u_nfw*u_nfw rather than pow(u_nfw,2). This would save an evalation?
double inner_I02(double log10mass, void *params){
  
// Integrand for the one-halo integral
IntI02Par *p=(IntI02Par *)params;
double halomass = pow(log10mass,10);
double c = ccl_halo_concentration(p->cosmo,halomass,p->a,p->status); //The halo concentration for this mass and scale factor  
double Wk_squared = pow(u_nfw_c(p->cosmo, c, halomass, p->k, p->a, p->status),2);
double rho_matter = RHO_CRITICAL*p->cosmo->params.Omega_m*pow(p->cosmo->params.h,2); // QUESTION: What units does CCL use for this?
//double rho_matter = 2.775e11*p->cosmo->params.Omega_m*pow(p->cosmo->params.h,2); // Comoving matter density in Msun/Mpc^3
  
// TODO: mass function should be the CCL call - check units due to changes (Msun vs Msun/h, etc)
//return massfunc_st(nu(p->cosmo, halomass, p->a, p->status))*halomass*pow(halomass/(RHO_CRITICAL*p->cosmo->params.Omega_m),2.0)*Wk_squared;
return massfunc_st(nu(p->cosmo, halomass, p->a, p->status))*halomass/rho_matter;
}
*/

/*
  double I02(ccl_cosmology *cosmo, double k, double a, int * status){
  // The actual one-halo integral
  int I02status=0, qagstatus;
  double result=0, eresult;
  double log10massmin=10;
  double log10massmax=15;
  IntI02Par ipar;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);

  ipar.cosmo=cosmo;
  ipar.k=k;
  ipar.a=a;
  ipar.status=&I02status;
  F.function=&inner_I02;
  F.params=&ipar;

  qagstatus=gsl_integration_qag(&F, log10massmin, log10massmax, 0, 1E-4,1000, GSL_INTEG_GAUSS41, w, &result, &eresult);

  gsl_integration_workspace_free(w);

  return result;
  }
*/
