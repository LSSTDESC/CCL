#include "ccl_utils.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_interp.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_sf_expint.h"
//#include "gsl/gsl_interp2d.h"
//#include "gsl/gsl_spline2d.h"
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

// TODO: possibly need to correct unit errors.
// TODO: possible that delta should be passed around for consistency checks
static double r_delta(ccl_cosmology *cosmo, double halomass, double a, int * status){
  // Converts halo mass to rdelta.
  double rho_m, delta;
  rho_m = RHO_CRITICAL*cosmo->params.Omega_m*cosmo->params.h*cosmo->params.h;
  // delta = 200.0;
  return pow(halomass*3.0/(4.0*M_PI*rho_m*Delta_v()),1.0/3.0);
}

static double r_Lagrangian(ccl_cosmology *cosmo, double halomass, double a, int * status){
  // Calculates the halo Lagrangian radius as a function of halo mass
  double rho_m, delta;
  rho_m = RHO_CRITICAL*cosmo->params.Omega_m*cosmo->params.h*cosmo->params.h;
  // delta = 200.0;
  return pow(halomass*3.0/(4.0*M_PI*rho_m),1.0/3.0);
}

// QUESTION: Mead - Why is 'sinl' used in the below routine as well as 'sin'
double u_nfw_c(ccl_cosmology *cosmo, double c, double halomass, double k, double a, int * status){
  // analytic FT of NFW profile, from Cooray & Sheth (2002; Section 3 of https://arxiv.org/abs/astro-ph/0206508)
  double rs, ks;
  double f1, f2, f3, fc;
  //x = k * r_delta(cosmo, halomass, a, status)/c; // x = k*rv/c = k*rs = ks
  //xu = (1.+c)*x; // xu = ks*(1+c)
  rs = r_delta(cosmo, halomass, a, status)/c; //Scale radius for NFW
  ks = k*rs; //Dimensionless wave-number variable
  f1 = sin(ks)*(gsl_sf_Si(ks*(1.+c))-gsl_sf_Si(ks));
  f2 = cos(ks)*(gsl_sf_Ci(ks*(1.+c))-gsl_sf_Ci(ks));
  f3 = sinl(c*ks)/(ks*(1.+c));
  fc = log(1.+c)-c/(1.+c);
  return (f1+f2-f3)/fc;
}

// TODO: move this into ccl_massfunc as standard function conversion.
// Alternatively - just use this in ccl_massfunc as necessary
double nu(ccl_cosmology *cosmo, double halomass, double a, int * status) {
  return delta_c()/ccl_sigmaM(cosmo,halomass,a,status);
}

// TODO: make consistency check so that ccl_halo_concentration only runs if called with appropriate definition
// e.g. if Delta != 200 rho_{mean}, should not function.
double ccl_halo_concentration(ccl_cosmology *cosmo, double halomass, double a, int * status)
{
  // Bhattacharya et al. 2011, Delta = 200 rho_{mean} (Table 2)
  return 9.*pow(nu(cosmo,halomass,a,status),-.29)*pow(ccl_growth_factor(cosmo, a, status)/ccl_growth_factor(cosmo, 1.0, status),1.15);
  //return 10.14*pow(m/2.e+12,-0.081)*pow(a,1.01); //Duffy et al. 2008 (Delta = 200 mean)
}

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

// TODO: serious simplification of this function.
//double inner_I0j (ccl_cosmology *cosmo, double halomass, double k, double a, void *para, int * status){
//  double *array = (double *) para;
//  long double u = 1.0; //the number one?
//  double arr= array[6]; //Array of ...
//  double c = ccl_halo_concentration(cosmo, halomass,a, status); //The halo concentration for this mass and scale factor
//  int l;
//  int j = (int)(array[5]);
//  for (l = 0; l< j; l++){
//    u = u*u_nfw_c(cosmo, c, halomass, k, a, status);
//  }
// TODO: mass function should be the CCL call - check units due to changes (Msun vs Msun/h, etc)
//  return massfunc(nu(cosmo,halomass,a,status))*halomass*pow(halomass/(RHO_CRITICAL*cosmo->params.Omega_m),(double)j)*u;
//}

typedef struct{
  // Parameters for the I02 integrand
  ccl_cosmology *cosmo;
  double k, a;
  int * status;
} IntI02Par;


// QUESTION: Mead - Why is u = u_nfw*u_nfw rather than pow(u_nfw,2). This would save an evalation?
static double inner_I02(double logmass, void *params){
  // Integrand for the one-halo integral
  IntI02Par *p=(IntI02Par *)params;
  double u;
  double halomass = exp(logmass);
  double c = ccl_halo_concentration(p->cosmo,halomass,p->a,p->status); //The halo concentration for this mass and scale factor
  u = u_nfw_c(p->cosmo, c, halomass, p->k, p->a, p->status)*u_nfw_c(p->cosmo, c, halomass, p->k, p->a, p->status);
  // TODO: mass function should be the CCL call - check units due to changes (Msun vs Msun/h, etc)
  return massfunc_st(nu(p->cosmo,halomass,p->a,p->status))*halomass*pow(halomass/(RHO_CRITICAL*p->cosmo->params.Omega_m),2.0)*u;
}

double I02(ccl_cosmology *cosmo, double k, double a, int * status){
  // The actual one-halo integral
  int I02status=0, qagstatus;
  double result=0,eresult;
  double logmassmin=3; // should be set to something more reasonable
  double logmassmax=9; // also should be set to reasonable
  IntI02Par ipar;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);

  ipar.cosmo=cosmo;
  ipar.k=k;
  ipar.a=a;
  ipar.status=&I02status;
  F.function=&inner_I02;
  F.params=&ipar;

  qagstatus=gsl_integration_qag(&F,logmassmin,logmassmax,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);

  gsl_integration_workspace_free(w);

  return result;
}

// TODO: pass the cosmology construct around.
double p_1h(ccl_cosmology *cosmo, double k, double a, int * status){
  // Computes the one-halo integral
  return I02(cosmo, k, a, status);
}
