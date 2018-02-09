#include "ccl_core.h"
#include "ccl_utils.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_interp.h"
#include "gsl/gsl_spline.h"
//#include "gsl/gsl_interp2d.h"
//#include "gsl/gsl_spline2d.h"
#include "ccl_placeholder.h"
#include "ccl_background.h"
#include "ccl_power.h"
#include "ccl_error.h"
#include "class.h"
#include "ccl_params.h"
#include "ccl_emu17.h"
#include "ccl_emu17_params.h"

double u_nfw_c(double c,double k, double m, double aa){
  // analytic FT of NFW profile, from Cooray & Sheth 2001
  double x, xu;
  double f1, f2, f3;
  x = k * r_Delta(m,aa)/c; // x = k*rv/c = k*rs = ks
  xu = (1.+c)*x; // xu = ks*(1+c)
  f1 = sin(x)*(gsl_sf_Si(xu)-gsl_sf_Si(x));
  f2 = cos(x)*(gsl_sf_Ci(xu)-gsl_sf_Ci(x));
  f3 = sinl(c*x)/xu;
  fc = log(1.+c)-c/(1.+c);
  return (f1+f2-f3)/fc;
}

double inner_I0j (double logm, void *para){
  double *array = (double *) para;
  double m = exp(logm); //halo mass
  long double u = 1.0; //the number one?
  double a= array[6]; //Array of ...
  double c = conc(m,a); //The halo concentration for this mass and scale factor
  int l;
  int j = (int)(array[5]);
  for (l = 0; l< j; l++){
    u = u*u_nfw_c(c,array[l],m,a);
  }
  return massfunc(m,a)*m*pow(m/(cosmology.rho_crit*cosmology.Omega_m),(double)j)*u;
}

double I0j (int j, double k1, double k2, double k3, double k4,double a){
  double array[7] = {k1,k2,k3,k4,0.,(double)j,a};
  return int_gsl_integrate_medium_precision(inner_I0j,(void*)array,log(limits.M_min),log(limits.M_max),NULL, 2000);
}

double p_1h(double k, double a)
{
  return I0j(2,k,k,0.,0.,a);
}

double halo_concentration(double m, double a)
{
  return 9.*pow(nu(m,a),-.29)*pow(growfac(a)/growfac(1.),1.15);// Bhattacharya et al. 2011, Delta = 200 rho_{mean} (Table 2)
  //return 10.14*pow(m/2.e+12,-0.081)*pow(a,1.01); //Duffy et al. 2008 (Delta = 200 mean)
}

double massfunc(double nu) {
  //Sheth Tormen mass function!
  //Note that nu=dc/sigma(M) and this Sheth & Tormen (1999) use nu=(dc/sigma)^2
  //This accounts for some small differences
  double p=0.3;
  double q=0.707;
  double A=0.21616;
  return A*(1.+((q*nu*nu)**(-p)))*exp(-q*nu*nu/2.);
}

double delta_c() {
  //Linear collapse threshold
  return 1.686;
}

double Delta_v() {
  //Halo mean density
  return 200.;
}

double nu(ccl_cosmology *cosmo, double halomass, double a, int * status) {
  //nu = delta_c/sigma(M)
  return delta_c()/ccl_sigmaM(cosmo,halomass,a);
}
