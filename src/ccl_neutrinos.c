#include "ccl_neutrinos.h"
#include "ccl_utils.h"
#include "ccl_constants.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_integration.h"
/* ------- ROUTINE: ccl_nu_integrand ------
INPUTS: x: dimensionless momentum, *r: pointer to a dimensionless mass / temperature
TASK: Integrand of phase-space massive neutrino integral
*/
static double ccl_nu_integrand(double x, void *r)
{
  double rat=*((double*)(r));
  return sqrt(x*x+rat*rat)/(exp(x)+1.0)*x*x; 
}

/* ------- ROUTINE: ccl_calculate_nu_phasespace_spline ------
TASK: Get the spline of the result of the phase-space integral required for massive neutrinos.
*/

gsl_spline* ccl_calculate_nu_phasespace_spline() {
  int N=CCL_NU_MNUT_N;
  double *mnut = ccl_linear_spacing(log(CCL_NU_MNUT_MIN),log(CCL_NU_MNUT_MAX),N);
  double *y=malloc(sizeof(double)*CCL_NU_MNUT_N);
  
  int status=0;
  gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc (1000);
  gsl_function F;
  F.function = &ccl_nu_integrand;
  for (int i=0; i<CCL_NU_MNUT_N; i++) {
    double mnut_=exp(mnut[i]);
    F.params = &(mnut_);
    status |= gsl_integration_cquad(&F, 0, 1000.0, 1e-7, 1e-7, workspace,&y[i], NULL, NULL); 
  }
  gsl_integration_cquad_workspace_free(workspace);
  double renorm=1./y[0];
  for (int i=0; i<CCL_NU_MNUT_N; i++) y[i]*=renorm;
  //  for (int i=0; i<CCL_NU_MNUT_N; i++) printf("%g %g \n",mnut[i],y[i]);
  gsl_spline* spl = gsl_spline_alloc(A_SPLINE_TYPE, CCL_NU_MNUT_N);
  status = gsl_spline_init(spl, mnut, y, CCL_NU_MNUT_N);
  // Check for errors in creating the spline
  if (status){
    free(mnut);
    free(y);
    gsl_spline_free(spl);
    fprintf(stderr, "Error creating mnu/T neutrino spline\n");
    return NULL;
  }
  free(mnut);
  free(y);
  return spl;
}

/* ------- ROUTINE: ccl_nu_phasespace_intg ------
INPUTS: spl: the gsl spline of the phase space integral, mnuOT: the dimensionless mass / temperature of a single massive neutrino
TASK: Get the value of the phase space integral at muOT
*/

double ccl_nu_phasespace_intg(gsl_spline* spl, double mnuOT) {
  if (mnuOT<CCL_NU_MNUT_MIN) return 7./8.;
  else if (mnuOT>CCL_NU_MNUT_MAX) return 0.2776566337*mnuOT; //evalf(45*Zeta(3)/(2*Pi^4));
  return gsl_spline_eval(spl, log(mnuOT),NULL)*7./8.;
}
/* -------- ROUTINE: Omeganuh2 ---------
INPUTS: a: scale factor, Neff: number of neutrino species, mnu: total mass in eV of neutrinos, TCMB: CMB temperature, psi: gsl spline of phase-space integral.
TASK: Compute the fractional energy density of neutrinos at scale factor a.
*/

double Omeganuh2 (double a, double Neff, double mnu, double TCMB, gsl_spline* psi) {
  if (Neff==0) return 0.0;
  double Tnu=TCMB*pow(4./11.,1./3.);
  // Tnu_eff is used in the massive case because CLASS uses an effective temperature of nonLCDM components to match to mnu / Omeganu =93.14eV. Tnu_eff = T_ncdm * TCMB = 0.71611 * TCMB
  double Tnu_eff = Tnu * 0.71611 / (pow(4./11.,1./3.)); 
  double a4=a*a*a*a;
  // prefix number is given by
  // type this into google:
  // 8*pi^5*(boltzmann constant)^4/(15*(h*c)^3))*(1 Kelvin)**4/(3*(100 km/s/Mpc)^2/(8*Pi*G)*(speed of light)^2)
  //
  //double prefix = 4.48130979e-7*Tnu*Tnu*Tnu*Tnu;
  // DL: I have updated the constant to use the exact same values used by CLASS, but it doesn't matter at a 10^{-4} level.
  double prefix_massless = 4.481627251529075e-7*Tnu*Tnu*Tnu*Tnu;
  
  // Massless case:
  if (mnu==0){ 
	return Neff*prefix_massless*7./8./a4;
  }
  // get the mass of one species
  double mnuone=mnu/Neff;
  // Get mass over T (mass (eV) / ((kb eV/s/K) Tnu_eff (K)) 
  // This returns the density at a normalized so that we get nuh2 at a=0
  // (1 eV) / (Boltzmann constant * 1 kelvin) = 11 604.5193 
  // DL: I have updated the constant to exactly match CLASS
  //double mnuOT=mnuone/(Tnu_eff/a)*11604.519; 
  double mnuOT=mnuone/(Tnu_eff/a)*11604.505289680865;
  // Get the value of the phase-space integral 
  double intval=ccl_nu_phasespace_intg(psi,mnuOT);
  // Define the prefix using the effective temperature (to get mnu / Omega = 93.14 eV) for the massive case:
  double prefix_massive = 4.481627251529075e-7*Tnu_eff*Tnu_eff*Tnu_eff*Tnu_eff; 
  return Neff*intval*prefix_massive/a4;
}

