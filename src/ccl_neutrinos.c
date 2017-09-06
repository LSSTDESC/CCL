#include "ccl_neutrinos.h"
#include "ccl_utils.h"
#include "ccl_constants.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_const_mksa.h"
#include "ccl_error.h"
#include "ccl_core.h"

// Global variable to hold the neutrino phase-space spline
gsl_spline* nu_spline=NULL;

/* ------- ROUTINE: nu_integrand ------
INPUTS: x: dimensionless momentum, *r: pointer to a dimensionless mass / temperature
TASK: Integrand of phase-space massive neutrino integral
*/
static double nu_integrand(double x, void *r)
{
  double rat=*((double*)(r));
  return sqrt(x*x+rat*rat)/(exp(x)+1.0)*x*x; 
}

/* ------- ROUTINE: ccl_calculate_nu_phasespace_spline ------
TASK: Get the spline of the result of the phase-space integral required for massive neutrinos.
*/

gsl_spline* calculate_nu_phasespace_spline(int *status) {
  int N=CCL_NU_MNUT_N;
  double *mnut = ccl_linear_spacing(log(CCL_NU_MNUT_MIN),log(CCL_NU_MNUT_MAX),N);
  double *y=malloc(sizeof(double)*CCL_NU_MNUT_N);
  if (y ==NULL) {
    // Not setting a status_message here because we can't easily pass a cosmology to this function - message printed in ccl_error.c.
    *status = CCL_ERROR_NU_INT;
  }
  int stat=0;
  gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc (1000);
  gsl_function F;
  F.function = &nu_integrand;
  for (int i=0; i<CCL_NU_MNUT_N; i++) {
    double mnut_=exp(mnut[i]);
    F.params = &(mnut_);
    stat |= gsl_integration_cquad(&F, 0, 1000.0, 1e-7, 1e-7, workspace,&y[i], NULL, NULL); 
  }
  gsl_integration_cquad_workspace_free(workspace);
  double renorm=1./y[0];
  for (int i=0; i<CCL_NU_MNUT_N; i++)
    y[i]*=renorm;
  gsl_spline* spl = gsl_spline_alloc(A_SPLINE_TYPE, CCL_NU_MNUT_N);
  stat |= gsl_spline_init(spl, mnut, y, CCL_NU_MNUT_N);
  
  // Check for errors in creating the spline
  if (stat) {
    // Not setting a status_message here because we can't easily pass a cosmology to this function - message printed in ccl_error.c.  
    *status = CCL_ERROR_NU_INT;  
    free(mnut);
    free(y);
    gsl_spline_free(spl);
  }
  free(mnut);
  free(y);
  return spl;
}

/* ------- ROUTINE: ccl_nu_phasespace_intg ------
INPUTS: accel: pointer to an accelerator which will evaluate the neutrino phasespace spline if defined, mnuOT: the dimensionless mass / temperature of a single massive neutrino
TASK: Get the value of the phase space integral at mnuOT
*/

double nu_phasespace_intg(gsl_interp_accel* accel, double mnuOT, int* status)
{
  // Check if the global variable for the phasespace spline has been defined yet:
  if (nu_spline==NULL) nu_spline =calculate_nu_phasespace_spline(status);
  ccl_check_status_nocosmo(status);
  
  double integral_value =0.;
  
  // First check the cases where we are in the limits.
  if (mnuOT<CCL_NU_MNUT_MIN) {
    integral_value = 7./8.;
  }
  else if (mnuOT>CCL_NU_MNUT_MAX) {
    integral_value = 0.2776566337*mnuOT; 
  }
	
  // Evaluate the spline - this will use the accelerator if it has been defined.
  *status |= gsl_spline_eval_e(nu_spline, log(mnuOT),accel, &integral_value);

  return integral_value*7./8.;
}

/* -------- ROUTINE: Omeganuh2 ---------
INPUTS: a: scale factor, Neff: number of neutrino species, mnu: total mass in eV of neutrinos, TCMB: CMB temperature, accel: pointer to an accelerator which will evaluate the neutrino phasespace spline if defined, status: pointer to status integer.
TASK: Compute Omeganu * h^2 as a function of time.
*/
double Omeganuh2 (double a, double Neff, double mnu, double TCMB, gsl_interp_accel* accel, int* status)
{
  double Tnu, a4, prefix_massless, mnuone, OmNuh2;
  double Tnu_eff, mnuOT, intval, prefix_massive;
  
  // First check if Neff if 0
  if (Neff==0) return 0.0;
  
  // Now handle the massless case
  Tnu=TCMB*pow(4./11.,1./3.);
  a4=a*a*a*a;  
  if ( mnu < 1e-12) {
    prefix_massless = NU_CONST  * Tnu * Tnu * Tnu * Tnu; 
    return Neff*prefix_massless*7./8./a4;
  }
  
  // And the remaining massive case
  mnuone=mnu/Neff;  // Get the mass of one species (assuming equal-mass neutrinos).
  // Tnu_eff is used in the massive case because CLASS uses an effective temperature of nonLCDM components to match to mnu / Omeganu =93.14eV. Tnu_eff = T_ncdm * TCMB = 0.71611 * TCMB
  Tnu_eff = Tnu * TNCDM / (pow(4./11.,1./3.));
    
  // Get mass over T (mass (eV) / ((kb eV/s/K) Tnu_eff (K)) 
  // This returns the density normalized so that we get nuh2 at a=0
  mnuOT = mnuone / (Tnu_eff/a) * (EV_IN_J / (KBOLTZ)); 
  
  // Get the value of the phase-space integral 
  intval=nu_phasespace_intg(accel,mnuOT, status);
  
  // Define the prefix using the effective temperature (to get mnu / Omega = 93.14 eV) for the massive case: 
  prefix_massive = NU_CONST * Tnu_eff * Tnu_eff * Tnu_eff * Tnu_eff;
  
  OmNuh2 = Neff*intval*prefix_massive/a4;
  
  return OmNuh2;
}

