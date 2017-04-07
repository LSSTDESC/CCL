#include "ccl_neutrinos.h"
#include "ccl_utils.h"
#include "ccl_constants.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_const_mksa.h"
#include "ccl_error.h"
#include "ccl_core.h"
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
  if (y ==NULL){
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
  for (int i=0; i<CCL_NU_MNUT_N; i++) y[i]*=renorm;
  gsl_spline* spl = gsl_spline_alloc(A_SPLINE_TYPE, CCL_NU_MNUT_N);
  stat |= gsl_spline_init(spl, mnut, y, CCL_NU_MNUT_N);
  
  // Check for errors in creating the spline
  if (stat){
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
INPUTS: spl: the gsl spline of the phase space integral, mnuOT: the dimensionless mass / temperature of a single massive neutrino
TASK: Get the value of the phase space integral at muOT
*/

double nu_phasespace_intg(gsl_spline* spl, double mnuOT) {
	
	double spline_val=0.;
	
	// First check the cases where we are in the limits.
    if (mnuOT<CCL_NU_MNUT_MIN){
		return 7./8.;
    }else if (mnuOT>CCL_NU_MNUT_MAX){
		return 0.2776566337*mnuOT; 
	}
	spline_val = gsl_spline_eval(spl, log(mnuOT),NULL)*7./8.;

  return spline_val;
}
/* -------- ROUTINE: Omeganuh2 ---------
INPUTS: a: scale factor, Neff: number of neutrino species, mnu: total mass in eV of neutrinos, TCMB: CMB temperature, psi: gsl spline of phase-space integral.
TASK: Compute Omeganu * h^2 as a function of time.
*/

double Omeganuh2 (double a, double Neff, double mnu, double TCMB, gsl_spline* psi) {
	
	double Tnu, a4, prefix_massless, mnuone;
	double Tnu_eff, mnuOT, intval, prefix_massive;
	
	// First check if Neff if 0
	if (Neff==0) return 0.0;
	
	// Now handle the massless case
	Tnu=TCMB*pow(4./11.,1./3.);
	a4=a*a*a*a;  
	if ( mnu < 0.00000000000001 ){
		prefix_massless = 8. * pow(M_PI,5) *pow((KBOLTZ/ HPLANCK),3)* KBOLTZ/(15. *pow( CLIGHT,3))* (8. * M_PI * GNEWT) / (3. * 100.*100.*1000.*1000. /MPC_TO_METER /MPC_TO_METER  * CLIGHT * CLIGHT)  * Tnu * Tnu * Tnu * Tnu; 
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
    intval=nu_phasespace_intg(psi,mnuOT);
		
    // Define the prefix using the effective temperature (to get mnu / Omega = 93.14 eV) for the massive case: 
    prefix_massive = 8. * pow(M_PI,5) *pow((KBOLTZ/ HPLANCK),3)* KBOLTZ/(15. *pow( CLIGHT,3))* (8. * M_PI * GNEWT) / (3. * 100.*100.*1000.*1000. /MPC_TO_METER /MPC_TO_METER  * CLIGHT * CLIGHT) * Tnu_eff * Tnu_eff * Tnu_eff * Tnu_eff;
    
    return Neff*intval*prefix_massive/a4;
}

