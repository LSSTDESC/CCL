#include "ccl_neutrinos.h"
#include "ccl_utils.h"
#include "ccl_constants.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_const_mksa.h"
#include <gsl/gsl_roots.h>
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
    return 7./8.;
  }
  else if (mnuOT>CCL_NU_MNUT_MAX) {
    return 0.2776566337*mnuOT; 
  }
	
  // Evaluate the spline - this will use the accelerator if it has been defined.
  *status |= gsl_spline_eval_e(nu_spline, log(mnuOT),accel, &integral_value);

  return integral_value*7./8.;
}

/* -------- ROUTINE: Omeganuh2 ---------
INPUTS: a: scale factor, Neff: number of neutrino species, mnu: total mass in eV of neutrinos, TCMB: CMB temperature, accel: pointer to an accelerator which will evaluate the neutrino phasespace spline if defined, status: pointer to status integer.
TASK: Compute Omeganu * h^2 as a function of time.
!! To all practical purposes, Neff is simply N_nu_mass !!
*/

double ccl_Omeganuh2 (double a, double Neff, double* mnu, double TCMB, gsl_interp_accel* accel, int* status)
{
  double Tnu, a4, prefix_massless, mnuone, OmNuh2;
  double Tnu_eff, mnuOT, intval, prefix_massive;
  double total_mass; // To check if this is the massless or massive case.
  
  // First check if Neff if 0
  if (Neff==0) return 0.0;  
  
  Tnu=TCMB*pow(4./11.,1./3.);
  a4=a*a*a*a;  
  // Check if mnu=0. We assume that in the massless case mnu is a pointer to a single element and that element is 0. This should in principle never be called.
  if (mnu[0] < 1e-12) {
    prefix_massless = NU_CONST  * Tnu * Tnu * Tnu * Tnu; 
    return Neff*prefix_massless*7./8./a4;
  }
  
  // And the remaining massive case. If we've got this far, then Neff = Nnumass and this will be the number of elements in the array to which mnu points.
  // Tnu_eff is used in the massive case because CLASS uses an effective temperature of nonLCDM components to match to mnu / Omeganu =93.14eV. Tnu_eff = T_ncdm * TCMB = 0.71611 * TCMB
  Tnu_eff = Tnu * TNCDM / (pow(4./11.,1./3.));
  
  // Define the prefix using the effective temperature (to get mnu / Omega = 93.14 eV) for the massive case: 
  prefix_massive = NU_CONST * Tnu_eff * Tnu_eff * Tnu_eff * Tnu_eff;
  
  OmNuh2 = 0.; // Initialize to 0 - we add to this for each massive neutrino species.
  for(int i=0; i<Neff; i++){
	// Get mass over T (mass (eV) / ((kb eV/s/K) Tnu_eff (K)) 
	// This returns the density normalized so that we get nuh2 at a=0
	mnuOT = mnu[i] / (Tnu_eff/a) * (EV_IN_J / (KBOLTZ)); 
  
	// Get the value of the phase-space integral 
	intval=nu_phasespace_intg(accel,mnuOT, status);
	OmNuh2 = intval*prefix_massive/a4 + OmNuh2;
  }
  
  return OmNuh2;
}

/* -------- ROUTINE: Omeganuh2_to_Mnu ---------
INPUTS: OmNuh2: neutrino mass density today Omeganu * h^2, label: how you want to split up the masses, see ccl_neutrinos.h for options, TCMB: CMB temperature, accel: pointer to an accelerator which will evaluate the neutrino phasespace spline if defined, status: pointer to status integer.
TASK: Given Omeganuh2 today, the method of splitting into masses, and the temperature of the CMB, output a pointer to the array of neutrino masses (may be length 1 if label asks for sum) 
*/

double* ccl_nu_masses(double OmNuh2, ccl_nu_masses_label label, double TCMB, gsl_interp_accel* accel, int* status){
  
  double sumnu;
  
  sumnu = 93.14 * OmNuh2;
  
  // Now split the sum up into three masses depending on the label given:
  
  if(label==ccl_nu_masses_normal_label){
	  
     double *mnu;
	 mnu = malloc(3*sizeof(double));
			
	 if (sumnu>0.59){
		 mnu[0] = (sumnu - 0.59) / 3.;
		 mnu[1] = mnu[0] + 0.009;
		 mnu[2] = mnu[0] + 0.05;
	 }else{
		 *status = CCL_ERROR_MNU_UNPHYSICAL;
	 }
	 
	 ccl_check_status_nocosmo(status);
	 return mnu; 
	 
  } else if (label==ccl_nu_masses_inverted_label){

	double *mnu;
	mnu = malloc(3*sizeof(double));
	if (sumnu > 0.109){
	    mnu[0] = (sumnu+0.041) / 3.;
		mnu[1] = mnu[0] + 0.009;
		mnu[2] = mnu[0] - 0.05;
	}else{
		*status = CCL_ERROR_MNU_UNPHYSICAL;
	}
	
	ccl_check_status_nocosmo(status);
	return mnu; 
	
  } else if (label==ccl_nu_masses_equal_label){
	  
	  double *mnu;
	  mnu = malloc(3*sizeof(double));
	  mnu[0] = sumnu/3.;
	  mnu[1] = sumnu/3.;
	  mnu[2] = sumnu/3.;
      
      return mnu;
      
  } else if (label == ccl_nu_masses_sum_label){
	  
	  double *mnu;
	  mnu = malloc(sizeof(double));
	  
      mnu[1] = sumnu;
      
      return mnu;
      
  } else{
	  
	  printf("WARNING:  mass option = %d not yet supported\n continuing with normal hierarchy\n",label);
	  double *mnu;
	  mnu = malloc(3*sizeof(double));
			
	 if (sumnu>0.59){
		 mnu[0] = (sumnu - 0.59) / 3.;
		 mnu[1] = mnu[0] + 0.009;
		 mnu[2] = mnu[0] + 0.05;
	 }else{
		 *status = CCL_ERROR_MNU_UNPHYSICAL;
	 }
	 
	 ccl_check_status_nocosmo(status);
	 return mnu; 
  }	
}	  
