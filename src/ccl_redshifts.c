#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include "ccl.h"
#include "ccl_params.h"
#include "ccl_redshifts.h"

// ---- LSST redshift distributions & current specs -----
// ---- Consider spline for input dN/dz - pending

/*------ ROUTINE: ccl_specs_bias_clustering -----
INPUT: ccl_cosmology * cosmo, double a, double par 
TASK: Return b(z), the bias of the clustering sample.
      This is input from LSS group.
      par is empirical and sample-dependent.
TODO: Check normalization of growth is consistent with LSS input.
*/
double ccl_specs_bias_clustering(ccl_cosmology * cosmo, double a, double par, int * status)
{
  double D = ccl_growth_factor(cosmo, a, status);
  return par/D;
}

/*------ ROUTINE: ccl_create_photoz_info ------
INPUT: void * params, (double *) pz_func (double, double, void *)
TASK: create a structure amalgamating the user-input information on the photo-z model.
The structure holds a pointer to the function which returns the probability of getting a certain z_ph (first double) 
given a z_spec (second double), and a pointer to the parameters which get passed to that function (other than z_ph and z_sp); */
pz_info* ccl_create_photoz_info(void * params,
					   double (*pz_func)(double, double,void*, int*))
{
  pz_info * this_info = malloc(sizeof(pz_info));
  this_info ->your_pz_params = params;
  this_info -> your_pz_func = pz_func;
  
  return this_info;
}

/*------ ROUTINE: ccl_photoz -----
INPUT: double z_ph, void *params
TASK:  Returns the value of p(z_photo, z). Change this function to 
       change the way true-z and photo-z's are related.
       This has to be in a form that gsl can integrate.
*/
// struct of parameters to pass to photo_z
struct pz_params{
  double z_true; // Gives the true redshift at which to evaluate 
  pz_info * pz_information; //Calls the photo-z scatter model
  int *status;
};

static double ccl_photoz(double z_ph, void * params)
{
  struct pz_params * p = (struct pz_params *) params;
  pz_info * user_stuff = (pz_info*) p->pz_information; 
  double z_s = p->z_true;	
  
  return (user_stuff->your_pz_func)(z_ph, z_s, user_stuff->your_pz_params,p->status);
}

// Gaussian photo-z function
double gaussian_pz(double z_ph, double z_s, void* params, int *status){
    double sigma_z0 = *((double*) params);

    double sigma_z = sigma_z0 * (1. + z_s);
    return exp(- (z_ph - z_s)*(z_ph - z_s) / (2.*sigma_z*sigma_z)) \
         / (sqrt(2.*M_PI) *sigma_z);
}

/*------ ROUTINE: ccl_specs_create_gaussian_photoz_info ------
INPUT: void * user_pz_params, (double *) user_pz_func (double, double, void *)
TASK: Convenience function for creating a Gaussian photo-z model with error
sigma(z) = sigma_z0 (1 + z). */

pz_info* ccl_create_gaussian_photoz_info(double sigma_z0){
    
    // Allocate memory so that this value persists
    double* sigma_z0_copy = malloc(sizeof(double));
    *sigma_z0_copy = sigma_z0;
    
    // Construct pz_info struct
    pz_info * this_info = malloc(sizeof(pz_info));
    this_info->your_pz_params = sigma_z0_copy;
    this_info->your_pz_func = &gaussian_pz;
    return this_info;
}

/* ------ ROUTINE: ccl_free_photoz_info -------
INPUT: pz_info my_photoz_info
TASK: free memory holding the structure containing user-input photoz information */

void ccl_free_photoz_info(pz_info *my_photoz_info)
{
  free(my_photoz_info);
}

/*------ ROUTINE: ccl_create_dNdz_info ------
INPUT: void * params, (double *) dNdz_func (double, void *, int*)
TASK: create a structure amalgamating the information on an analytic true dNdz model.
The structure holds a pointer to the function which returns the analytic dNdz 
* and a pointer to the parameters which get passed to that function (other than z); */
dNdz_info* ccl_create_dNdz_info(void * params, double(*dNdz_func)(double,void*,int*))
{
  dNdz_info * this_info = malloc(sizeof(dNdz_info));
  this_info ->your_dN_params = params;
  this_info -> your_dN_func = dNdz_func;
  
  return this_info;
}

/*------ ROUTINE: dNdz_smail ----------
 * INPUT: z, params (containing: alpha, beta, z0), status
 * OUTPUT: Analytic Smail et al. type dNdz (NOT normalized) */

double dNdz_smail(double z, void* params, int *status){
    double alpha = ((smail_params*) params)->alpha;
    double beta = ((smail_params*) params)->beta;
    double z0 = ((smail_params*) params)->z0;

    return pow(z, alpha) * exp(- pow(z/z0, beta) );
}

/*------ ROUTINE: ccl_create_smail_dNdz_info ------
INPUT: alpha, beta, z0
TASK: Convenience function for creating an analytic dNdz wrt true z
* of the 'smail' form: dNdz ~ z^alpha exp(- (z/z0)^beta) */

dNdz_info* ccl_create_Smail_dNdz_info(double alpha, double beta, double z0){
    
    // Allocate a smail type parmaeter structure
    smail_params * smail_par = malloc(sizeof(smail_params));
    smail_par->alpha = alpha;
    smail_par->beta = beta;
    smail_par->z0 = z0;
    
    // Construct dNdz_info struct
    dNdz_info * this_info = malloc(sizeof(dNdz_info));
    this_info->your_dN_params = smail_par;
    this_info->your_dN_func = &dNdz_smail;
    return this_info;
}

/*------ ROUTINE: ccl_dNdz -----
INPUT: double z, void *params
TASK:  Returns the value of dNdz(z). Change this function to 
       change the way true-z and photo-z's are related.
       This has to be in a form that gsl can integrate.
*/

static double ccl_dNdz(double z, dNdz_info*  params, int* status)
{  
  return (params->your_dN_func)(z, params->your_dN_params, status);
}

/* ------ ROUTINE: ccl_free_dNdz_info -------
INPUT: dNdz_info my_dNdz_info
TASK: free memory holding the structure containing dNdz information */

void ccl_free_dNdz_info(dNdz_info *my_dNdz_info)
{
  free(my_dNdz_info);
}

/*------ ROUTINE: ccl_specs_norm_integrand -----
INPUT: double z_ph, void *params
TASK:  Returns the integrand which is integrated to get the normalization of 
       dNdz in a given photometric redshift bin (the denominator from dNdz_sources_tomog). 
       This has to be an separate function that gsl can integrate.
*/

// struct of parameters to pass to norm_integrand
struct norm_params {
  double bin_zmin_;
  double bin_zmax_;
  pz_info * pz_information;
  dNdz_info * dN_information;
  int *status;
};

static double ccl_norm_integrand(double z, void* params)
{
  // This is a struct that contains a true redshift and a pointer to the information about the photo_z model
  struct pz_params pz_val_p; // parameters for the photoz pdf wrt true-z
  
  // This is a struct that contains a true redshift and a pointer to the information about the analytic dNdz
  //struct dN_params dN_val_p;
  
  double pz_int=0; // pointer to the value of the integral over the photoz model
  struct norm_params *p = (struct norm_params *) params; // parameters of the current function (because of form required for gsl integration)
  
  double z_min = p->bin_zmin_;
  double z_max = p->bin_zmax_;
  
  // Set up parameters for the pz part of the intermediary integral.
  pz_val_p.z_true = z;
  pz_val_p.status = p->status;
  pz_val_p.pz_information = p-> pz_information;
  
  // Do the intermediary integral over the model relating  photo-z to true-z	
  gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc(ccl_gsl->N_ITERATION);
  gsl_function F;
  F.function = ccl_photoz;
  F.params = &pz_val_p;
  int gslstatus = gsl_integration_cquad(&F, z_min, z_max, 0.0,ccl_gsl->INTEGRATION_DNDZ_EPSREL,workspace,&pz_int, NULL, NULL);
  if(gslstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_redshifts.c: ccl_specs_norm_integrand():");
    *p->status|= gslstatus;
  }
  gsl_integration_cquad_workspace_free(workspace);
 
  return ccl_dNdz(z, p->dN_information, p->status);
}

/*------ ROUTINE: ccl_specs_dNdz_tomog -----
INPUT: double z, , double bin_zmin, double bin_zmax, dNdz function pointer, sigma_z function pointer
       tomographic boundaries are [bin_zmin,bin_zmax]
TASK:  dNdz in a particular tomographic bin, 
       convolved with a photo-z model (defined by the user), and normalized.
       returns a status integer 0 if called with an allowable type of dNdz, non-zero otherwise
       (this is different from the regular status handling procedure because we don't pass a cosmology to this function)
*/
void ccl_dNdz_tomog(double z, double bin_zmin, double bin_zmax, 
              pz_info * photo_info,  dNdz_info * dN_info, double *tomoout, int *status)
{
  // This uses equation 33 of Joachimi & Schneider 2009, arxiv:0905.0393
  double numerator_integrand=0, denom_integrand=0, dNdz_t;
  // This struct contains a spec redshift and a pointer to a photoz information struct.
  struct pz_params pz_p_val; //parameters for the integral over the photoz's
  struct norm_params norm_p_val;	
  //struct dN_params dN_p_val; 
  
  // Set up the parameters to pass to the normalising integral (of type struct norm_params
  norm_p_val.bin_zmin_=bin_zmin;
  norm_p_val.bin_zmax_=bin_zmax;
  norm_p_val.pz_information = photo_info;	
  norm_p_val.status = status;
  
  dNdz_t = ccl_dNdz(z, dN_info, status);
  
  // Set up the parameters for the integral over the photo z function in the numerator (of type struct pz_params)
  pz_p_val.z_true = z;
  pz_p_val.status = status;
  pz_p_val.pz_information = photo_info; // pointer to user information
  
  // Integrate over the assumed pdf of photo-z wrt true-z in this bin (this goes in the numerator of the result):
  gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc(ccl_gsl->N_ITERATION);
  gsl_function F;
  F.function = ccl_photoz;
  F.params = &pz_p_val;
  int gslstatus = gsl_integration_cquad(&F, bin_zmin, bin_zmax, 0.0,ccl_gsl->INTEGRATION_DNDZ_EPSREL,workspace,&numerator_integrand, NULL, NULL);
  if(gslstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_redshifts.c: ccl_specs_norm_integrand():");
    *status |= gslstatus;
  }  
  gsl_integration_cquad_workspace_free(workspace);	
  
  // Now get the denominator, which normalizes dNdz over the photometric bin
  workspace = gsl_integration_cquad_workspace_alloc(ccl_gsl->N_ITERATION);
  F.function = ccl_norm_integrand;
  F.params = &norm_p_val;
  gslstatus = gsl_integration_cquad(&F, Z_MIN_SOURCES, Z_MAX_SOURCES, 0.0,ccl_gsl->INTEGRATION_DNDZ_EPSREL,workspace,&denom_integrand, NULL, NULL);
  if(gslstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_redshifts.c: ccl_specs_norm_integrand():");
    *status |= gslstatus;
  } 
  gsl_integration_cquad_workspace_free(workspace);
  if (*status) {
    *status = CCL_ERROR_INTEG;
    return;
  }
  *tomoout = dNdz_t * numerator_integrand / denom_integrand;
}
