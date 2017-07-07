#include "ccl_core.h"
#include "ccl_utils.h"
#include "ccl_placeholder.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"
#include "ccl_background.h"
#include "ccl_constants.h"
#include "ccl_error.h"
#include "ccl_lsst_specs.h"
#include "ccl_params.h"

// ---- LSST redshift distributions & current specs -----
// ---- Consider spline for input dN/dz - pending

/*------ ROUTINE: ccl_specs_dNdz_clustering -----
INPUT: double z
TASK: Return unnormalized dN/dz for clustering sample
TODO: Tomography/convolution with photo-z/redshift range of validity?
*/
static double ccl_specs_dNdz_clustering(double z, void* params)
{
  double z0=0.3; //probably move this to the cosmo params file
  double zdivz0=z/z0;
  return 0.5/z0*zdivz0*zdivz0*exp(-zdivz0);
}

/*------ ROUTINE: ccl_specs_sigmaz_clustering -----
INPUT: double z
TASK: Return sigma(z), the photo-z dispersion, for the clustering sample
      This is if you want to assume Gaussian uncertainties.
*/
double ccl_specs_sigmaz_clustering(double z)
{
  return 0.03*(1.0+z);
}

/*------ ROUTINE: ccl_specs_sigmaz_sources -----
INPUT: double z
TASK: Return sigma(z), the photo-z dispersion, for the lensing sample
      This is if you want to assume Gaussian uncertainties.
*/
double ccl_specs_sigmaz_sources(double z)
{
  return 0.05*(1.0+z);
}

/*------ ROUTINE: ccl_specs_bias_clustering -----
INPUT: ccl_cosmology * cosmo, double a
TASK: Return b(z), the bias of the clustering sample.
      This is input from LSS group.
TODO: Check normalization of growth is consistent with LSS input.
*/
double ccl_specs_bias_clustering(ccl_cosmology * cosmo, double a, int * status)
{
  double D = ccl_growth_factor(cosmo, a, status);
  return 0.95/D;
}

/*------ ROUTINE: ccl_specs_dNdz_sources_unnormed -----
INPUT: double z, void* params
       void * params includes "type", indicating which Chang et al 2013 dNdz we want.
       type = 1 <-> k=0.5, type = 2 <-> k=1, type =3 <-> k=2.
TASK: dNdz for weak lensing sources over the full allowed z range, not yet normalised, 
      intended to be suitable for integration using gsl integration.
WARNING:  This is not the function to call directly and use (that is dNdz_sources_tomog).
TODO: if incorrect type, use ccl_error to exit.
*/

// struct of params to pass to ccl_specs_dNdz_sources_unnormed
struct dNdz_sources_params{
  int type_; // Sets which Chang et al. 2013 dNdz you are using; pick 1 for k=5, 2 for k=1, and 3 for k=2.
};

static double ccl_specs_dNdz_sources_unnormed(double z, void *params)
{
  double alpha, beta, z0=0.0, zdivz0;
  
  struct dNdz_sources_params * p = (struct dNdz_sources_params *) params;
  int type = p->type_;
  
  if (type==DNDZ_WL_CONS) {
    // Chang et al. 2013, k=0.5, pessimistic	
    alpha=1.28;
    beta=0.97;
    z0=0.41;
  } else if(type ==DNDZ_WL_FID) {
    // Chang et al. 2013, k=1, fiducial
    alpha=1.24;
    beta=1.01;
    z0=0.51;
  } else if(type ==DNDZ_WL_OPT) {
    // Chang et al. 2013, k=2, optimistic
    alpha=1.23;
    beta=1.05;
    z0=0.59;
  } 
  
  zdivz0= z/z0;
  
  if((z>=Z_MIN_SOURCES) && (z<=Z_MAX_SOURCES)) {
    return pow(z,alpha)*exp(-pow(zdivz0,beta));
  }else{
    return 0.;
  }
}

/*------ ROUTINE: ccl_specs_create_photoz_info ------
INPUT: void * user_pz_params, (double *) user_pz_func (double, double, void *)
TASK: create a structure amalgamating the user-input information on the photo-z model.
The structure holds a pointer to the function which returns the probability of getting a certain z_ph (first double) 
given a z_spec (second double), and a pointer to the parameters which get passed to that function (other than z_ph and z_sp); */
user_pz_info* ccl_specs_create_photoz_info(void * user_params,
					   double (*user_pz_func)(double, double,void*, int*))
{
  user_pz_info * this_user_info = malloc(sizeof(user_pz_info));
  this_user_info ->your_pz_params = user_params;
  this_user_info -> your_pz_func = user_pz_func;
  
  return this_user_info;
}


/* ------ ROUTINE: ccl_specs_free_photoz_info -------
INPUT: user_pz_info my_photoz_info
TASK: free memory holding the structure containing user-input photoz information */

void ccl_specs_free_photoz_info(user_pz_info *my_photoz_info)
{
  free(my_photoz_info);
}


/*------ ROUTINE: ccl_specs_photoz -----
INPUT: double z_ph, void *params
TASK:  Returns the value of p(z_photo, z). Change this function to 
       change the way true-z and photo-z's are related.
       This has to be in a form that gsl can integrate.
*/
// struct of parameters to pass to photo_z
struct pz_params{
  double z_true; // Gives the true redshift at which to evaluate 
  user_pz_info * user_information; //Calls the photo-z scatter model
  int *status;
};

static double ccl_specs_photoz(double z_ph, void * params)
{
  struct pz_params * p = (struct pz_params *) params;
  user_pz_info * user_stuff = (user_pz_info*) p->user_information; 
  double z_s = p->z_true;
  // user_stuff contains a pointer to the user function for the photo_z and to the user struct for the parameters of that function 
  //void * user_stuff = p->user_information;	
  
  return (user_stuff->your_pz_func)(z_ph, z_s, user_stuff->your_pz_params,p->status);
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
  int type_;
  user_pz_info * user_information;
  double (*unnormedfunc)(double,void *);
  int *status;
};


static double ccl_specs_norm_integrand(double z, void* params)
{
  // This is a struct that contains a true redshift and a pointer to the user_defined information about the photo_z model
  struct pz_params valparams; // parameters for the photoz pdf wrt true-z
  
  double pz_int=0; // pointer to the value of the integral over the photoz model
  struct norm_params *p = (struct norm_params *) params; // parameters of the current function (because of form required for gsl integration)
  
  double z_min = p->bin_zmin_;
  double z_max = p->bin_zmax_;
  int type = p->type_;
  
  // Extract "type" (which denotes which Chang et al 2013 dndz we use) 
  // and pass it to dNdz params.
  struct dNdz_sources_params dNdz_vals; // parameters of dNdz unnormalized function.
  dNdz_vals.type_=type;
  
  // Set up parameters for the intermediary integral.
  valparams.z_true = z;
  valparams.status = p->status;
  valparams.user_information = p-> user_information;
  
  // Do the intermediary integral over the model relating  photo-z to true-z	
  gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc (1000);
  gsl_function F;
  F.function = ccl_specs_photoz;
  F.params = &valparams;
  *p->status|= gsl_integration_cquad(&F, z_min, z_max, 0.0,EPSREL_DNDZ,workspace,&pz_int, NULL, NULL);
  gsl_integration_cquad_workspace_free(workspace);
  
  // Now return this value with the value of dNdz at z, to be integrated itself elsewhere
  if ((dNdz_vals.type_!= DNDZ_NC) ) {
    return p->unnormedfunc(z, &dNdz_vals) * pz_int;
  }
  else {
    return p->unnormedfunc(z,NULL) * pz_int;
  }
}

/*------ ROUTINE: ccl_specs_dNdz_tomog -----
INPUT: double z, , double bin_zmin, double bin_zmax, dNdz function pointer, sigma_z function pointer
       tomographic boundaries are [bin_zmin,bin_zmax]
TASK:  dNdz in a particular tomographic bin, 
       convolved with a photo-z model (defined by the user), and normalized.
       returns a status integer 0 if called with an allowable type of dNdz, non-zero otherwise
       (this is different from the regular status handling procedure because we don't pass a cosmology to this function)
*/
void ccl_specs_dNdz_tomog(double z, int dNdz_type, double bin_zmin, double bin_zmax,
			  user_pz_info * user_info, double *tomoout, int *status)
{
  // This uses equation 33 of Joachimi & Schneider 2009, arxiv:0905.0393
  double numerator_integrand=0, denom_integrand=0, dNdz_t;
  // This struct contains a spec redshift and a pointer to a user information struct.
  struct pz_params valparams; //parameters for the integral over the photoz's
  struct norm_params norm_p_val;	
  struct dNdz_sources_params dNdz_p_val; 
  
  // Set up the parameters to pass to the normalising integral (of type struct norm_params
  norm_p_val.bin_zmin_=bin_zmin;
  norm_p_val.bin_zmax_=bin_zmax;
  norm_p_val.user_information = user_info;	
  norm_p_val.status = status;	
  
  if((dNdz_type==DNDZ_WL_OPT) ||(dNdz_type==DNDZ_WL_FID) || (dNdz_type==DNDZ_WL_CONS)) {
    dNdz_p_val.type_ = dNdz_type;
    norm_p_val.type_=dNdz_type;
    norm_p_val.unnormedfunc=ccl_specs_dNdz_sources_unnormed;
  }
  else if (dNdz_type==DNDZ_NC) {
    norm_p_val.type_= dNdz_type;
    norm_p_val.unnormedfunc = ccl_specs_dNdz_clustering;
  }
  else {
    *status |= CCL_ERROR_PARAMETERS;
    return;
  }

  // First get the value of dNdz(true) at z
  if((dNdz_type==DNDZ_WL_OPT) ||(dNdz_type==DNDZ_WL_FID) || (dNdz_type==DNDZ_WL_CONS)) {
    dNdz_t = ccl_specs_dNdz_sources_unnormed(z, &dNdz_p_val);
  }
  else {
    dNdz_t = ccl_specs_dNdz_clustering(z, NULL);
  }

  // Set up the parameters for the integral over the photo z function in the numerator (of type struct pz_params)
  valparams.z_true = z;
  valparams.status = status;
  valparams.user_information = user_info; // pointer to user information
  
  
  // Integrate over the assumed pdf of photo-z wrt true-z in this bin (this goes in the numerator of the result):
  gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc (1000);
  gsl_function F;
  F.function = ccl_specs_photoz;
  F.params = &valparams;
  *status |=gsl_integration_cquad(&F, bin_zmin, bin_zmax, 0.0,EPSREL_DNDZ,workspace,&numerator_integrand, NULL, NULL);
  gsl_integration_cquad_workspace_free(workspace);	
  
  // Now get the denominator, which normalizes dNdz over the photometric bin
  workspace = gsl_integration_cquad_workspace_alloc (1000);
  F.function = ccl_specs_norm_integrand;
  F.params = &norm_p_val;
  *status |=gsl_integration_cquad(&F, Z_MIN_SOURCES, Z_MAX_SOURCES, 0.0,EPSREL_DNDZ,workspace,&denom_integrand, NULL, NULL);
  gsl_integration_cquad_workspace_free(workspace);
  if (*status) {
    *status = CCL_ERROR_INTEG;
    return;
  }
  *tomoout = dNdz_t * numerator_integrand / denom_integrand;
}
