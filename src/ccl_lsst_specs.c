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
#include "ccl_lsst_specs.h"

// ---- LSST redshift distributions & current specs -----
// ---- Consider spline for input dN/dz - pending

/*------ ROUTINE: dNdz_clustering -----
INPUT: double z
TASK: Return unnormalized dN/dz for clustering sample
TODO: Tomography/convolution with photo-z/redshift range of validity?
*/
double dNdz_clustering(double z, void* params)
{
  double z0=0.3; //probably move this to the cosmo params file
  double zdivz0=z/z0;
  return 0.5/z0*zdivz0*zdivz0*exp(-zdivz0);
}

/*------ ROUTINE: sigmaz_clustering -----
INPUT: double z
TASK: Return sigma(z), the photo-z dispersion, for the clustering sample
      We are assuming Gaussian uncertainties.
*/
double sigmaz_clustering(double z)
{
  return 0.03*(1.0+z);
}

/*------ ROUTINE: sigmaz_sources -----
INPUT: double z
TASK: Return sigma(z), the photo-z dispersion, for the lensing sample
      We are assuming Gaussian uncertainties.
*/
double sigmaz_sources(double z)
{
  return 0.05*(1.0+z);
}

/*------ ROUTINE: bias_clustering -----
INPUT: ccl_cosmology * cosmo, double a
TASK: Return b(z), the bias of the clustering sample.
      This is input from LSS group.
TODO: Check normalization of growth is consistent with LSS input.
*/
double bias_clustering(ccl_cosmology * cosmo, double a)
{
  double D = ccl_growth_factor(cosmo, a);
  return 0.95/D;
}

/*------ ROUTINE: dNdz_sources_unnormed -----
INPUT: double z, void* params
       void * params includes "type", indicating which Chang et al 2013 dNdz we want.
       type = 1 <-> k=0.5, type = 2 <-> k=1, type =3 <-> k=2.
TASK: dNdz for weak lensing sources over the full allowed z range, not yet normalised, 
      intended to be suitable for integration using gsl integration.
WARNING:  This is not the function to call directly and use (that is dNdz_sources_tomog).
TODO: if incorrect type, use ccl_error to exit.
*/

double dNdz_sources_unnormed(double z, void *params)
{
	double alpha, beta, z0, zdivz0;
 
	struct dNdz_sources_params * p = (struct dNdz_sources_params *) params;
	int type = p->type_;

	if (type==1){
		// Chang et al. 2013, k=0.5, pessimistic	
		alpha=1.28;
  		beta=0.97;
  		z0=0.41;
	}else if (type ==2){
		// Chang et al. 2013, k=1, fiducial
		alpha=1.24;
  		beta=1.01;
  		z0=0.51;
	}else if (type ==3){
		// Chang et al. 2013, k=2, optimistic
		alpha=1.23;
  		beta=1.05;
  		z0=0.59;
	}else{
		printf("You specified an incorrect dNdz for lensing sources\n");
	}

	zdivz0= z/z0;

	if((z>=z_min_sources) && (z<=z_max_sources)){
    		return pow(z,alpha)*exp(-pow(zdivz0,beta));
  	}else{
    		return 0.;
  }
}

/*------ ROUTINE: photoz -----
INPUT: double z_ph, void *params
TASK:  Returns the value of p(z_photo, z). Change this function to 
       change the way true-z and photo-z's are related.
       This has to be in a form that gsl can integrate.
*/
double photoz(double z_ph, void *params){
	
	struct pz_params * p = (struct pz_params *) params;
        double z_tr = p->z_true;
	double result;

	result = exp(- (z_ph-z_tr)*(z_ph-z_tr) / (2.*p->sigmaz(z_tr)*p->sigmaz(z_tr))) / (pow(2.*M_PI,0.5)*p->sigmaz(z_tr)*p->sigmaz(z_tr));	

	return result;
	}

/*------ ROUTINE: norm_integrand -----
INPUT: double z_ph, void *params
TASK:  Returns the integrand which is integrated to get the normalization of 
       dNdz in a given photometric redshift bin (the denominator from dNdz_sources_tomog). 
       This has to be an separate function that gsl can integrate.
*/
static double norm_integrand(double z, void* params){
	
	struct pz_params *pz_p, valparams; // parameters for the photoz pdf wrt true-z
	double * pz_int; // pointer to the value of the integral over the photoz model
	double init; // just for initializing the result of the intermediary integratl
	struct norm_params *p = (struct norm_params *) params; // parameters of the current function (because of form required for gsl integration)

	double z_min = p->bin_zmin_;
	double z_max = p->bin_zmax_;
        int type = p->type_;
	double (*sigmazin)(double) = p->sigmaz;

	// Extract "type" (which denotes which Chang et al 2013 dndz we use) 
        // and pass it to dNdz params.
	struct dNdz_sources_params * dNdz_pointer, dNdz_vals; // parameters of dNdz unnormalized function.
	dNdz_vals.type_=type;
	dNdz_pointer=&dNdz_vals;

	// Set up parameters for the intermediary integral.
	init=0;
	pz_int = &init;
	valparams.z_true = z;
	valparams.sigmaz = sigmazin;
        pz_p = &valparams;

	// Do the intermediary integral over the model relating  photo-z to true-z	
        gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc (1000);
        gsl_function F;
        F.function = photoz;
        F.params = pz_p;
        gsl_integration_cquad(&F, z_min, z_max, 0.0,EPSREL_DNDZ,workspace,pz_int, NULL, NULL);
        gsl_integration_cquad_workspace_free(workspace);

	// Now return this value with the value of dNdz at z, to be integrated itself elsewhere
	if ((dNdz_pointer-> type_!= DNDZ_NC) ){ 
		return p->unnormedfunc(z, dNdz_pointer) * (*pz_int);
	}else{
		return p->unnormedfunc(z,NULL) *(*pz_int);
	}
}


/*------ ROUTINE: dNdz_tomog -----
INPUT: double z, , double bin_zmin, double bin_zmax, dNdz function pointer, sigma_z function pointer
       tomographic boundaries are [bin_zmin,bin_zmax]
TASK:  dNdz in a particular tomographic bin, 
       convolved with a photo-z model (defined in photoz function), and normalized.
*/

//double dNdz_tomog(double z, int dNdz_type, double bin_zmin, double bin_zmax, double (*dNdz)(double,void *), double (*sigmazin)(double)){
double dNdz_tomog(double z, int dNdz_type, double bin_zmin, double bin_zmax){

	// This uses equation 33 of Joachimi & Schneider 2009, arxiv:0905.0393

	double *numerator_integrand, *denom_integrand, dNdz_t;
        double init_num = 0.0; // Just to initialise the pointer to the answer.
	double init_denom = 0.0; 
	struct pz_params *pz_p, valparams; //parameters for the integral over the photoz's
	struct norm_params *norm_p, norm_p_val;
	struct dNdz_sources_params *dNdz_p, dNdz_p_val; 

	// Set up the parameters to pass to the normalising integral (of type struct norm_params
	norm_p_val.bin_zmin_=bin_zmin;
	norm_p_val.bin_zmax_=bin_zmax;
	

	if((dNdz_type==DNDZ_WL_OPT) ||(dNdz_type==DNDZ_WL_FID) || (dNdz_type==DNDZ_WL_CONS)){ 
	  dNdz_p_val.type_ = dNdz_type;
	  norm_p_val.type_=dNdz_type;
	  norm_p_val.unnormedfunc=dNdz_sources_unnormed;
          norm_p_val.sigmaz=sigmaz_sources;
	} else {
	  norm_p_val.type_= dNdz_type;
	  norm_p_val.unnormedfunc = dNdz_clustering;
          norm_p_val.sigmaz=sigmaz_clustering;
	}
	norm_p=&norm_p_val;
	dNdz_p =&dNdz_p_val;

	// First get the value of dNdz(true) at z
	if((dNdz_type==DNDZ_WL_OPT) ||(dNdz_type==DNDZ_WL_FID) || (dNdz_type==DNDZ_WL_CONS)){
		dNdz_t = dNdz_sources_unnormed(z, dNdz_p);
	}else{
		dNdz_t = dNdz_clustering(z, NULL);
	}


	// Set up the parameters for the integral over the photo z function in the numerator (of type struct pz_params)
	valparams.z_true = z;
	if((dNdz_type==DNDZ_WL_OPT) ||(dNdz_type==DNDZ_WL_FID) || (dNdz_type==DNDZ_WL_CONS)){
                valparams.sigmaz = sigmaz_sources;
        }else{
                valparams.sigmaz = sigmaz_clustering;
        }

	//valparams.sigmaz = sigmazin;
	pz_p = &valparams;
	
	// Integrate over the assumed pdf of photo-z wrt true-z in this bin (this goes in the numerator of the result):
	numerator_integrand = &init_num;
	gsl_integration_cquad_workspace * workspace_two = gsl_integration_cquad_workspace_alloc (1000);
        gsl_function G;
        G.function = photoz;
        G.params = pz_p;
        gsl_integration_cquad(&G, bin_zmin, bin_zmax, 0.0,EPSREL_DNDZ,workspace_two,numerator_integrand, NULL, NULL);
        gsl_integration_cquad_workspace_free(workspace_two);	

	// Now get the denominator, which normalizes dNdz over the photometric bin
	denom_integrand =&init_denom;
	gsl_integration_cquad_workspace * workspace_three = gsl_integration_cquad_workspace_alloc (1000);
        gsl_function H;
        H.function = norm_integrand;
        H.params = norm_p;
        gsl_integration_cquad(&H, z_min_sources, z_max_sources, 0.0,EPSREL_DNDZ,workspace_three,denom_integrand, NULL, NULL);
        gsl_integration_cquad_workspace_free(workspace_three);	

	return 	dNdz_t * (*numerator_integrand) / (*denom_integrand);
}



