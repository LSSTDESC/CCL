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

// ---- LSST redshift distributions & current specs -----
// ---- Consider spline for input dN/dz - pending

//dN/dz for clustering sample
double dNdz_clustering(double z)
{
  //What is the redshift range of validity?
  double z0=0.3; //probably move this to the cosmo params file
  double zdivz0=z/z0;
  return 0.5/z0*zdivz0*zdivz0*exp(-zdivz0);
}

//sigma(z) photoz errors for clustering (assuming Gaussian)
double sigmaz_clustering(double z)
{
  return 0.03*(1.0+z);
}

//sigma(z) photoz errors for sources
double sigmaz_sources(double z)
{
  return 0.05*(1.0+z);
}

//Bias of the clustering sample
double bias_clustering(ccl_cosmology * cosmo, double a)
{
  //Growth is currently normalized to 1 today, is this what LSS needs?
  double D = ccl_growth_factor(cosmo, a);
  return 0.95/D;
}

// Fiducial (unnormalised) source redshift distribution from Chang et al. 2013 (k=1).
// THIS IS NOT THE UP TO DATE VERSION, JUST A BASIC FUNCTION IN CASE YOU DON'T WANT TO USE dNdz_sources_tomog BELOW
double dNdz_sources_basic(double z, void* params)
{
  // Argument void * params is not needed here but is required for gsl integration purposes, just pass NULL.
  double alpha=1.24; 
  double beta=1.01;
  double z0=0.51;
  double zdivz0=z/z0;
  double zmin_sources=0.1;
  double zmax_sources=3.0;

  if((z>=zmin_sources) && (z<=zmax_sources)){
    return pow(z,alpha)*exp(-pow(zdivz0,beta));
  } else {
    return 0.;
  }
}

// dNdz for weak lensing sources over the full allowed z range, not yet normalised, intended to be suitable for integration using gsl integration.
// This is not the function to call directly and use (that is dNdz_sources_tomog).
double dNdz_sources_unnormed(double z, void *params)
{
	double alpha, beta, z0, zdivz0;
	double zmin_sources = 0.1;
	double zmax_sources = 3.0;

	//type is passed as part of params, indicating which Chang et al 2013 dNdz we want.
	// type = 1 <-> k=0.5, type = 2 <-> k=1, type =3 <-> k=2.
	struct dNdz_sources_params * p = (struct dNdz_sources_params *) params;
	int type = p->type_;	

	if (type==1){
		// Chang et al. 2013, k=0.5	
		alpha=1.28;
  		beta=0.97;
  		z0=0.41;
	}else if (type ==2){
		// Chang et al. 2013, k=1
		alpha=1.24;
  		beta=1.01;
  		z0=0.51;
	}else if (type ==3){
		// Chang et al. 2013, k=2
		alpha=1.23;
  		beta=1.05;
  		z0=0.59;
	}else{
		printf("You specified an incorrect dNdz for lensing sources\n");
	}

	zdivz0= z/z0;

	if((z>=zmin_sources) && (z<=zmax_sources)){
    		return pow(z,alpha)*exp(-pow(zdivz0,beta));
  	}else{
    		return 0.;
  }
}

// dNdz of lensing sources in a particular tomographic bin, convolved with a photo-z model (defined in photoz function), and normalized.
double dNdz_sources_tomog(double z, void * dNdz_params, double bin_zmin, double bin_zmax){

	// This uses equation 33 of Joachimi & Schneider 2009, 0905.0393

	double *numerator_integrand, *denom_integrand;
        double init_num = 0.0; // Just to initialise the pointer to the answer.
	double init_denom = 0.0; 
	struct pz_params *pz_p, valparams; //parameters for the integral over the photoz's
	struct norm_params *norm_p, norm_p_val;
	struct dNdz_sources_params * dNdz_p = (struct dNdz_sources_params *) dNdz_params;

	// Set up the parameters to pass to the normalising integral (of type struct norm_params)
	norm_p_val.bin_zmin_=bin_zmin;
	norm_p_val.bin_zmax_=bin_zmax;
	norm_p_val.type_=dNdz_p->type_;
	norm_p=&norm_p_val;

	// Set up the parameters for the integral over the photo z function in the numerator (of type struct pz_params)
	valparams.z_true = z;
	pz_p = &valparams;
	
	// Integrate over the assumed pdf of photo-z wrt true-z in this bin (this goes in the numerator of the result):
	numerator_integrand = &init_num;
	gsl_integration_cquad_workspace * workspace_two = gsl_integration_cquad_workspace_alloc (1000);
        gsl_function G;
        G.function = photoz;
        G.params = pz_p;
        gsl_integration_cquad(&G, bin_zmin, bin_zmax, 0.0,EPSREL_GROWTH,workspace_two,numerator_integrand, NULL, NULL);
        gsl_integration_cquad_workspace_free(workspace_two);	

	// Now get the denominator, which normalizes dNdz over the photometric bin
	denom_integrand =&init_denom;
	gsl_integration_cquad_workspace * workspace_three = gsl_integration_cquad_workspace_alloc (1000);
        gsl_function H;
        H.function = norm_integrand;
        H.params = norm_p;
        gsl_integration_cquad(&H, 0.1, 3.0, 0.0,EPSREL_GROWTH,workspace_three,denom_integrand, NULL, NULL);
        gsl_integration_cquad_workspace_free(workspace_three);	

	return 	dNdz_sources_unnormed(z, dNdz_params) * (*numerator_integrand) / (*denom_integrand);
}

// Returns the value of p(z_photo, z). Change this function to change the way true-z and photo-z's are related.
// This has to be in a form that gsl can integrate
double photoz(double z_ph, void *params){
	
	struct pz_params * p = (struct pz_params *) params;
        double z_tr = p->z_true;
	double result;

	result = exp(- (z_ph-z_tr)*(z_ph-z_tr) / (2.*sigmaz_sources(z_tr)*sigmaz_sources(z_tr))) / (pow(2.*pi,0.5)*sigmaz_sources(z_tr)*sigmaz_sources(z_tr));	

	return result;
	}

// Returns the integrand which is integrated to get the normalization of dNdz in a given photometric redshift bin (the denominator from dNdz_sources_tomog).
// This has to be an separate function that gsl can integrate.
double norm_integrand(double z, void* params){
	
	struct pz_params *pz_p, valparams; // parameters for the photoz pdf wrt true-z
	double * pz_int; // pointer to the value of the integral over the photoz model
	double init; // just for initializing the result of the intermediary integratl
	struct norm_params *p = (struct norm_params *) params; // parameters of the current function (because of form required for gsl integration)

	double z_min = p->bin_zmin_;
	double z_max = p->bin_zmax_;
        int type = p->type_;

	// Extract "type" (which denotes which Chang et al 2013 dndz we use) and pass it to dNdz params.
	struct dNdz_sources_params * dNdz_pointer, dNdz_vals; // parameters of dNdz unnormalized function.
	dNdz_vals.type_=type;
	dNdz_pointer=&dNdz_vals;


	// Set up parameters for the intermediary integral.
	init=0;
	pz_int = &init;
	valparams.z_true = z;
        pz_p = &valparams;

	// Do the intermediary integral over the model relating  photo-z to true-z	
        gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc (1000);
        gsl_function F;
        F.function = photoz;
        F.params = pz_p;
        gsl_integration_cquad(&F, z_min, z_max, 0.0,EPSREL_GROWTH,workspace,pz_int, NULL, NULL);
        gsl_integration_cquad_workspace_free(workspace);

	// Now return this value with the value of dNdz at z, to be integrated itself elsewhere
	return dNdz_sources_unnormed(z, dNdz_pointer) * (*pz_int);

}

//----------------------------------------


//TODO: why is all of this not in ccl_power?


/*------ ROUTINE: Tsqr_BBKS ----- 
INPUT: ccl_parameters and k wavenumber in 1/Mpc
TASK: provide the square of the BBKS transfer function with baryonic correction
*/

static double Tsqr_BBKS(ccl_parameters * params, double k)
{
  double q = k/(params->Omega_m*params->h*params->h)*exp(-params->Omega_b*(1.0+pow(2.*params->h,.5)/params->Omega_m));
  return pow(log(1.+2.34*q)/(2.34*q),2.0)/pow(1.+3.89*q+pow(16.1*q,2.0)+pow(5.46*q,3.0)+pow(6.71*q,4.0),0.5);
}


/*------ ROUTINE: ccl_bbks_power ----- 
INPUT: ccl_parameters and k wavenumber in 1/Mpc
TASK: provide the BBKS power spectrum with baryonic correction at single k
*/

//Calculate Normalization see Cosmology Notes 8.105 (TODO: whose comment is this?)
double ccl_bbks_power(ccl_parameters * params, double k){
    return pow(k/K_PIVOT,params->n_s)*Tsqr_BBKS(params, k);
}


//-------------------- sigmaR for generic radius----------------------

struct sigmaR_args {
    gsl_spline* P;
    double R;
    int * status;
};


/* ------- ROUTINE: sigmaR_integrand ------- 
INPUT: k [1/Mpc]
TASK: give integrand for sigma_R
*/

static
double sigmaR_integrand(double k, void * args)
{
    struct sigmaR_args * s_args = (struct sigmaR_args*) args;
    gsl_spline * spline = s_args->P;
    double kR = k*s_args->R; // r=R in Mpc; k in 1/Mpc
    double p = exp(gsl_spline_eval(spline, log(k), NULL));
    double w,res;
    if(kR<0.1) {
      w =1.-0.1*kR*kR+0.003571429*kR*kR*kR*kR
        -6.61376E-5*kR*kR*kR*kR*kR*kR
        +7.51563E-7*kR*kR*kR*kR*kR*kR*kR*kR;
    }
    else
      w = 3.*(sin(kR) - kR*cos(kR))/(kR*kR*kR);
    res = p*w*w*k*k; 
    return res;
}

/* ------- ROUTINE: ccl_sigmaR ------- 
INPUT: matter power spectrum, R [Mpc],
TASK: give sigma_R (not the sqr)
*/
double ccl_sigmaR(gsl_spline * P, double R, int * status){
  
    struct sigmaR_args s_args;
    s_args.P = P;
    s_args.status = status;
    s_args.R = R; //in Mpc

    gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc (1000);
    
    gsl_function F;
    F.function = &sigmaR_integrand;
    F.params = &s_args;

    double sigma_R;
    *status |= gsl_integration_cquad(&F, K_MIN_INT, K_MAX_INT, 0.0, 1e-5, workspace, &sigma_R, NULL, NULL);
    gsl_integration_cquad_workspace_free(workspace);

    //TODO: I think there should be the sqrt here
    return sqrt(sigma_R/(2.*M_PI*M_PI));
}

/* ------- ROUTINE: ccl_sigma8 ------- 
INPUT: matter power spectrum, h,
TASK: give sigma_8 (not the sqr)
*/
double ccl_sigma8(gsl_spline * P, double h, int * status){
  return ccl_sigmaR(P,8/h,status);
}
