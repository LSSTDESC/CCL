#include "ccl_core.h"
#include "ccl_utils.h"
#include "ccl_placeholder.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"
#include "ccl_background.h"

// ---- LSST redshift distributions & current specs -----
// ---- Tests pending 
// ---- Normalizations of dN/dz pending
// ---- Consider spline for input dN/dz - pending

//dN/dz for clustering sample
//static double dNdz_clustering(double z)
double dNdz_clustering(double z)
{
  //What is the redshift range of validity?
  double z0=0.3; //probably move this to the cosmo params file
  double zdivz0=z/z0;
  return 0.5/z0*zdivz0*zdivz0*exp(-zdivz0);
}

//This is the fiducial redshift distribution of Chang et al.
//Table 2, column corresponding to k=1 (fiducial case)
// This is unnormalised.
// static double dNdz_sources_k1(double z)
double dNdz_sources_k1(double z)
{
  double alpha=1.24; //These probably need to move to the cosmo params file
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

//This is a non-fiducial redshift distribution from Chang et al.
//Table 2, column corresponding to k=2
// This is unnormalised
// static double dNdz_sources_k2(double z)
double dNdz_sources_k2(double z)
{
  double alpha=1.23; //These probably need to move to the cosmo params file
  double beta=1.05;
  double z0=0.59;
  double zdivz0=z/z0;
  double zmin_sources=0.1;
  double zmax_sources=3.0;
  if((z>=zmin_sources) && (z<=zmax_sources)){
    return pow(z,alpha)*exp(-pow(zdivz0,beta));
  } else {
    return 0.;
  }
}


//This is a non-fiducial redshift distribution from Chang et al.
//Table 2, column corresponding to k=0.5
// This is unnormalised.
// static double dNdz_sources_k0pt5
double dNdz_sources_k0pt5(double z)
{
  double alpha=1.28; //These probably need to move to the cosmo params file
  double beta=0.97;
  double z0=0.41;
  double zdivz0=z/z0;
  double zmin_sources=0.1;
  double zmax_sources=3.0;
  if((z>=zmin_sources) && (z<=zmax_sources)){
    return pow(z,alpha)*exp(-pow(zdivz0,beta));
  } else {
    return 0.;
  }
}

//sigma(z) photoz errors for clustering (assuming Gaussian)
//static double sigmaz_clustering(double z)
double sigmaz_clustering(double z)
{
  return 0.03*(1.0+z);
}

//sigma(z) photoz errors for sources
//static double sigmaz_sources(double z)
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

// This is a toy photometric redshift model which assumes perfect photo-zs, to test dNdz_sources_tomog
double photoz_dNdz(double z, double (*dndz_func)(double))
{
return (*dndz_func)(z);
}

//dNdz in a redshift bin, for tomographic binning
// the output of this is not necessarily properly normalised
double dNdz_sources_tomog(double z, double zmin, double zmax, double (*dndz_func)(double), double (*photoz_func)(double, double (double) ))
{
  if ((z<=zmax) && (z>=zmin)){
     return (*photoz_func)(z, (*dndz_func));
  }else{
     return 0;
  } 
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
