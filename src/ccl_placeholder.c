#include "ccl_core.h"
#include "ccl_utils.h"
#include "ccl_placeholder.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"

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
