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
  //DAVID: Based on Peacock and Dodds and the Dodelson book, this k should be in h/Mpc.
  //      double q = k/(params->Omega_m*params->h*params->h)*exp(-params->Omega_b*(1.0+pow(2.*params->h,.5)/params->Omega_m));
  //      Since we pass k in 1/Mpc, I correct for that k->k/h:
  double q = k/(params->Omega_m*params->h*params->h*params->h)*exp(-params->Omega_b*(1.0+pow(2.*params->h,.5)/params->Omega_m));
  //DAVID: above, can we cross-check the definition of q? I think the exponential has the wrong sign, also where did you take this expr from?
  return pow(log(1.+2.34*q)/(2.34*q),2.0)/pow(1.+3.89*q+pow(16.1*q,2.0)+pow(5.46*q,3.0)+pow(6.71*q,4.0),0.5);
  //DAVID: 16.1 -> 14.1 in Peacock and Dodds (1994) (?)
}


/*------ ROUTINE: ccl_bbks_power ----- 
INPUT: ccl_parameters and k wavenumber in 1/Mpc
TASK: provide the BBKS power spectrum with baryonic correction at single k
*/

//Calculate Normalization see Cosmology Notes 8.105 (TODO: whose comment is this?)
double ccl_bbks_power(ccl_parameters * params, double k){
    return pow(k/K_PIVOT,params->n_s)*Tsqr_BBKS(params, k);
}


//-------------------- sigma8 ----------------------

struct sigma8_args {
    gsl_spline* P;
    double h;
    int * status;
};

/* ------- ROUTINE: sigma8_integrand ------- 
INPUT: k [1/Mpc]
TASK: give the integrand of sigma_8^2
*/

static
double sigma8_integrand(double k, void * args)
{
    struct sigma8_args * s_args = (struct sigma8_args*) args;
    gsl_spline * spline = s_args->P;
    double kR = k*8.0/s_args->h; // r=8 Mpc/h
    double x = 3.*(sin(kR) - kR*cos(kR))/pow(kR,3.0);
    double p = exp(gsl_spline_eval(spline, log(k), NULL));
    double res = p*x*x*k*k; 
    return res;
}

/* ------- ROUTINE: ccl_sigma8 ------- 
INPUT: matter power spectrum, h
TASK: give sigma_8^2
*/

double ccl_sigma8(gsl_spline * P, double h, int * status){
  
    struct sigma8_args s_args;
    s_args.P = P;
    s_args.status = status;
    s_args.h = h;

    gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc (1000);
    
    gsl_function F;
    F.function = &sigma8_integrand;
    F.params = &s_args;

    double sigma_8;
    //TODO: Why not integrating in ln space?
    *status |= gsl_integration_cquad(&F, K_MIN_INT, K_MAX_INT, 0.0, 1e-5, workspace, &sigma_8, NULL, NULL);
    
    gsl_integration_cquad_workspace_free(workspace);

    //TODO: Printing sigma_8^2
    return sigma_8/(2.*M_PI*M_PI);
}

//-------------------- sigmaR for generic radius----------------------

struct sigmaR_args {
    gsl_spline* P;
    double R;
    double h;
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
    double x = 3.*(sin(kR) - kR*cos(kR))/pow(kR,3.0);
    double p = exp(gsl_spline_eval(spline, log(k), NULL)); 
    double res = p*x*x*k*k;
    return res;
}

/* ------- ROUTINE: ccl_sigmaR ------- 
INPUT: matter power spectrum, R [Mpc], h
TASK: give sigma_R (not the sqr)
*/

double ccl_sigmaR(gsl_spline * P, double R, double h, int * status){
  
    struct sigmaR_args s_args;
    s_args.P = P;
    s_args.status = status;
    s_args.h = h;
    s_args.R = R; //in Mpc

    gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc (1000);
    
    gsl_function F;
    F.function = &sigmaR_integrand;
    F.params = &s_args;

    double sigma_R;
    *status |= gsl_integration_cquad(&F, K_MIN_INT, K_MAX_INT, 0.0, 1e-5, workspace, &sigma_R, NULL, NULL);
    gsl_integration_cquad_workspace_free(workspace);

    //TODO: I think there should be the sqrt here
    return pow(sigma_R/(2.*M_PI*M_PI),0.5);
}
