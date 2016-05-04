#include "ccl_core.h"
#include "ccl_utils.h"
#include "ccl_placeholder.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"

//TODO: why is all of this not in ccl_power?
static
double Tsqr_BBKS(ccl_parameters * params, double k)
{
    double q = k/(params->Omega_m*params->h*params->h)*exp(-params->Omega_b*(1.0+pow(2.*params->h,.5)/params->Omega_m));
    return pow(log(1.+2.34*q)/(2.34*q),2.0)/pow(1.+3.89*q+pow(16.1*q,2.0)+pow(5.46*q,3.0)+pow(6.71*q,4.0),0.5);
}


//Calculate Normalization see Cosmology Notes 8.105
double ccl_bbks_power(ccl_parameters * params, double k){
    return pow(k/K_PIVOT,params->n_s)*Tsqr_BBKS(params, k);
}

struct sigma8_args {
    gsl_spline* P;
    double h;
    int * status;
};

//TODO: couldn't this be done from a generic sigma(R) routine?
static
double sigma8_integrand(double k, void * args)
{
    struct sigma8_args * s_args = (struct sigma8_args*) args;
    gsl_spline * spline = s_args->P;
    double kR = k*8.0*s_args->h; // r=8 Mpc/h
    double x = 3.*(sin(kR) - kR*cos(kR))/pow(kR,3.0);
    double p = exp(gsl_spline_eval(spline, log(k), NULL));
    double res = p*x*x*k*k/(2.*M_PI*M_PI);
    return res;
}

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
    *status |= gsl_integration_cquad(&F, K_MIN*1.1, K_MAX/1.1, 0.0, 1e-5, workspace, &sigma_8, NULL, NULL);
    gsl_integration_cquad_workspace_free(workspace);

    return sigma_8;
}
