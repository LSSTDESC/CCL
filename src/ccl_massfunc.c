#include "ccl.h"
#include "ccl_core.h"
#include "ccl_utils.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_interp.h"
#include "gsl/gsl_spline.h"
#include "ccl_power.h"

double ccl_massfunc_tinker(ccl_cosmology * cosmo, void *params, double halo_mass_low, double halo_mass_high, double redshift);
double ccl_massfunc_ftinker(ccl_cosmology * cosmo, void *params, double halo_mass, double redshift);
double ccl_massfunc_halomtor(ccl_cosmology * cosmo, void *params, double halo_mass);

/*----- ROUTINE: ccl_massfunc -----
INPUT: ccl_cosmology * cosmo, ccl_config to decide on which mass func
TASK: return dn/dM according to some methodology
*/

/*
void ccl_massfunc(ccl_cosmology *cosmo)
{
// code here determines which methodology has been asked for and
// then goes about calculating it, calling a further function.
}
*/

double ccl_massfunc_halomtor(ccl_cosmology * cosmo, void *params, double halo_mass){
    double rho_m;

    rho_m = (3.0*cosmo->params.H0*cosmo->params.H0)/(8.0*M_PI*GNEWT);

    return pow((3.0*halo_mass) / (4*M_PI*rho_m), (1.0/3.0));
}


/*----- ROUTINE: ccl_massfunc_tinker -----
INPUT: whatever it takes to calculate Tinker (2008) hmf
TASK: output Tinker (2008) hmf
*/

double ccl_massfunc_tinker(ccl_cosmology *cosmo, void *params, double halo_mass_low, double halo_mass_high, double redshift)
{
// Tinker (2008) HMF of the form dn/dM = f(sigma)*rho_m*(d ln sigma^-1/dM)
// will need to calculate the f(sigma) and the d ln sigma^-1/dM. The rest
// pretty straightforward. So something here will logicall call for f(sigma).

    double ftinker_low, ftinker_high, ftinker_avg;
    double halo_radius_low, halo_radius_high;
    double sigma_low, sigma_high, dlninvsigma;
    double rho_m, dmass;

    rho_m = (3.0*cosmo->params.H0*cosmo->params.H0)/(8.0*M_PI*GNEWT);
    dmass = halo_mass_high - halo_mass_low;

    halo_radius_low = ccl_massfunc_halomtor(cosmo, &params, halo_mass_low);
    halo_radius_high = ccl_massfunc_halomtor(cosmo, &params, halo_mass_high);

    ftinker_low = ccl_massfunc_ftinker(cosmo, &params, halo_radius_low, redshift);
    ftinker_high = ccl_massfunc_ftinker(cosmo, &params, halo_radius_high, redshift);
    ftinker_avg = (ftinker_high + ftinker_low) / 2.0;

    sigma_low = ccl_sigmaR(cosmo, halo_radius_low);
    sigma_high = ccl_sigmaR(cosmo, halo_radius_high);
    dlninvsigma = log10(1.0/sigma_high) - log10(1.0/sigma_low);

    printf("ftinkers: %lf %lf %lf\n", ftinker_low, ftinker_high, ftinker_avg);
    printf("sigmas: %lf %lf\n", sigma_low, sigma_high);
    printf("dmass: %le\n", dmass);

    return ftinker_avg*rho_m*dlninvsigma/dmass;
}


/*----- ROUTINE: ccl_massfunc_ftinker -----
INPUT: cosmology so that it can calculate sigma(R) and possibly
convert this into sigma(M). Probably needs a specific M.
TASK: output f(sigma) as a single number.
*/


double ccl_massfunc_ftinker(ccl_cosmology *cosmo, void *params, double halo_mass, double redshift)
{
// here we will need to call for sigma(R) and slap it together
// with the fit parameters A, a, b, and c from simulation. 
    double tinker_A, tinker_a, tinker_b, tinker_c;
    double ftinker, sigmaR;
    double halo_radius, alpha;

// probably should set these up as a data structure that can be called
// on demand.
    tinker_A = 0.186*pow(1+redshift, -0.14);
    tinker_a = 1.47*pow(1+redshift, -0.06);
    alpha = pow(10, -1.0*pow(0.75 / log10(200 / 75), 1.2 ));
    tinker_b = 2.57*pow(1+redshift, -1.0*alpha);
    tinker_c = 1.19;

// probably can find rho_m as an existing cosmological parameter. If not,
// calculate it!

    halo_radius = ccl_massfunc_halomtor(cosmo, &params, halo_mass);

    sigmaR = ccl_sigmaR(cosmo, halo_radius);

    //printf("SigmaR calculated. LogInvSig: %lf\n", log10(1.0/sigmaR));
    //printf("LogSig: %lf\n", log10(sigmaR));

    ftinker = tinker_A*( pow( (sigmaR / tinker_b), -1.0*tinker_a)+1.0)*(exp(-1.0*tinker_c / (sigmaR*sigmaR) ) );

    //printf("log10 ftinker = %lf\n", log10(ftinker));

    return ftinker;
}


// just a test main function until things are working.
int main(){
    // set base cosmology for testing purposes
    double test, halo_mass_low, halo_mass_high, redshift;
    double Omega_c = 0.25;
    double Omega_b = 0.05;
    double h = 0.7;
    double A_s = 2.1E-9;
    double n_s = 0.96;

    ccl_configuration config = default_config;
    config.transfer_function_method = ccl_bbks;

    ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, A_s, n_s);
    params.sigma_8 = 0.8; // default for testing purposes since NaN

    ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

    printf("Cosmology Generated.\n");

    halo_mass_low = 5.0E12;
    halo_mass_high = 1.0E13;
    redshift = 0.0;

    test = ccl_massfunc_tinker(cosmo, &params, halo_mass_low, halo_mass_high, redshift);

    printf("dn/dM = %lf\n", test);

    return 0;
}
