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

// to avoid any implicit declarations, should be cleaned up in the future!
double ccl_massfunc(ccl_cosmology *cosmo);
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

/*---- ROUTINE: ccl_massfunc_halomtor -----
INPUT: ccl_cosmology * cosmo, void *params, halo_mass in units of Msun/h
TASK: takes smoothing halo mass and converts to smoothing halo radius
  in units of Mpc.
*/

double ccl_massfunc_halomtor(ccl_cosmology * cosmo, void *params, double halo_mass){
    double rho_m, rho_crit;

    // critical density of matter used as rho_m
    rho_crit = (3.0*100.0*100.0*cosmo->params.h*cosmo->params.h)/(8.0*M_PI*GNEWT);
    rho_crit = rho_crit*1000.0*1000.0*MPC_TO_METER/SOLAR_MASS;
    rho_m = rho_crit*(cosmo->params.Omega_b+cosmo->params.Omega_c);

    return pow((3.0*halo_mass) / (4*M_PI*rho_m), (1.0/3.0));
}


/*----- ROUTINE: ccl_massfunc_tinker -----
INPUT: cosmology+parameters, some halo mass bin edges, and a redshift
TASK: outputs dn/dM assuming the mass binning is fairly flat. No
  derivatives calculated!
*/

double ccl_massfunc_tinker(ccl_cosmology *cosmo, void *params, double halo_mass_low, double halo_mass_high, double redshift)
{

    double ftinker_low, ftinker_high, ftinker_avg;
    double halo_radius_low, halo_radius_high;
    double sigma_low, sigma_high, dlninvsigma;
    double rho_m, rho_crit, dmass;
    double massavg;

    rho_crit = (3.0*100.0*100.0*cosmo->params.h*cosmo->params.h)/(8.0*M_PI*GNEWT);
    rho_crit = rho_crit*1000.0*1000.0*MPC_TO_METER/SOLAR_MASS;
    rho_m = rho_crit*(cosmo->params.Omega_b+cosmo->params.Omega_c);

    dmass = halo_mass_high - halo_mass_low;
    massavg = (halo_mass_high + halo_mass_low) / 2.0;

    halo_radius_low = ccl_massfunc_halomtor(cosmo, &params, halo_mass_low);
    halo_radius_high = ccl_massfunc_halomtor(cosmo, &params, halo_mass_high);

    ftinker_low = ccl_massfunc_ftinker(cosmo, &params, halo_radius_low, redshift);
    ftinker_high = ccl_massfunc_ftinker(cosmo, &params, halo_radius_high, redshift);
    ftinker_avg = (ftinker_high + ftinker_low) / 2.0;

    sigma_low = ccl_sigmaR(cosmo, halo_radius_low);
    sigma_high = ccl_sigmaR(cosmo, halo_radius_high);
    dlninvsigma = log(1.0/sigma_high) - log(1.0/sigma_low);

    printf("halor: %lf %lf\n", halo_radius_low, halo_radius_high);
    printf("f: %lf %lf %lf\n", ftinker_low, ftinker_high, ftinker_avg);
    printf("lnf: %lf %lf %lf\n", log(ftinker_low), log(ftinker_high), log(ftinker_avg));
    printf("log10f: %lf %lf %lf\n", log10(ftinker_low), log10(ftinker_high), log10(ftinker_avg));
    printf("lninvsigmas: %lf %lf %lf\n", log(1.0/sigma_low), log(1.0/sigma_high), dlninvsigma);
    printf("log10invs: %lf %lf\n", log10(1.0/sigma_low), log10(1.0/sigma_high));
    printf("dmass: %le\n", dmass);

// ftinker_avg*rho_m*dlninvsigma/dmass/massavg;
    return log(massavg*ftinker_avg*dlninvsigma/dmass);
}


/*----- ROUTINE: ccl_massfunc_ftinker -----
INPUT: cosmology+parameters, a mass smoothing scale, and a redshift
TASK: outputs ftinker for calculation in the halo mass function. Assumes
  Tinker mass function from tinker et al 2008!
*/

double ccl_massfunc_ftinker(ccl_cosmology *cosmo, void *params, double halo_mass, double redshift)
{
    double tinker_A, tinker_a, tinker_b, tinker_c;
    double ftinker, sigmaR;
    double halo_radius, alpha;

// probably should set these up as a data structure that can be called
// on demand.
    tinker_A = 0.186*pow(1+redshift, -0.14);
    tinker_a = 1.47*pow(1+redshift, -0.06);
    alpha = pow(10, -1.0*pow(0.75 / log(200.0 / 75.0), 1.2 ));
    tinker_b = 2.57*pow(1+redshift, -1.0*alpha);
    tinker_c = 1.19;

    halo_radius = ccl_massfunc_halomtor(cosmo, &params, halo_mass);

    sigmaR = ccl_sigmaR(cosmo, halo_radius);

    ftinker = tinker_A*(pow(sigmaR/tinker_b,-tinker_a)+1.0)*exp(-tinker_c/sigmaR/sigmaR);

    return ftinker;
}

// just a test main function until things are working. Not for final dist.
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

    halo_mass_low = 1.0E14; // units of Msun/h
    halo_mass_high = 1.0001E14; // units of Msun/h
    redshift = 0.0;

    test = ccl_massfunc_tinker(cosmo, &params, halo_mass_low, halo_mass_high, redshift);

    printf("dn/dM = %le\n", test);

    return 0;
}
