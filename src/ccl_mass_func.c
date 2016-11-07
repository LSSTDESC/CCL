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



/*----- ROUTINE: ccl_mass_func -----
INPUT: ccl_cosmology * cosmo, ccl_config to decide on which mass func
TASK: return dn/dM according to some methodology
*/

/*
void ccl_mass_func(ccl_cosmology *cosmo)
{
// code here determines which methodology has been asked for and
// then goes about calculating it, calling a further function.
}
*/

/*----- ROUTINE: ccl_mass_func_tinker -----
INPUT: whatever it takes to calculate Tinker (2008) hmf
TASK: output Tinker (2008) hmf
*/

/*
void ccl_mass_func_tinker(ccl_cosmology *cosmo)
{
// Tinker (2008) HMF of the form dn/dM = f(sigma)*rho_m*(d ln sigma^-1/dM)
// will need to calculate the f(sigma) and the d ln sigma^-1/dM. The rest
// pretty straightforward. So something here will logicall call for f(sigma).

double ftinker;

ftinker = ccl_mass_func_ftinker()
dndM = ftinker*rho_m*dlninvsigmadM
}
*/

/*----- ROUTINE: ccl_mass_func_ftinker -----
INPUT: cosmology so that it can calculate sigma(R) and possibly
convert this into sigma(M). Probably needs a specific M.
TASK: output f(sigma) as a single number.
*/


double ccl_mass_func_ftinker(ccl_cosmology *cosmo, void *params, double halo_M)
{
// here we will need to call for sigma(R) and slap it together
// with the fit parameters A, a, b, and c from simulation. 
    double tinker_A, tinker_a, tinker_b, tinker_c;
    double ftinker, sigmaR;
    double rho_m, halo_R;

    tinker_A = 0.186;
    tinker_a = 1.47;
    tinker_b = 2.57;
    tinker_c = 1.19;

// probably can find rho_m as an existing cosmological parameter. If not,
// calculate it!
// can't find Newton constant, so we're slamming it in quickly for now.
    rho_m = (3.0*cosmo->params.h*cosmo->params.h)/(8.0*M_PI*GNEWT);

    printf("Parameters calculated.\n");

    halo_R = pow((3.0*halo_M) / (4*M_PI*rho_m), (1.0/3.0));

    printf("Test halo radius is: %lf\n",halo_R);

    sigmaR = ccl_sigma8(cosmo);

    printf("SigmaR calculated. LogInvSig: %lf\n", log10(1.0/sigmaR));

    ftinker = tinker_A*( pow( (sigmaR / tinker_b), -1.0*tinker_a)+1.0)*(exp(-1.0*tinker_c / (sigmaR*sigmaR) ) );

    return ftinker;
}


// just a test main function until things are working.
int main(){
    double test, halo_mass;
    double Omega_c = 0.25;
    double Omega_b = 0.05;
    double h = 0.7;
    double A_s = 2.1E-9;
    double n_s = 0.96;

    ccl_configuration config = default_config;
    config.transfer_function_method = ccl_boltzmann;
    // note - transfer function to boltzmann is currently required
    // to avoid segfaults. We hate segfaults. Boo segfaults.

    ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, A_s, n_s);
    ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

    printf("Cosmology Generated.\n");

// probably unit errors. We can clear that up later, once the code
// runs through to completion
    halo_mass = 1.0E13;

    test = ccl_mass_func_ftinker(cosmo, &params, halo_mass);

    printf("ftinker generated: %lf", test);

    return 0;
}
