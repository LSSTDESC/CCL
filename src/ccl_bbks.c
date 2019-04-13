#include <math.h>
#include "ccl.h"


/*------ ROUTINE: tsqr_BBKS -----
INPUT: ccl_parameters and k wavenumber in Mpc^-1
TASK: provide the square of the BBKS transfer function with baryonic correction
NOTE: Bardeen et al. (1986) as implemented in Sugiyama (1995)
*/
static double tsqr_BBKS(ccl_parameters* params, double k) {
  double tfac = params->T_CMB / 2.7;
  double q = tfac * tfac * k / (
    params->Omega_m * params->h * params->h *
    exp(-params->Omega_b * (1.0 + pow(2. * params->h, .5) / params->Omega_m)));
  return (
    pow(log(1. + 2.34*q) / (2.34*q), 2.0) /
    pow(1. + 3.89*q + pow(16.1*q, 2.0) + pow(5.46*q, 3.0) + pow(6.71*q, 4.0), 0.5));
}

/*------ ROUTINE: bbks_power -----
INPUT: ccl_parameters and k wavenumber in 1/Mpc
TASK: compute the unnormalized BBKS power spectrum
*/
double ccl_bbks_power(ccl_parameters* params, double k) {
  return pow(k, params->n_s) * tsqr_BBKS(params, k);
}
