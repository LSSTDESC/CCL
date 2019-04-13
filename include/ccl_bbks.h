/** @file */
#ifndef __CCL_BBKS_H_INCLUDED__
#define __CCL_BBKS_H_INCLUDED__

CCL_BEGIN_DECLS

/*
 * Calcualte the unnormalized BBKS power spectrum
 * @param params Cosmological parameters
 * @param k, wavenumber in units of Mpc^-1
 */
double ccl_bbks_power(ccl_parameters* params, double k);


CCL_END_DECLS

#endif
