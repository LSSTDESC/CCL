/** @file */
#ifndef __CCL_MASSFUNC_H_INCLUDED__
#define __CCL_MASSFUNC_H_INCLUDED__

CCL_BEGIN_DECLS

/**
 * Computes sigma(R), the power spectrum normalization, over log-spaced values of mass and radii
 * The result is attached to the cosmology object
 * @param cosmo Cosmological parameters
 * @param psp linear matter power spectrum
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.
 */
void ccl_cosmology_compute_sigma(ccl_cosmology *cosmo, ccl_f2d_t *psp, int *status);

/**
 * Calculate the standard deviation of density at smoothing mass M via interpolation.
 * Return sigma from the sigmaM interpolation.
 * @param cosmo Cosmological parameters
 * @param log_halomass log10(Mass) to compute at, in units of Msun
 * @param a Scale factor, normalized to a=1 today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.
 * @return sigmaM, the standard deviation of density at mass scale M
 */
double ccl_sigmaM(ccl_cosmology *cosmo, double log_halomass, double a, int *status);

/**
 * Calculate the logarithmic derivative of the standard deviation of density at smoothing mass M
 * via interpolation.
 * @param cosmo Cosmological parameters
 * @param log_halomass log10(Mass) to compute at, in units of Msun
 * @param a scale factor, normalized to a=1 today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.
 * @return sigmaM, the standard deviation of density at mass scale M
 */
double ccl_dlnsigM_dlogM(ccl_cosmology *cosmo, double log_halomass, double a, int *status);

CCL_END_DECLS

#endif
