/** @file */

#ifndef __CCL_POWER_H_INCLUDED__
#define __CCL_POWER_H_INCLUDED__

CCL_BEGIN_DECLS

/**
 * Linear matter power spectrum.
 * Returns P_lin(k,a) [Mpc^3] for given cosmology, using the method specified in cosmo->config.transfer_function_method.
 * @param cosmo Cosmology parameters and configurations
 * @param k Fourier mode, in [1/Mpc] units
 * @param a scale factor, normalized to 1 for today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return P_lin(k,a).
 */
double ccl_linear_matter_power(ccl_cosmology * cosmo, double k, double a,int * status);

/**
 * Non-linear matter power spectrum.
 * Returns P_NL(k,a) [Mpc^3] for given cosmology, using the method specified in cosmo->config.transfer_function_method and cosmo->config.matter_power_spectrum_method.
 * @param cosmo Cosmology parameters and configurations
 * @param k Fourier mode, in [1/Mpc] units
 * @param a scale factor, normalized to 1 for today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return P_NL(k,a).
 */

double ccl_nonlin_matter_power(ccl_cosmology * cosmo, double k, double a,int * status);

/**
 * Compute the linear power spectrum and create a 2d spline P(k,z) to be stored
 * in the cosmology structure.
 * @param cosmo Cosmological parameters
 * param psp The linear power spectrum to use for Boltzmann solvers, if any.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_cosmology_compute_linear_power(ccl_cosmology * cosmo, ccl_f2d_t *psp, int* status);

/**
 * Compute the non-linear power spectrum and create a 2d spline P(k,z) to be stored
 * in the cosmology structure.
 * @param cosmo Cosmological parameters
 * @param psp existing 2d spline.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_cosmology_compute_nonlin_power(ccl_cosmology * cosmo,
                                        ccl_f2d_t *psp,
                                        int* status);

/**
 * Compute the non-linear power spectrum from an existing 2d spline P(k,z).
 * @param cosmo Cosmological parameters
 * @param psp existing 2d spline.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_cosmology_compute_nonlin_power_from_f2d(ccl_cosmology *cosmo,
                                                 ccl_f2d_t *psp, int *status);

/**
 * Variance of the matter density field with (top-hat) smoothing scale R [Mpc].
 * Returns sigma(R) for specified cosmology at a = 1.
 * @param cosmo Cosmology parameters and configurations
 * @param R Smoothing scale, in [Mpc] units
 * @param a scale factor
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return sigma(R).
 */
double ccl_sigmaR(ccl_cosmology *cosmo, double R, double a, int * status);

/**
 * Variance of the displacement field with (top-hat) smoothing scale R [Mpc]
 * Returns sigma(V(R)) for specified cosmology at a = 1.
 * @param cosmo Cosmology parameters and configurations
 * @param R smoothing scale, in [Mpc] units
 * @param a scale factor
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return sigma(R).
 */
double ccl_sigmaV(ccl_cosmology *cosmo, double R, double a, int * status);

/**
 * Computes sigma8, variance of the matter density field with (top-hat) smoothing scale R = 8 Mpc/h, from linear power spectrum.
 * Returns sigma8 for specified cosmology.
 * @param cosmo Cosmology parameters and configurations
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return sigma8.
 */
double ccl_sigma8(ccl_cosmology *cosmo, int * status);

CCL_END_DECLS

#endif
