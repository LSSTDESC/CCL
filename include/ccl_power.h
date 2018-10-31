/* @file */

#ifndef __CCL_POWER_H_INCLUDED__
#define __CCL_POWER_H_INCLUDED__

CCL_BEGIN_DECLS

/**
 * CLASS power spectrum without splines.
 * Write k, P(k,z) [1/Mpc, Mpc^3] for given cosmology at the k values used within CLASS (spectra.ln_k[]), using the method specified in config.matter_power_spectrum_method.
 * @param filename File into which k, P(k,a) will be written
 * @param cosmo Cosmology parameters and configurations
 * @param z Redshift at which the power spectrum is evaluated
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
void ccl_cosmology_write_power_class_z(char *filename, ccl_cosmology * cosmo, double z, int * status);

/**
 * Correction for the impact of baryonic physics on the matter power spectrum.
 * Returns f(k,a) [dimensionless] for given cosmology, using the method specified for the baryonic transfer function.
 * f(k,a) is the fractional change in the nonlinear matter power spectrum from the Baryon Correction Model (BCM) of Schenider & Teyssier (2015). The parameters of the model are passed as part of the cosmology class.
 * @param cosmo Cosmology parameters and configurations, including baryonic parameters.
 * @param k Fourier mode, in [1/Mpc] units
 * @param a scale factor, normalized to 1 for today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return f(k,a).
 */
double ccl_bcm_model_fka(ccl_cosmology * cosmo, double k, double a, int *status);

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
 * Compute the power spectrum and create a 2d spline P(k,z) to be stored
 * in the cosmology structure.
 * @param cosmo Cosmological parameters 
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_cosmology_compute_power(ccl_cosmology * cosmo, int* status);

/**
 * Compute the power spectrum and create a 2d spline P(k,z) to be stored
 * in the cosmology structure.
 * @param cosmo Cosmological parameters
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_cosmology_compute_power(ccl_cosmology * cosmo, int* status);

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
