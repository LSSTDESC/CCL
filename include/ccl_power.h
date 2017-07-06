#pragma once

#include "ccl_core.h"

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
 * Variance of the matter density field with (top-hat) smoothing scale R [Mpc].
 * Returns sigma(R) for specified cosmology at a = 1.
 * @param cosmo Cosmology parameters and configurations
 * @param R Smoothing scale, in [Mpc] units
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return sigma(R).
 */
double ccl_sigmaR(ccl_cosmology *cosmo, double R, int * status);

/**
 * Computes sigma_8, variance of the matter density field with (top-hat) smoothing scale R = 8 Mpc/h, from linear power spectrum.
 * Returns sigma_8 for specified cosmology.
 * @param cosmo Cosmology parameters and configurations
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return sigma_8.
 */
double ccl_sigma8(ccl_cosmology *cosmo, int * status);
