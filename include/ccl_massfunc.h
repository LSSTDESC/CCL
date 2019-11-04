/** @file */
#ifndef __CCL_MASSFUNC_H_INCLUDED__
#define __CCL_MASSFUNC_H_INCLUDED__

CCL_BEGIN_DECLS

/**
 * Computes sigma(R), the power spectrum normalization, over log-spaced values of mass and radii
 * The result is attached to the cosmology object
 * @param cosmo Cosmological parameters
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.
 */
void ccl_cosmology_compute_sigma(ccl_cosmology *cosmo, int *status);

/**
 * Computes the mass function parameter splines
 * The result is attached to the cosmology object
 * @param cosmo Cosmological parameters
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.
 */
void ccl_cosmology_compute_hmfparams(ccl_cosmology *cosmo, int *status);

/**
 * Compute halo mass function at a given mass for a given cosmology as dn/ dlog10(M)
 * @param cosmo Cosmological parameters
 * @param halomass Mass to compute at, in units of Msun
 * @param a Scale factor, normalized to a=1 today
 * @param odelta choice of Delta
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.
 * @return massfunc, the value of the mass function at the specified parameters
 */
double ccl_massfunc(ccl_cosmology *cosmo, double halomass, double a, double odelta, int *status);

/**
 * Compute the linear halo bias for a cosmology and mass scale
 * @param cosmo Cosmological parameters
 * @param halomass Mass to compute at, in units of Msun
 * @param a Scale factor, normalized to a=1 today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.
 * @return hb, the halo bias at the specified parameters
 */
double ccl_halo_bias(ccl_cosmology *cosmo, double halomass, double a, double odelta, int *status);

/**
 * Convert smoothing halo mass in units of Msun to smoothing halo radius in units of Mpc.
 * @param cosmo Cosmological parameters
 * @param halo_mass Mass to compute at, in units of Msun
 * @param a Scale factor, normalized to a=1 today
 * @param odelta choice of Delta
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.
 * @return smooth_radius, the equivalent tophat smoothing radius corresponding to halo_mass
 */
double ccl_massfunc_m2r(ccl_cosmology *cosmo, double halo_mass, int *status);

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
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.
 * @return sigmaM, the standard deviation of density at mass scale M
 */
double ccl_dlnsigM_dlogM(ccl_cosmology *cosmo, double log_halomass, int *status);

/**
 * Fitting function for the spherical-model critical linear density for collapse
 * Fitting formula from Nakamura & Suto (1997; arXiv:astro-ph/9710107)
 * @param cosmo Cosmological parameters
 * @param a, scale factor, normalized to a=1 today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
double dc_NakamuraSuto(ccl_cosmology *cosmo, double a, int *status);

/**
 * Fitting function for virial collapse density contrast assuming LCDM.
 * Density contrast is relative to background *matter* density, *not* critical density
 * Fitting formula from Bryan & Norman (1998; arXiv:astro-ph/9710107)
 * @param cosmo Cosmological parameters
 * @param a, scale factor, normalized to a=1 today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
double Dv_BryanNorman(ccl_cosmology *cosmo, double a, int *status);

/**
 * Calcualtes the comoving halo radius assuming a given overdensity criteria
 * @param cosmo Cosmological parameters
 * @param halomass in units of Msun
 * @param a, scale factor, normalized to a=1 today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
double r_delta(ccl_cosmology *cosmo, double halomass, double a, double odelta, int *status);

CCL_END_DECLS

#endif
