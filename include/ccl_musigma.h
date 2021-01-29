/** @file */
#ifndef __CCL_CLASS_H_INCLUDED__
#define __CCL_CLASS_H_INCLUDED__

CCL_BEGIN_DECLS

/*
 * Spline the linear power spectrum for mu-Sigma MG cosmologies.
 * @param cosmo Cosmological parameters
 ^ @param psp The linear power spectrum to spline.
 * @param status, integer indicating the status
 */
void ccl_rescale_musigma_s8(ccl_cosmology* cosmo, ccl_f2d_t *psp,
                            int mg_rescale, int* status);

/**
 * Sigma(a,k) in the mu / Sigma parameterisation of modified gravity at a given redshift and scale.
 * @param cosmo Cosmological parameters
 * @param a scale factor, normalized to 1 for today
 * @param k wavenumber
 * @param status 0 if there are no errors, nonzero otherwise.
 *  For specific cases see documentation for ccl_error.c
 * @return Sigma(a,k), function of the mu / Sigma parameterisation of modified gravity.
*/
double ccl_Sig_MG(ccl_cosmology * cosmo, double a, double k, int *status);

/**
 * mu(a,k) in the mu / Sigma parameterisation of modified gravity at a given redshift and scale.
 * @param cosmo Cosmological parameters
 * @param a scale factor, normalized to 1 for today
 * @param k wavenumber
 * @param status 0 if there are no errors, nonzero otherwise.
 *  For specific cases see documentation for ccl_error.c
 * @return mu(a,k), function of the mu / Sigma parameterisation of modified gravity.
*/
double ccl_mu_MG(ccl_cosmology * cosmo, double a, double k, int *status);

CCL_END_DECLS

#endif
