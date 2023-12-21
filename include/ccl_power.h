/** @file */

#ifndef __CCL_POWER_H_INCLUDED__
#define __CCL_POWER_H_INCLUDED__

CCL_BEGIN_DECLS

ccl_f2d_t *ccl_compute_linpower_bbks(ccl_cosmology *cosmo, int *status);

ccl_f2d_t *ccl_compute_linpower_eh(ccl_cosmology *cosmo, int wiggled, int *status);

ccl_f2d_t *ccl_apply_halofit(ccl_cosmology* cosmo, ccl_f2d_t *plin, int *status);

void ccl_rescale_linpower(ccl_cosmology* cosmo, ccl_f2d_t *psp,
                          int rescale_mg, int rescale_norm,
                          int *status);

/**
 * Variance of the projected matter density field with 2D (top-hat) smoothing scale R [Mpc].
 * Returns sigma2(R) for specified cosmology at a = 1.
 * @param cosmo Cosmology parameters and configurations
 * @param R Smoothing scale, in [Mpc] units
 * @param a scale factor
 * @param psp input power spectrum.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return sigma(R).
 */
double ccl_sigma2B(ccl_cosmology *cosmo,double R,double a,
                   ccl_f2d_t *psp, int *status);

/**
 * As `ccl_sigma2B`, calculated for an array of scale factors and smoothing scales.
 * @param cosmo Cosmology parameters and configurations
 * @param na number of scale factor values
 * @param a scale factor values
 * @param R Smoothing scale values, in [Mpc] units
 * @param sigma2B_out output values of the variance calculated for the input a and R.
 * @param psp input power spectrum.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return sigma(R,a).
 */
void ccl_sigma2Bs(ccl_cosmology *cosmo,int na, double *a, double *R,
                  double *sigma2B_out, ccl_f2d_t *psp, int *status);

/**
 * Variance of the matter density field with (top-hat) smoothing scale R [Mpc].
 * Returns sigma(R) for specified cosmology at a = 1.
 * @param cosmo Cosmology parameters and configurations
 * @param R Smoothing scale, in [Mpc] units
 * @param a scale factor
 * @param psp input power spectrum.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 */
double ccl_sigmaR(ccl_cosmology *cosmo, double R, double a,
                  ccl_f2d_t *psp, int * status);

/**
 * Variance of the displacement field with (top-hat) smoothing scale R [Mpc]
 * Returns sigma(V(R)) for specified cosmology at a = 1.
 * @param cosmo Cosmology parameters and configurations
 * @param R smoothing scale, in [Mpc] units
 * @param a scale factor
 * @param psp input power spectrum.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return sigma(R).
 */
double ccl_sigmaV(ccl_cosmology *cosmo, double R, double a,
                  ccl_f2d_t *psp, int * status);

/**
 * Computes sigma8, variance of the matter density field with (top-hat) smoothing scale R = 8 Mpc/h, from linear power spectrum.
 * Returns sigma8 for specified cosmology.
 * @param cosmo Cosmology parameters and configurations
 * @param psp input power spectrum.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return sigma8.
 */
double ccl_sigma8(ccl_cosmology *cosmo, ccl_f2d_t *psp,
                  int * status);

/**
 * Scale for the non-linear cut.
 * Returns k_NL for specified cosmology at specified scale factor.
 * @param cosmo Cosmology parameters and configurations
 * @param a scale factor
 * @param psp input power spectrum.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return kNL.
 */
double ccl_kNL(ccl_cosmology *cosmo, double a,
               ccl_f2d_t *psp, int * status);

CCL_END_DECLS

#endif
