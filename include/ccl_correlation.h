/** @file */

#pragma once

#define CCL_CORR_LGNDRE 1001
#define CCL_CORR_FFTLOG 1002
#define CCL_CORR_BESSEL 1003
#define CCL_CORR_GG 2001
#define CCL_CORR_GL 2002
#define CCL_CORR_LP 2003
#define CCL_CORR_LM 2004

/**
 * Computes the correlation function (wrapper)
 * @param cosmo :Cosmological parameters
 * @param n_ell : number of multipoles in the input power spectrum
 * @param ell : multipoles at which the power spectrum is evaluated
 * @param cls : input power spectrum
 * @param n_theta : number of output values of the separation angle (theta)
 * @param theta : values of the separation angle in degrees.
 * @param wtheta : the values of the correlation function at the angles above will be returned in this array, which should be pre-allocated
 * @param do_taper_cl :
 * @param taper_cl_limits
 * @param flag_method : method to compute the correlation function. Choose between:
 *  - CCL_CORR_FFTLOG : fast integration with FFTLog
 *  - CCL_CORR_BESSEL : direct integration over the Bessel function
 *  - CCL_CORR_LGNDRE : brute-force sum over legendre polynomials
 * @param corr_type : type of correlation function. Choose between:
 *  - CCL_CORR_GG : galaxy-galaxy
 *  - CCL_CORR_GL : galaxy-shear
 *  - CCL_CORR_LP : shear-shear (xi+)
 *  - CCL_CORR_LM : shear-shear (xi-)
 */
void ccl_correlation(ccl_cosmology *cosmo,
		     int n_ell,double *ell,double *cls,
		     int n_theta,double *theta,double *wtheta,
		     int corr_type,int do_taper_cl,double *taper_cl_limits,int flag_method,
		     int *status);
