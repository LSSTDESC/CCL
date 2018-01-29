/** @file */
#ifdef __cplusplus
extern "C" {
#endif

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
 *  - CCL_CORR_GG : spin0-spin0
 *  - CCL_CORR_GL : spin0-spin2
 *  - CCL_CORR_LP : spin2-spin2 (xi+)
 *  - CCL_CORR_LM : spin2-spin2 (xi-)
 * Currently supported spin-0 fields are number counts and CMB lensing. The only spin-2 is currently shear.
 */
void ccl_correlation(ccl_cosmology *cosmo,
		     int n_ell,double *ell,double *cls,
		     int n_theta,double *theta,double *wtheta,
		     int corr_type,int do_taper_cl,double *taper_cl_limits,int flag_method,
		     int *status);

/**
 * Computes the 3dcorrelation function (wrapper)
 * @param cosmo :Cosmological parameters
 * @param a : scale factor
 * @param n_r : number of output values of distance r
 * @param r : values of the distance in Mpc
 * @param xi : the values of the correlation function at the distances above will be returned in this array, which should be pre-allocated
 * @param do_taper_pk : key for tapering (using cosine tapering by default)
 * @param taper_pk_limits: limits of tapering
 */
void ccl_correlation_3d(ccl_cosmology *cosmo,double a,
		     int n_r,double *r,double *xi,
		     int do_taper_pk,double *taper_pk_limits,
		     int *status);

#ifdef __cplusplus
}
#endif
