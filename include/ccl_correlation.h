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
 * @param cosmo Cosmological parameters
 * @param n_theta Number of points where to compute the correlation
 * @param theta Vector of angles in radians
 * @param ct1 one of the tracers
 * @param ct2 another tracer
 * @param i_bessel the bessel function order (0 or 4)
 * @param taper_cl 
 * @param taper_cl_limits
 * @param corr_func the output vector with the correlation function
 * @return int
 */
void ccl_correlation(ccl_cosmology *cosmo,
		     int n_ell,double *ell,double *cls,
		     int n_theta,double *theta,double *wtheta,
		     int corr_type,int do_taper_cl,double *taper_cl_limits,int flag_method,
		     int *status);
