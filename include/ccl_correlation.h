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

#ifdef _NODEF
/**
 * Auxiliar function that computes 1/l. This is a ccl_angular_cl-like function for test case. Hankel tranform of 1./l is 1./theta (up to factors of 2\pi)
 * @param cosmo Cosmological parameters
 * @param l angular multipole
 * @param ct1 one of the tracers
 * @param ct2 another tracer
 * @param status status for catching errors
 * @return 1/l
 */
double angular_l_inv2(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,
		      CCL_ClTracer *clt2, int * status);
  
/**
* Computes the correlation function using FFTlog
 * @param cosmo Cosmological parameters
 * @param n_theta Number of points where to compute the correlation
 * @param theta Vector of angles in radians
 * @param ct1 one of the tracers
 * @param ct2 another tracer
 * @param i_bessel the bessel function order (0 or 4)
 * @param taper_cl 
 * @param taper_cl_limits
 * @param corr_func the output vector with the correlation function
 * @param an input structure that contains the Cls for these tracers
 * @return int 
 */
int ccl_tracer_corr_fftlog(ccl_cosmology *cosmo, int n_theta, double **theta,
		     CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel,
		     bool taper_cl,double *taper_cl_limits,double **corr_func,
		     double (*angular_cl)(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,
					  CCL_ClTracer *clt2, int * status) );
/**
 * Computes the correlation function using Legendre polynomials
 * @param cosmo Cosmological parameters
 * @param n_theta Number of points where to compute the correlation
 * @param theta Vector of angles in radians
 * @param ct1 one of the tracers
 * @param ct2 another tracer
 * @param i_bessel the bessel function order (0 or 4)
 * @param taper_cl 
 * @param taper_cl_limits
 * @param corr_func the output vector with the correlation function
 * @param an input structure that contains the Cls for these tracers
 * @return void
 */
int ccl_tracer_corr_legendre(ccl_cosmology *cosmo, int n_theta, double **theta,
                     CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel,
                     bool taper_cl,double *taper_cl_limits,double **corr_func,
                     double (*angular_cl)(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,
                                          CCL_ClTracer *clt2, int * status) );

/**
 * A simplified version of the correlation function which is called by the python wrapper. 
 * @param theta_in the value of theta in radians where to compute the correlation
 * @param cosmo Cosmological parameters
 * @param ct1 one of the tracers
 * @param ct2 another tracer
 * @param i_bessel the bessel function order (0 or 4)
 * @return the value of the correlation function at this theta
 */
double ccl_single_tracer_corr(double theta_in,ccl_cosmology *cosmo,
			      CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel);
#endif //_NODEF
