/** @file */
#ifndef __CCL_FFTLOG_H_INCLUDED__
#define __CCL_FFTLOG_H_INCLUDED__

CCL_BEGIN_DECLS


/**
 * Compute the function
 *   \xi_\mu(\theta) = \int \frac{d\ell}{2\pi} \ell J_\mu(\ell\theta)\,C_\ell
 * C_\ell will be multiplied by ell^{1-\epsilon}, so \epsilon can be used to minimize ringing.
 * @param mu Bessel function order.
 * @param epsilon FFTLog bias exponent.
 * @param ncl number of power spectra that should be converted into correlation functions.
 * @param N size of l (and the output th).
 * @param l logarithmically spaced values of l.
 * @param cl array of power spectra to be converted. Each of them should be sampled at the values of l.
 * @param th output values of theta (N of them, logarithmically spaced). This array is modified on output.
 * @param xi array of output correlation functions sampled at th.
 */
void ccl_fftlog_ComputeXi2D(double mu,double epsilon,
			    int ncl, int N, double *l,double **cl,
			    double *th, double **xi, int *status);

/**
 * Compute the function
 *   \xi_\ell(r) = \int \frac{dk k^2}{2\pi^2} P_k j_\ell(kr)
 * P(k) will be multiplied by k^{3/2-\epsilon}, so \epsilon can be used to minimize ringing.
 * @param mu Bessel function order.
 * @param epsilon FFTLog bias exponent.
 * @param npk number of power spectra that should be converted into correlation functions.
 * @param N size of k (and the output r).
 * @param k logarithmically spaced values of k.
 * @param pk array of power spectra to be converted. Each of them should be sampled at the values of k.
 * @param r output values of r (N of them, logarithmically spaced). This array is modified on output.
 * @param xi array of output correlation functions sampled at r.
 */
void ccl_fftlog_ComputeXi3D(double l, double epsilon,
			    int npk, int N, double *k, double **pk,
			    double *r, double **xi, int *status);

CCL_END_DECLS
#endif
