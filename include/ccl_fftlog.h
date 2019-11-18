#ifdef __cplusplus
extern "C" {
#endif

#ifndef FFTLOG_H
#define FFTLOG_H

/****************************************************************

This is the famous FFTLog. 

First imlplemented by the living legend Andrew Hamilton:

http://casa.colorado.edu/~ajsh/FFTLog/

This version is a C version that was adapted from the C++ version found
in Copter JWG Carlson, another big loss for the cosmology community.

https://github.com/jwgcarlson/Copter

I've transformed this from C++ to C99 as the lowest common denominator
and provided bindings for C++ and python.

These are the C++ bindings

*****************************************************************/

/* Compute the function
 *   \xi_\mu(\theta) = \int \frac{d\ell}{2\pi} \ell J_\mu(\ell\theta)\,C_\ell
 * C_\ell will be multiplied by ell^{1-\epsilon}, so \epsilon can be used to minimize ringing.
 */
void ccl_fftlog_ComputeXi2D(double mu,double epsilon,
			    int N, const double l[],const double cl[],
			    double th[], double xi[]);

/* Compute the function
 *   \xi_\ell(r) = \int \frac{dk k^2}{2\pi^2} P_k j_\ell(kr)
 * P(k) will be multiplied by k^{3/2-\epsilon}, so \epsilon can be used to minimize ringing.
 */
void ccl_fftlog_ComputeXi3D(double l, double epsilon,
			    int N, const double k[], const double pk[],
			    double r[], double xi[]);
#endif // FFTLOG_H

#ifdef __cplusplus
}
#endif
