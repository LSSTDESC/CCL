/** @file */
#ifndef __CCL_UTILS_H_INCLUDED__
#define __CCL_UTILS_H_INCLUDED__

#include <gsl/gsl_spline.h>

#define CCL_MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define CCL_MAX(a, b)  (((a) > (b)) ? (a) : (b))

CCL_BEGIN_DECLS

typedef enum ccl_integration_t {
  ccl_integration_qag_quad = 500,  // GSL's quad
  ccl_integration_spline = 501,  // Spline integral
} ccl_integration_t;

/**
 * Compute bin edges of N-1 linearly spaced bins on the interval [xmin,xmax]
 * @param xmin minimum value of spacing
 * @param xmax maximum value of spacing
 * @param N number of bins plus one (number of bin edges)
 * @return x, bin edges in range [xmin, xmax]
 */
double * ccl_linear_spacing(double xmin, double xmax, int N);

/**
 * Compute bin edges of N-1 logarithmically and then linearly spaced bins on the interval [xmin,xmax]
 * @param xminlog minimum value of spacing
 * @param xmin value when logarithmical ends and linear spacing begins
 * @param xmax maximum value of spacing
 * @param Nlin number of linear bins plus one (number of bin edges)
 * @param Nlog number of logarithmic bins plus one (number of bin edges)
 * @return x, bin edges in range [xminlog, xmax]
 */
double * ccl_linlog_spacing(double xminlog, double xmin, double xmax, int Nlin, int Nlog);

/**
 * Compute bin edges of N-1 logarithmically spaced bins on the interval [xmin,xmax]
 * @param xmin minimum value of spacing
 * @param xmax maximum value of spacing
 * @param N number of bins plus one (number of bin edges)
 * @return x, bin edges in range [xmin, xmax]
 */
double * ccl_log_spacing(double xmin, double xmax, int N);
//Returns array of N logarithmically-spaced values between xmin and xmax

double ccl_j_bessel(int l,double x);
//Spherical Bessel function of order l (adapted from CAMB)

/**
 * Compute spline integral.
 * @param nx number of elements in input array.
 * @param ny number of y arrays.
 * @param x input x-values.
 * @param y input y-values (ny arrays with nx elements).
 * @param a lower end of integration range.
 * @param b upper end of integration range (use b<a if you just want to integrate over all of y).
 * @param result array of output spline integral values.
 * @param T spline type.
 * @param status status flag.
 */
void ccl_integ_spline(int ny, int nx,double *x,double **y,
                      double a, double b, double *result,
                      const gsl_interp_type *T, int *status);

int ccl_openmp_version();

int ccl_openmp_threads();

CCL_END_DECLS

#endif
