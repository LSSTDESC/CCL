/** @file */
#ifdef __cplusplus
extern "C" {
#endif

#pragma once
#include "gsl/gsl_spline.h"

#define CCL_MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define CCL_MAX(a, b)  (((a) > (b)) ? (a) : (b))

/**
 * Compute bin edges of N-1 linearly spaced bins on the interval [xmin,xmax]
 * @param xmin minimum value of spacing
 * @param xmax maximum value of spacing
 * @param N number of bins plus one (number of bin edges)
 * @return x, bin edges in range [xmin, xmax]
 */
double * ccl_linear_spacing(double xmin, double xmax, int N);


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
 * Spline wrapper
 * Used to take care of evaluations outside the supported range
 */
typedef struct {
  gsl_interp_accel *intacc; //GSL spline
  gsl_spline *spline;
  double x0,xf; //Interpolation limits
  double y0,yf; //Constant values to use beyond interpolation limit
} SplPar;

SplPar *ccl_spline_init(int n,double *x,double *y,double y0,double yf);

double ccl_spline_eval(double x,SplPar *spl);

void ccl_spline_free(SplPar *spl);

#ifdef __cplusplus
}
#endif
