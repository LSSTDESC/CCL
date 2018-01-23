/** @file */
#ifdef __cplusplus
extern "C" {
#endif

#pragma once
#include "gsl/gsl_spline.h"

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
