#pragma once

#include "gsl/gsl_spline.h"

double * ccl_linear_spacing(double xmin, double xmax, int N);
//Returns array of  N linearly-spaced values between xmin and xmax

double * ccl_log_spacing(double xmin, double xmax, int N);
//Returns array of N logarithmically-spaced values between xmin and xmax

//Spline wrapper
//Used to take care of evaluations outside the supported range
typedef struct {
  gsl_interp_accel *intacc; //GSL spline
  gsl_spline *spline;
  double x0,xf; //Interpolation limits
  double y0,yf; //Constant values to use beyond interpolation limit
} SplPar;

SplPar *ccl_spline_init(int n,double *x,double *y,double y0,double yf);

double ccl_spline_eval(double x,SplPar *spl);

void ccl_spline_free(SplPar *spl);
