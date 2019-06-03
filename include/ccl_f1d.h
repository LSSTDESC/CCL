/** @file */

#ifndef __CCL_F1D_H_INCLUDED__
#define __CCL_F1D_H_INCLUDED__

#include <gsl/gsl_spline.h>

CCL_BEGIN_DECLS

/*
 * Spline wrapper
 * Used to take care of evaluations outside the supported range
 */
typedef struct {
  gsl_spline *spline;
  double x0,xf; //Interpolation limits
  double y0,yf; //Constant values to use beyond interpolation limit
} ccl_f1d_t;

ccl_f1d_t *ccl_f1d_t_new(int n,double *x,double *y,double y0,double yf);

double ccl_f1d_t_eval(ccl_f1d_t *spl,double x);

void ccl_f1d_t_free(ccl_f1d_t *spl);

CCL_END_DECLS

#endif
