/** @file */

#ifndef __CCL_F1D_H_INCLUDED__
#define __CCL_F1D_H_INCLUDED__

#include <gsl/gsl_spline.h>

CCL_BEGIN_DECLS


typedef enum ccl_f1d_extrap_t {
  ccl_f1d_extrap_0 = 0,  // No extrapolation
  ccl_f1d_extrap_const = 410,  // Constant extrapolation
  ccl_f1d_extrap_linx_liny = 411,  // Linear x, linear y
  ccl_f1d_extrap_linx_logy = 412,  // Linear x, log y
  ccl_f1d_extrap_logx_liny = 413,  // Log x, linear y
  ccl_f1d_extrap_logx_logy = 414,  // Log x, log y
} ccl_f1d_extrap_t;


/*
 * Spline wrapper
 * Used to take care of evaluations outside the supported range
 */
typedef struct {
  gsl_spline *spline;
  double y0,yf; //Constant values to use beyond interpolation limit
  ccl_f1d_extrap_t extrap_lo_type;
  ccl_f1d_extrap_t extrap_hi_type;
  double x_ini, x_end; //Interpolation limits
  double y_ini, y_end;
  double der_lo;
  double der_hi;
} ccl_f1d_t;


ccl_f1d_t *ccl_f1d_t_new(int n,double *x,double *y,double y0,double yf,
			 ccl_f1d_extrap_t extrap_lo_type,
			 ccl_f1d_extrap_t extrap_hi_type, int *status);

double ccl_f1d_t_eval(ccl_f1d_t *spl,double x);

void ccl_f1d_t_free(ccl_f1d_t *spl);

CCL_END_DECLS

#endif
