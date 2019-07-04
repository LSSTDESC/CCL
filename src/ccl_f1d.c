#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_errno.h>

#include "ccl.h"



//Spline creator
//n     -> number of points
//x     -> x-axis
//y     -> f(x)-axis
//y0,yf -> values of f(x) to use beyond the interpolation range
ccl_f1d_t *ccl_f1d_t_new(int n,double *x,double *y,double y0,double yf)
{
  ccl_f1d_t *spl=malloc(sizeof(ccl_f1d_t));
  if(spl==NULL)
    return NULL;

  spl->spline=gsl_spline_alloc(gsl_interp_cspline,n);
  int parstatus=gsl_spline_init(spl->spline,x,y,n);
  if(parstatus) {
    gsl_spline_free(spl->spline);
    free(spl);
    return NULL;
  }

  spl->x0=x[0];
  spl->xf=x[n-1];
  spl->y0=y0;
  spl->yf=yf;

  return spl;
}

//Evaluates spline at x checking for bound errors
double ccl_f1d_t_eval(ccl_f1d_t *spl,double x)
{
  if(x<=spl->x0)
    return spl->y0;
  else if(x>=spl->xf)
    return spl->yf;
  else {
    double y;
    int stat=gsl_spline_eval_e(spl->spline,x,NULL,&y);
    if (stat!=GSL_SUCCESS) {
      ccl_raise_gsl_warning(stat, "ccl_utils.c: ccl_splin_eval():");
      return NAN;
    }
    return y;
  }
}

//Spline destructor
void ccl_f1d_t_free(ccl_f1d_t *spl)
{
  if (spl != NULL) {
    gsl_spline_free(spl->spline);
  }
  free(spl);
}
