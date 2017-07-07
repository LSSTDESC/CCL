#include "ccl_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ccl_params.h"
#include "ccl_error.h"
#include <gsl/gsl_errno.h>

/* ------- ROUTINE: ccl_linear spacing ------
INPUTS: [xmin,xmax] of the interval to be divided in N bins
OUTPUT: bin edges in range [xmin,xmax]
*/

double * ccl_linear_spacing(double xmin, double xmax, int N)
{
  double dx = (xmax-xmin)/(N -1.);
  
  double * x = malloc(sizeof(double)*N);
  if (x==NULL) {
    fprintf(stderr, "ERROR: Could not allocate memory for linear-spaced array (N=%d)\n", N);
    return x;
  }
  
  for (int i=0; i<N; i++) {
    x[i] = xmin + dx*i;
  }
  x[0]=xmin; //Make sure roundoff errors don't spoil edges
  x[N-1]=xmax; //Make sure roundoff errors don't spoil edges
  
  return x;
}

/* ------- ROUTINE: ccl_log spacing ------
INPUTS: [xmin,xmax] of the interval to be divided logarithmically in N bins
TASK: divide an interval in N logarithmic bins
OUTPUT: bin edges in range [xmin,xmax]
*/

double * ccl_log_spacing(double xmin, double xmax, int N)
{
  if (N<2) {
    fprintf(stderr, "ERROR: Cannot make log-spaced array with %d points - need at least 2\n", N);
    return NULL;
  }
  
  if (!(xmin>0 && xmax>0)) {
    fprintf(stderr, "ERROR: Cannot make log-spaced array xmax or xmax non-positive (had %le, %le)\n", xmin, xmax);
    return NULL;
  }
  
  double log_xmax = log(xmax);
  double log_xmin = log(xmin);
  double dlog_x = (log_xmax - log_xmin) /  (N-1.);
  
  double * x = malloc(sizeof(double)*N);
  if (x==NULL) {
    fprintf(stderr, "ERROR: Could not allocate memory for log-spaced array (N=%d)\n", N);
    return x;
  }
  
  for (int i=0; i<N; i++) {
    x[i] = exp(log_xmin + dlog_x*i);
  }
  x[0]=xmin; //Make sure roundoff errors don't spoil edges
  x[N-1]=xmax; //Make sure roundoff errors don't spoil edges
  
  return x;
}


//Spline creator
//n     -> number of points
//x     -> x-axis
//y     -> f(x)-axis
//y0,yf -> values of f(x) to use beyond the interpolation range
SplPar *ccl_spline_init(int n,double *x,double *y,double y0,double yf)
{
  SplPar *spl=malloc(sizeof(SplPar));
  if(spl==NULL)
    return NULL;
  
  spl->intacc=gsl_interp_accel_alloc();
  spl->spline=gsl_spline_alloc(gsl_interp_cspline,n);
  int parstatus=gsl_spline_init(spl->spline,x,y,n);
  if(parstatus) {
    gsl_interp_accel_free(spl->intacc);
    gsl_spline_free(spl->spline);
    return NULL;
  }

  spl->x0=x[0];
  spl->xf=x[n-1];
  spl->y0=y0;
  spl->yf=yf;

  return spl;
}

//Evaluates spline at x checking for bound errors
double ccl_spline_eval(double x,SplPar *spl)
{
  if(x<=spl->x0)
    return spl->y0;
  else if(x>=spl->xf) 
    return spl->yf;
  else {
    double y;
    int stat=gsl_spline_eval_e(spl->spline,x,spl->intacc,&y);
    if (stat!=GSL_SUCCESS) {
      ccl_raise_exception(stat,"ccl_utils.c: ccl_splin_eval(): gsl error\n");
      return NAN;
    }
    return y;
  }
}

//Spline destructor
void ccl_spline_free(SplPar *spl)
{
  gsl_spline_free(spl->spline);
  gsl_interp_accel_free(spl->intacc);
  free(spl);
}
