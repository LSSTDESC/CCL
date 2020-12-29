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
ccl_f1d_t *ccl_f1d_t_new(int n,double *x,double *y,double y0,double yf,
			 ccl_f1d_extrap_t extrap_lo_type,
			 ccl_f1d_extrap_t extrap_hi_type, int *status)
{
  ccl_f1d_t *spl=malloc(sizeof(ccl_f1d_t));
  if(spl==NULL) {
    *status = CCL_ERROR_MEMORY;
    return NULL;
  }

  spl->spline=gsl_spline_alloc(gsl_interp_akima,n);
  if (spl->spline == NULL) {
    *status = CCL_ERROR_MEMORY;
    free(spl);
    return NULL;
  }
  int parstatus=gsl_spline_init(spl->spline,x,y,n);
  if(parstatus) {
    *status = CCL_ERROR_SPLINE;
    gsl_spline_free(spl->spline);
    free(spl);
    return NULL;
  }

  if(*status==0) {
    spl->y0=y0;
    spl->yf=yf;
    spl->x_ini=x[0];
    spl->x_end=x[n-1];
    spl->y_ini=y[0];
    spl->y_end=y[n-1];
    spl->extrap_lo_type=extrap_lo_type;
    spl->extrap_hi_type=extrap_hi_type;
  }

  // Compute derivatives
  // Low-end
  if(spl->extrap_lo_type == ccl_f1d_extrap_const) {
    spl->der_lo = 0;
  } 
  else if(spl->extrap_lo_type == ccl_f1d_extrap_linx_liny) {
    spl->der_lo = (y[1]-y[0])/(x[1]-x[0]);
  }
  else if(spl->extrap_lo_type == ccl_f1d_extrap_linx_logy) {
    if(y[1]*y[0]<=0)
      *status = CCL_ERROR_SPLINE;
    else
      spl->der_lo = log(y[1]/y[0])/(x[1]-x[0]);
  }
  else if(spl->extrap_lo_type == ccl_f1d_extrap_logx_liny) {
    if(x[1]*x[0]<=0)
      *status = CCL_ERROR_SPLINE;
    else
      spl->der_lo = (y[1]-y[0])/log(x[1]/x[0]);
  }
  else if(spl->extrap_lo_type == ccl_f1d_extrap_logx_logy) {
    if((y[1]*y[0]<=0) || (x[1]*x[0]<=0))
      *status = CCL_ERROR_SPLINE;
    else
      spl->der_lo = log(y[1]/y[0])/log(x[1]/x[0]);
  }
  else // No extrapolation
    spl->der_lo = 0;

  // High-end
  if(spl->extrap_hi_type == ccl_f1d_extrap_const) {
    spl->der_hi = 0;
  } 
  else if(spl->extrap_hi_type == ccl_f1d_extrap_linx_liny) {
    spl->der_hi = (y[n-1]-y[n-2])/(x[n-1]-x[n-2]);
  }
  else if(spl->extrap_hi_type == ccl_f1d_extrap_linx_logy) {
    if(y[n-1]*y[n-2]<=0)
      *status = CCL_ERROR_SPLINE;
    else
      spl->der_hi = log(y[n-1]/y[n-2])/(x[n-1]-x[n-2]);
  }
  else if(spl->extrap_hi_type == ccl_f1d_extrap_logx_liny) {
    if(x[n-1]*x[n-2]<=0)
      *status = CCL_ERROR_SPLINE;
    else
      spl->der_hi = (y[n-1]-y[n-2])/log(x[n-1]/x[n-2]);
  }
  else if(spl->extrap_hi_type == ccl_f1d_extrap_logx_logy) {
    if((y[n-1]*y[n-2]<=0) || (x[n-1]*x[n-2]<=0))
      *status = CCL_ERROR_SPLINE;
    else
      spl->der_hi = log(y[n-1]/y[n-2])/log(x[n-1]/x[n-2]);
  }
  else // No extrapolation
    spl->der_hi = 0;

  // If one of the derivatives could not be calculated
  // then return NULL.
  if (*status){
    ccl_f1d_t_free(spl);
    spl = NULL;
  }

  return spl;
}

//Evaluates spline at x checking for bound errors
double ccl_f1d_t_eval(ccl_f1d_t *spl,double x)
{
  // Extrapolation
  // Lin_x, lin_y
  // f(x) = f_n + (x-x_n) * der
  //   der = (f_n - f_{n-1}) / (x_n - x_{n-1})
  //   

  // Log_x, lin_y
  // f(x) = f_n + log(x/x_n) * der
  // der = (f_n - f_{n-1}) / log(x_n/x_{n-1})

  // Lin_x, log_y
  // f(x) = f_n * exp[ (x - x_n) * der ]
  // der = log(f_n/f_{n-1})/(x_n-x_{n-1})

  // Log_x, log_y
  // f(x) = f_n * exp[ log(x/x_n) * der ]
  // der = log(f_n/f_{n-1})/log(x_n/x_{n-1})
  if(x<=spl->x_ini) {
    if(spl->extrap_lo_type==ccl_f1d_extrap_const)
      return spl->y0;
    else if(spl->extrap_lo_type==ccl_f1d_extrap_linx_liny) {
      return spl->y_ini + spl->der_lo * (x - spl->x_ini);
    }
    else if(spl->extrap_lo_type==ccl_f1d_extrap_logx_liny) {
      return spl->y_ini + spl->der_lo * log(x/spl->x_ini);
    }
    else if(spl->extrap_lo_type==ccl_f1d_extrap_linx_logy) {
      return spl->y_ini * exp(spl->der_lo * (x - spl->x_ini));
    }
    else if(spl->extrap_lo_type==ccl_f1d_extrap_logx_logy) {
      return spl->y_ini * pow(x/spl->x_ini, spl->der_lo);
    }
    else {
      ccl_raise_gsl_warning(CCL_ERROR_SPLINE_EV,
                            "ccl_f1d.c: ccl_f1d_t_eval(): "
                            "x-value below range.");
      return NAN;
    }
  }
  else if(x>=spl->x_end) {
    if(spl->extrap_hi_type==ccl_f1d_extrap_const)
      return spl->yf;
    else if(spl->extrap_hi_type==ccl_f1d_extrap_linx_liny) {
      return spl->y_end + spl->der_hi * (x - spl->x_end);
    }
    else if(spl->extrap_hi_type==ccl_f1d_extrap_logx_liny) {
      return spl->y_end + spl->der_hi * log(x/spl->x_end);
    }
    else if(spl->extrap_hi_type==ccl_f1d_extrap_linx_logy) {
      return spl->y_end * exp(spl->der_hi * (x - spl->x_end));
    }
    else if(spl->extrap_hi_type==ccl_f1d_extrap_logx_logy) {
      return spl->y_end * pow(x/spl->x_end, spl->der_hi);
    }
    else {
      ccl_raise_gsl_warning(CCL_ERROR_SPLINE_EV,
                            "ccl_f1d.c: ccl_f1d_t_eval(): "
                            "x-value above range.");
      return NAN;
    }
  }
  else {
    double y;
    int stat=gsl_spline_eval_e(spl->spline,x,NULL,&y);
    if (stat!=GSL_SUCCESS) {
      ccl_raise_gsl_warning(stat, "ccl_f1d.c: ccl_f1d_t_eval(): "
                            "x-value outside range.");
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
