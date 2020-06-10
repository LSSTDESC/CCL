#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_expint.h>
#include "ccl.h"

static double einasto_norm_integrand(double x, void *params)
{
  double alpha = *((double *)(params));
  return x*x*exp(-2*(pow(x,alpha)-1)/alpha);
}

void ccl_einasto_norm_integral(int n_m, double *r_s, double *r_delta, double *alpha,
			       double *norm_out,int *status)
{
#pragma omp parallel default(none)			\
  shared(n_m, r_s, r_delta, alpha, norm_out, status)
  {
    int ii;
    int status_this=0;
    gsl_function F;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    
    if (w == NULL)
      status_this = CCL_ERROR_MEMORY;
    
    if(status_this == 0) {
#pragma omp for
      for(ii=0;ii<n_m;ii++) {
	int qagstatus;
	double result, eresult;
	double x_max = r_delta[ii]/r_s[ii];
	F.function = &einasto_norm_integrand;
	F.params = &(alpha[ii]);
	qagstatus = gsl_integration_qag(&F, 0, x_max, 0, 1E-4,
					1000, GSL_INTEG_GAUSS31,
					w, &result, &eresult);
	if(qagstatus != GSL_SUCCESS) {
	  ccl_raise_gsl_warning(qagstatus, "ccl_haloprofile.c: ccl_einasto_norm_integral():");
	  status_this = CCL_ERROR_INTEG;
	  result = NAN;
	}
	norm_out[ii] = 4 * M_PI * r_s[ii] * r_s[ii] * r_s[ii] * result;
      }
    } //end omp for
  
    gsl_integration_workspace_free(w);
    if(status_this) {
      #pragma omp atomic write
      *status = status_this;
    }
  } //end omp parallel
}

static double hernquist_norm_integrand(double x, void *params)
{
  double opx=1+x;
  return x*x/(x*opx*opx*opx);
}

void ccl_hernquist_norm_integral(int n_m, double *r_s, double *r_delta,
			       double *norm_out,int *status)
{
#pragma omp parallel default(none)		\
  shared(n_m, r_s, r_delta, norm_out, status)
  {
    int ii;
    int status_this=0;
    gsl_function F;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    
    if (w == NULL)
      status_this = CCL_ERROR_MEMORY;
    
    if(status_this == 0) {
#pragma omp for
      for(ii=0;ii<n_m;ii++) {
	int qagstatus;
	double result, eresult;
	double x_max = r_delta[ii]/r_s[ii];
	F.function = &hernquist_norm_integrand;
	F.params = NULL;
	qagstatus = gsl_integration_qag(&F, 0, x_max, 0, 1E-4,
					1000, GSL_INTEG_GAUSS31,
					w, &result, &eresult);
	if(qagstatus != GSL_SUCCESS) {
	  ccl_raise_gsl_warning(qagstatus, "ccl_haloprofile.c: ccl_hernquist_norm_integral():");
	  status_this = CCL_ERROR_INTEG;
	  result = NAN;
	}
	norm_out[ii] = 4 * M_PI * r_s[ii] * r_s[ii] * r_s[ii] * result;
      }
    } //end omp for
  
    gsl_integration_workspace_free(w);
    if(status_this) {
      #pragma omp atomic write
      *status = status_this;
    }
  } //end omp parallel
}
