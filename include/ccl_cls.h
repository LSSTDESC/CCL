#pragma once

#include "ccl_core.h"
#include "gsl/gsl_spline.h"

#define CL_TRACER_NC 1
#define CL_TRACER_WL 2

typedef struct {
  gsl_interp_accel *intacc;
  gsl_spline *spline;
  double x0,xf;
  double y0,yf;
} SplPar;

typedef struct {
  int tracer_type;
  double prefac_lensing;
  double chimax;
  double chimin;
  SplPar *spl_nz;
  SplPar *spl_bz;
  SplPar *spl_wL;
} CCL_ClTracer;

CCL_ClTracer *ccl_cl_tracer_new(ccl_cosmology *cosmo,int tracer_type,
				int nz_n,double *z_n,double *n,
				int nz_b,double *z_b,double *b);
void ccl_tracer_free(CCL_ClTracer *clt);
double ccl_angular_cl(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,CCL_ClTracer *clt2);
