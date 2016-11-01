#pragma once

#include "ccl_core.h"
#include "gsl/gsl_spline.h"

#define CL_TRACER_NC 1 //Tracer type 1: number counts
#define CL_TRACER_WL 2 //Tracer type 2: weak lensing

//Spline wrapper
//Used to take care of evaluations outside the supported range
typedef struct {
  gsl_interp_accel *intacc; //GSL spline
  gsl_spline *spline;
  double x0,xf; //Interpolation limits
  double y0,yf; //Constant values to use beyond interpolation limit
} SplPar;

typedef struct {
  int tracer_type; //Type (see above)
  double prefac_lensing; //3*O_M*H_0^2/2
  double chimax; //Limits in chi where we care about this tracer
  double chimin;
  SplPar *spl_nz; //Spline for normalized N(z)
  SplPar *spl_bz; //Spline for linear bias
  SplPar *spl_wL; //Spline for lensing kernel
} CCL_ClTracer;

//CCL_ClTracer creator
CCL_ClTracer *ccl_cl_tracer_new(ccl_cosmology *cosmo,int tracer_type,
				int nz_n,double *z_n,double *n,
				int nz_b,double *z_b,double *b);
//CCL_ClTracer destructor
void ccl_cl_tracer_free(CCL_ClTracer *clt);
//Computes limber power spectrum for two different tracers
double ccl_angular_cl(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,CCL_ClTracer *clt2);
