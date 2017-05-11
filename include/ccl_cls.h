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
  double zmin; //Limits in chi where we care about this tracer
  double zmax;
  int has_rsd;
  int has_magnification;
  int has_intrinsic_alignment;
  SplPar *spl_nz; //Spline for normalized N(z)
  SplPar *spl_bz; //Spline for linear bias
  SplPar *spl_sz; //Spline for magnification bias
  SplPar *spl_rf; //Spline for red fraction
  SplPar *spl_ba; //Spline for alignment bias
  SplPar *spl_wL; //Spline for lensing kernel
  SplPar *spl_wM; //Spline for magnification
} CCL_ClTracer;

//Generic CCL_ClTracer creator
// Tracer_type: pass CL_TRACER_NC (number counts) or CL_TRACER_WL (weak lensing)
// * has_rsd -> set to 1 if you want to compute the RSD contribution to number counts (0 otherwise)
// * has_magnification -> set to 1 if you want to compute the magnification contribution to
//   number counts (0 otherwise)
// * has_intrinsic_alignment -> set to 1 if you want to compute the IA contribution to shear
// * nz_n, z_n, n -> z_n and n are arrays for the number count of objects per redshift interval
//                   (arbitrary normalization - renormalized inside).
//                   These arrays should contain nz_n elements each.
// * nz_b, z_b, b -> same as above for the clustering bias
// * nz_s, z_s, s -> same as above for the magnification bias
// * nz_ba, z_ba, ba -> same as above for the alignment bias
// * nz_rf, z_rf, rf -> same as above for the aligned (red) fraction
CCL_ClTracer *ccl_cl_tracer_new(ccl_cosmology *cosmo,int tracer_type,
				int has_rsd,int has_magnification,int has_intrinsic_alignment,
				int nz_n,double *z_n,double *n,
				int nz_b,double *z_b,double *b,
				int nz_s,double *z_s,double *s,
				int nz_ba,double *z_ba,double *ba,
				int nz_rf,double *z_rf,double *rf, int * status);
//Simplified version of the above for number counts
CCL_ClTracer *ccl_cl_tracer_number_counts_new(ccl_cosmology *cosmo,
					      int has_rsd,int has_magnification,
					      int nz_n,double *z_n,double *n,
					      int nz_b,double *z_b,double *b,
					      int nz_s,double *z_s,double *s, int * status);
//More simplified version (no RSD, no magnification) of the above for number counts
CCL_ClTracer *ccl_cl_tracer_number_counts_simple_new(ccl_cosmology *cosmo,
						     int nz_n,double *z_n,double *n,
						     int nz_b,double *z_b,double *b, int * status);
//Simplified version of the above for shear
CCL_ClTracer *ccl_cl_tracer_lensing_new(ccl_cosmology *cosmo,
					int has_alignment,
					int nz_n,double *z_n,double *n,
					int nz_ba,double *z_ba,double *ba,
					int nz_rf,double *z_rf,double *rf, int * status);
//More simplified version (no IA) of the above for shear
CCL_ClTracer *ccl_cl_tracer_lensing_simple_new(ccl_cosmology *cosmo,
					       int nz_n,double *z_n,double *n, int * status);
//CCL_ClTracer destructor
void ccl_cl_tracer_free(CCL_ClTracer *clt);


#define CCL_NONLIMBER_METHOD_NATIVE 1
#define CCL_NONLIMBER_METHOD_ANGPOW 2
//Workspace for C_ell computations
typedef struct {
  int nlimb_method;
  double dchi; //Spacing in comoving distance to use for the LOS integrals
  int lmax; //Maximum multipole
  int l_limber; //All power spectra for l>l_limber will be computed using Limber's approximation
  double l_logstep; //Logarithmic step factor used at low l
  int l_linstep; //Linear step used at high l
  int n_ls; //Number of multipoles that result from the previous combination of parameters
  int *l_arr; //Array of multipole values resulting from the previous parameters
} CCL_ClWorkspace;

//CCL_ClWorkspace constructor
CCL_ClWorkspace *ccl_cl_workspace_new(int lmax,int l_limber,int non_limber_method,
				      double l_logstep,int l_linstep,double dchi,int *status);
//CCL_ClWorkspace simplified constructor
CCL_ClWorkspace *ccl_cl_workspace_new_default(int lmax,int l_limber,int *status);
//CCL_ClWorkspace destructor
void ccl_cl_workspace_free(CCL_ClWorkspace *w);

//Computes limber power spectrum for two different tracers
void ccl_angular_cls(ccl_cosmology *cosmo,CCL_ClWorkspace *w,
		     CCL_ClTracer *clt1,CCL_ClTracer *clt2,
		     int nl_out,int *l,double *cl,int *status);
