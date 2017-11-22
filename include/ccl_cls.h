/** @file */
#ifdef __cplusplus
extern "C" {
#endif

#pragma once

#include "ccl_core.h"
#include "ccl_utils.h"

#define CL_TRACER_NC 1 //Tracer type 1: number counts
#define CL_TRACER_WL 2 //Tracer type 2: weak lensing


/**
 * ClTracer structure, used to contain everything
 * that a Cl tracer could have, such as splines for
 * various quantities and limits on the value of chi
 * that this tracer deals with.
 */
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
  int computed_transfer;
  int n_ls;
  int *n_k;
  SplPar **spl_transfer;
} CCL_ClTracer;

/**
 * Constructor for a ClTracer.
 * @param Tracer_type pass CL_TRACER_NC (number counts) or CL_TRACER_WL (weak lensing)
 * @param has_rsd Set to 1 if you want to compute the RSD contribution to number counts (0 otherwise)
 * @param has_magnification Set to 1 if you want to compute the magnification contribution to number counts (0 otherwise)
 * @param has_intrinsic_alignment Set to 1 if you want to compute the IA contribution to shear
 * @param nz_n Number of bins in z_n and n
 * @param z_n Redshifts for each redshift interval of n
 * @param n Number count of objects per redshift interval (Note: arbitrary normalization - renormalized inside)
 * @param nz_b Number of bins in z_b and b
 * @param z_b Redshifts for each redshift interval of b
 * @param b Clustering bias in each redshift bin
 * @param nz_s Number of bins in z_s and s
 * @param z_s Redshifts for each redshift interval of s
 * @param s Magnification bias in each redshift bin
 * @param nz_ba Number of bins in z_ba and ba
 * @param z_ba Redshifts for each redshift interval of ba
 * @param ba Alignment bias in each redshift bin
 * @param nz_rf Number of bins in z_f and f
 * @param z_rf Redshifts for each redshift interval of rf
 * @param rf Aligned red fraction in each redshift bin
 * @return CCL_ClTracer object
 */
CCL_ClTracer *ccl_cl_tracer_new(ccl_cosmology *cosmo,int tracer_type,
				int has_rsd,int has_magnification,int has_intrinsic_alignment,
				int nz_n,double *z_n,double *n,
				int nz_b,double *z_b,double *b,
				int nz_s,double *z_s,double *s,
				int nz_ba,double *z_ba,double *ba,
				int nz_rf,double *z_rf,double *rf, int * status);

/**
 * Simplified constructor for a clustering ClTracer.
 * @param cosmo Cosmological parameters
 * @param has_rsd Set to 1 if you want to compute the RSD contribution to number counts (0 otherwise)
 * @param has_magnification Set to 1 if you want to compute the magnification contribution to number counts (0 otherwise)
 * @param nz_n Number of bins in z_n and n
 * @param z_n Redshifts for each redshift interval of n
 * @param n Number count of objects per redshift interval (Note: arbitrary normalization - renormalized inside)
 * @param nz_b Number of bins in z_b and b
 * @param z_b Redshifts for each redshift interval of b
 * @param b Clustering bias in each redshift bin
 * @param nz_s Number of bins in z_s and s
 * @param z_s Redshifts for each redshift interval of s
 * @param s Magnification bias in each redshift bin
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return CCL_ClTracer object
 */
CCL_ClTracer *ccl_cl_tracer_number_counts_new(ccl_cosmology *cosmo,
					      int has_rsd,int has_magnification,
					      int nz_n,double *z_n,double *n,
					      int nz_b,double *z_b,double *b,
					      int nz_s,double *z_s,double *s, int * status);


/**
 * Simplified constructor for a ClTracer without magnification nor RSD.
 * @param nz_n Number of bins in z_n and n
 * @param z_n Redshifts for each redshift interval of n
 * @param n Number count of objects per redshift interval (Note: arbitrary normalization - renormalized inside)
 * @param nz_b Number of bins in z_b and b
 * @param z_b Redshifts for each redshift interval of b
 * @param b Clustering bias in each redshift bin
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return CCL_ClTracer object
 */
CCL_ClTracer *ccl_cl_tracer_number_counts_simple_new(ccl_cosmology *cosmo,
						     int nz_n,double *z_n,double *n,
						     int nz_b,double *z_b,double *b, int * status);

/**
 * Simplified constructor for a lensing ClTracer.
 * @param has_intrinsic_alignment Set to 1 if you want to compute the IA contribution to shear
 * @param nz_n Number of bins in z_n and n
 * @param z_n Redshifts for each redshift interval of n
 * @param n Number count of objects per redshift interval (Note: arbitrary normalization - renormalized inside)
 * @param nz_ba Number of bins in z_ba and ba
 * @param z_ba Redshifts for each redshift interval of ba
 * @param ba Alignment bias in each redshift bin
 * @param nz_rf Number of bins in z_f and f
 * @param z_rf Redshifts for each redshift interval of rf
 * @param rf Aligned red fraction in each redshift bin
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return CCL_ClTracer object
 */
CCL_ClTracer *ccl_cl_tracer_lensing_new(ccl_cosmology *cosmo,
					int has_alignment,
					int nz_n,double *z_n,double *n,
					int nz_ba,double *z_ba,double *ba,
					int nz_rf,double *z_rf,double *rf, int * status);

/**
 * Simplified constructor for a lensing ClTracer without intrinsic alignment.
 * @param nz_n Number of bins in z_n and n
 * @param z_n Redshifts for each redshift interval of n
 * @param n Number count of objects per redshift interval (Note: arbitrary normalization - renormalized inside)
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return CCL_ClTracer object
 */
CCL_ClTracer *ccl_cl_tracer_lensing_simple_new(ccl_cosmology *cosmo,
					       int nz_n,double *z_n,double *n, int * status);


/**
 * Destructor for a Cltracer
 * @param clt a Cltracer
 * @return void
 */
void ccl_cl_tracer_free(CCL_ClTracer *clt);


#define CCL_NONLIMBER_METHOD_NATIVE 1
#define CCL_NONLIMBER_METHOD_ANGPOW 2
//Workspace for C_ell computations
typedef struct {
  int nlimb_method;
  double zmin;
  double dchi; //Spacing in comoving distance to use for the LOS integrals
  double dlk; //Logarithmic spacing in wavenumber
  int lmax; //Maximum multipole
  int l_limber; //All power spectra for l>l_limber will be computed using Limber's approximation
  double l_logstep; //Logarithmic step factor used at low l
  int l_linstep; //Linear step used at high l
  int n_ls; //Number of multipoles that result from the previous combination of parameters
  int *l_arr; //Array of multipole values resulting from the previous parameters
} CCL_ClWorkspace;

//CCL_ClWorkspace constructor
CCL_ClWorkspace *ccl_cl_workspace_new(int lmax,int l_limber,int non_limber_method,
				      double l_logstep,int l_linstep,
				      double dchi,double dlk,double zmin,int *status);
//CCL_ClWorkspace simplified constructor
CCL_ClWorkspace *ccl_cl_workspace_new_default(int lmax,int l_limber,int *status);
//CCL_ClWorkspace destructor
void ccl_cl_workspace_free(CCL_ClWorkspace *w);

/**
 * Computes limber power spectrum for two different tracers
 * @param cosmo Cosmological parameters
 * @param w a ClWorkspace
 * @param clt1 a Cltracer
 * @param clt2 a Cltracer
 * @param nl_out the maximum to ell to compute C_ell
 * @param l an array of ell values
 * @param cl the C_ell output array
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_angular_cls(ccl_cosmology *cosmo,CCL_ClWorkspace *w,
		     CCL_ClTracer *clt1,CCL_ClTracer *clt2,
		     int nl_out,int *l,double *cl,int *status);

/*
 * Computes limber power spectrum for two different tracers
 * @param cosmo Cosmological parameters
 * @param clt1 a Cltracer
 * @param clt2 a Cltracer
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
  /*
double ccl_angular_cl(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,CCL_ClTracer *clt2, int * status);
  
*/


#ifdef __cplusplus
}
#endif
