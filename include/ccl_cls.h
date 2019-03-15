/** @file */
#ifndef __CCL_CLS_H_INCLUDED__
#define __CCL_CLS_H_INCLUDED__

typedef enum ccl_tracer_t
{
  ccl_number_counts_tracer = 1,
  ccl_weak_lensing_tracer  = 2,
  ccl_cmb_lensing_tracer   = 3,
} ccl_tracer_t;

typedef enum ccl_tracer_func_t
{
  ccl_trf_nz = 201, //Redshift distribution
  ccl_trf_bz = 202, //Clustering bias
  ccl_trf_sz = 203, //Magnification bias
  ccl_trf_rf = 204, //Aligned fraction
  ccl_trf_ba = 205, //Alignment bias
  ccl_trf_wL = 206, //Weak lensing window function
  ccl_trf_wM = 207, //Magnification window function
} ccl_tracer_func_t;

CCL_BEGIN_DECLS

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
  double chi_source; //Comoving distance to the source (for CMB lensing)
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


/**
 * Constructor for a ClTracer.
 * @param Tracer_type pass ccl_number_counts_tracer (number counts), ccl_weak_lensing_tracer (weak lensing) or ccl_cmb_lensing_tracer (CMB lensing)
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
 * @param z_source Redshift of source plane for CMB lensing (z~1100 for CMB lensing).
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return CCL_ClTracer object
 */
CCL_ClTracer *ccl_cl_tracer(ccl_cosmology *cosmo,int tracer_type,
				int has_rsd,int has_magnification,int has_intrinsic_alignment,
				int nz_n,double *z_n,double *n,
				int nz_b,double *z_b,double *b,
				int nz_s,double *z_s,double *s,
				int nz_ba,double *z_ba,double *ba,
				int nz_rf,double *z_rf,double *rf,
				double z_source,int * status);

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
CCL_ClTracer *ccl_cl_tracer_number_counts(ccl_cosmology *cosmo,
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
CCL_ClTracer *ccl_cl_tracer_number_counts_simple(ccl_cosmology *cosmo,
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
CCL_ClTracer *ccl_cl_tracer_lensing(ccl_cosmology *cosmo,
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
CCL_ClTracer *ccl_cl_tracer_lensing_simple(ccl_cosmology *cosmo,
					       int nz_n,double *z_n,double *n, int * status);


/**
 * Simplified constructor for a CMB lensing ClTracer.
 * @param z_source Redshift of source plane (z~1100 for CMB lensing).
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return CCL_ClTracer object
 */
CCL_ClTracer *ccl_cl_tracer_cmblens(ccl_cosmology *cosmo,double z_source,int *status);

/**
 * Destructor for a Cltracer
 * @param clt a Cltracer
 * @return void
 */
void ccl_cl_tracer_free(CCL_ClTracer *clt);

/**
 * Method to return certain redshift or distance-dependent internal quantities for a given tracer.
 * @param cosmo Cosmological parameters
 * @param clt ClTracer object
 * @param a scale factor at which the function is to be evaluated
 * @param func_code integer defining which internal function to evaluate. Choose between:
 * ccl_trf_nz (redshift distribution), ccl_trf_bz (clustering bias), ccl_trf_sz (magnification bias),
 * ccl_trf_rf (aligned fraction), ccl_trf_ba (alignment bias),
 * ccl_trf_wL (weak lensing window function), ccl_trf_wM (magnification window function)
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return interpolated value of the requested function
 */
double ccl_get_tracer_fa(ccl_cosmology *cosmo,CCL_ClTracer *clt,double a,int func_code,int *status);

/**
 * Method to return certain redshift or distance-dependent internal quantities for a given tracer.
 * @param cosmo Cosmological parameters
 * @param clt ClTracer object
 * @param na number of points at which the function will be evaluated
 * @param a na values of the scale factor at which the function is to be evaluated
 * @param fa output array with na values that will store the interpolated function values
 * @param func_code integer defining which internal function to evaluate. Choose between:
 * ccl_trf_nz (redshift distribution), ccl_trf_bz (clustering bias), ccl_trf_sz (magnification bias),
 * ccl_trf_rf (aligned fraction), ccl_trf_ba (alignment bias),
 * ccl_trf_wL (weak lensing window function), ccl_trf_wM (magnification window function)
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
int ccl_get_tracer_fas(ccl_cosmology *cosmo,CCL_ClTracer *clt,int na,double *a,double *fa,
		       int func_code,int *status);

//Workspace for C_ell computations
typedef struct {
  int lmax; //*Maximum multipole
  int l_limber; //*All power spectra for l>l_limber will be computed using Limber's approximation
  double l_logstep; //*Logarithmic step factor used at low l
  int l_linstep; //*Linear step used at high l
  int n_ls; //Number of multipoles that result from the previous combination of parameters
  int *l_arr; //*Array of multipole values resulting from the previous parameters
} CCL_ClWorkspace;

//CCL_ClWorkspace constructor
CCL_ClWorkspace *ccl_cl_workspace_new(int lmax,int l_limber,
				      double l_logstep,int l_linstep,
				      int *status);
//CCL_ClWorkspace simplified constructor
CCL_ClWorkspace *ccl_cl_workspace_new_limber(int lmax, double l_logstep,int l_linstep,int *status);
//CCL_ClWorkspace destructor
void ccl_cl_workspace_free(CCL_ClWorkspace *w);

/**
 * Computes limber or non-limber power spectrum for two different tracers
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
		     CCL_ClTracer *clt1,CCL_ClTracer *clt2,ccl_p2d_t *psp,
		     int nl_out,int *l,double *cl,int *status);

CCL_END_DECLS


#endif
