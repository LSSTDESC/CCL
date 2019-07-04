/** @file */

#ifndef __CCL_CLTRACERS_H_INCLUDED__
#define __CCL_CLTRACERS_H_INCLUDED__

CCL_BEGIN_DECLS

//This is used to determine the limits of integration along
//the radial direction for a given tracer.
//The limits are given by the lowest and highest radial
//distances at which the radial kernel is CCL_FRAC_RELEVANT
//times smaller than its maximum.
#define CCL_FRAC_RELEVANT 5E-4

typedef struct {
  int der_bessel; //Bessel derivative order.
  int der_angles; //Ell-dependent prefactor.
  ccl_f2d_t *transfer; //Transfer function.
  ccl_f1d_t *kernel; //Radial kernel.
  double chi_min; //Minimum radial comoving distance for this tracer.
  double chi_max; //Maximum radial comoving distance for this tracer.
} ccl_cl_tracer_t;

/**
 * Constructor for a ccl_cl_tracer_t. See CCL note for a description of how tracers are used to compute power spectra.
 * @param cosmo Cosmology structure.
 * @param der_bessel Bessel function derivative order (-1, 0, 1 or 2). For 0, 1 and 2 this is just the derivative order. For -1, the tracer uses j_l(k*chi)/(k*chi)^2 instead of j_l(k*chi).
 * @param der_angles ell-dependent prefactor type (0, 1 or 2). 0 -> 1, 1 -> l*(l+1), 2 -> sqrt((l+2)!/(l-2)!).
 * @param n_w number of array elements in radial kernel.
 * @param chi_w values of the radial comoving distance for the radial kernel.
 * @param w_w corresponding values of the radial kernel.
 * @param na_ka number of scale factor values used to describe the transfer function.
 * @param a_ka scale factor values for transfer function.
 * @param nk_ka number of wavenumber values used to describe the transfer function.
 * @param lk_ka natural logarithm of the wavenumber in Mpc^-1 used for the transfer function.
 * @param fka_arr array containing the 2D transfer function. Should have size na_ka * nk_ka, with the wave number being the fastest varying variable. If NULL, the transfer function will be assumed factorizable.
 * @param fk_arr k-dependent part of a factorizable transfer function. Should have size nk_ka. If NULL and if fka_arr is also null, the transfer function will be assumed scale-independent (i.e. this factor is 1 everywhere).
 * @param fa_arr a-dependent part of a factorizable transfer function. Should have size na_ka. If NULL and if fka_arr is also null, the transfer function will be assumed time-independent (i.e. this factor is 1 everywhere).
 * @param is_fka_log if not zero, then it will be assumed that fka_arr, fk_arr and fa_arr hold the natural logarithm of the quantities they represent.
 * @param is_factorizable if not zero, fka_arr will be ignored, and a factorizable transfer function will be assumed using the contents of fk_arr and fa_arr.
 * @param extrap_order_lok Order of the polynomial that extrapolates on wavenumbers smaller than the minimum of lk_ka. Allowed values: 0 (constant), 1 (linear extrapolation) and 2 (quadratic extrapolation). Extrapolation happens in ln(k).
 * @param extrap_order_hik Order of the polynomial that extrapolates on wavenumbers larger than the maximum of lk_ka. Allowed values: 0 (constant), 1 (linear extrapolation) and 2 (quadratic extrapolation). Extrapolation happens in ln(k).
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * @return ccl_cl_tracer_t structure.
 */
ccl_cl_tracer_t *ccl_cl_tracer_t_new(ccl_cosmology *cosmo,
				     int der_bessel,
				     int der_angles,
				     int n_w,double *chi_w,double *w_w,
				     int na_ka,double *a_ka,
				     int nk_ka,double *lk_ka,
				     double *fka_arr,
				     double *fk_arr,
				     double *fa_arr,
				     int is_fka_log,
				     int is_factorizable,
				     int extrap_order_lok,
				     int extrap_order_hik,
				     int *status);

/**
 * ccl_tracer_t_free destructor
 */
void ccl_cl_tracer_t_free(ccl_cl_tracer_t *tr);

/**
 * Return the ell-dependent part of a tracer.
 * @param tr tracer.
 * @param ell multipole value.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * @return ell prefactor value.
 */
double ccl_cl_tracer_t_get_f_ell(ccl_cl_tracer_t *tr,double ell,int *status);

/**
 * Return the value of the radial kernel of a tracer.
 * @param tr tracer.
 * @param chi radial comoving distance in Mpc.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * @return radial kernel value.
 */
double ccl_cl_tracer_t_get_kernel(ccl_cl_tracer_t *tr,double chi,int *status);


/**
 * Return the value of the transfer function of a tracer.
 * @param tr tracer.
 * @param lk natural logarithm of the wavenumber in units of Mpc^-1.
 * @param a scale factor value.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * @return transfer function value.
 */
double ccl_cl_tracer_t_get_transfer(ccl_cl_tracer_t *tr,double lk,double a,int *status);

/**
 * Computes the radial kernel for number counts
 * @param nz number of samples over which the redshift distribution is sampled.
 * @param z_arr array of input redshifts.
 * @param nz_arr array of redshift distribution values.
 * @param normalize_nz if not zero, the input redshift distribution will be normalized to unit integral.
 * @param pchi_arr output array containing the radial kernel.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * @return transfer function value.
 */
void ccl_get_number_counts_kernel(ccl_cosmology *cosmo,
				  int nz,double *z_arr,double *nz_arr,
				  int normalize_nz,
				  double *pchi_arr,int *status);
/**
 * Return the number of samples over which the lensing kernel will be computed.
 * @param nz number of input redshifts.
 * @param z_arr array of redshifts at which the redshift distribution is sampled.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * @return number of samples
 */
int ccl_get_nchi_lensing_kernel(int nz,double *z_arr,int *status);

/**
 * Return values of the radial distance on which the lensing kernel will be computed.
 * @param nchi number of distance values.
 * @param z_max maximum redshift.
 * @param chis output array of distances.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
void ccl_get_chis_lensing_kernel(ccl_cosmology *cosmo,
				 int nchi,double z_max,
				 double *chis,int *status);

/**
 * Return lensing kernel
 * @param cosmo cosmology.
 * @param nz number of input redshifts for the redshift distribution.
 * @param z_arr input redshifts for the redshift distribution.
 * @param nz_arr input redshift distribution.
 * @param normalize_nz if not zero, will normalize redshift distribution to unit integral.
 * @param z_max maximum redshift for integrals.
 * @param nz_s number of input redshifts for the magnification bias.
 * @param zs_arr input redshifts for the magnification bias.
 * @param sz_arr magnification bias. If NULL, magnification bias will be assumed to be zero.
 * @param nchi number of distance values.
 * @param chis input array of distances.
 * @param wL_arr lensing kernel.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
void ccl_get_lensing_mag_kernel(ccl_cosmology *cosmo,
				int nz,double *z_arr,double *nz_arr,
				int normalize_nz,double z_max,
				int nz_s,double *zs_arr,double *sz_arr,
				int nchi,double *chi_arr,double *wL_arr,int *status);

/**
 * Return radial kernel for CMB lensing convergence.
 * @param cosmo cosmology.
 * @param chi_source comoving distance to source plane.
 * @param nchi number of samples in distance.
 * @param chi_arr array of radial distance values.
 * @param wchi output array containing the kernel.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
void ccl_get_kappa_kernel(ccl_cosmology *cosmo,
			  double chi_source,
			  int nchi,double *chi_arr,
			  double *wchi,int *status);


//Maximum number of tracers per collection
#define CCL_MAX_TRACERS_PER_COLLECTION 100

typedef struct {
  int n_tracers; //Number of tracers in this collection
  ccl_cl_tracer_t **ts; //Array of tracers
} ccl_cl_tracer_collection_t;

/**
 * ccl_cl_tracer_collection_t constructor.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * @return ccl_cl_tracer_collection_t structure.
 */
ccl_cl_tracer_collection_t *ccl_cl_tracer_collection_t_new(int *status);

/**
 * ccl_cl_tracer_collection_t destructor.
 */
void ccl_cl_tracer_collection_t_free(ccl_cl_tracer_collection_t *trc);

/**
 * Add a tracer to the collection.
 * @param trc collection of tracers.
 * @param tr tracer to be added.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
void ccl_add_cl_tracer_to_collection(ccl_cl_tracer_collection_t *trc,ccl_cl_tracer_t *tr,int *status);

CCL_END_DECLS

#endif
