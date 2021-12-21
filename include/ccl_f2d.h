/** @file */

#ifndef __CCL_F2D_H_INCLUDED__
#define __CCL_F2D_H_INCLUDED__

#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

CCL_BEGIN_DECLS

//f2d extrapolation types for early times
typedef enum ccl_f2d_extrap_growth_t
{
  ccl_f2d_cclgrowth = 401, //Use CCL's linear growth
  ccl_f2d_constantgrowth = 403, //Use a constant growth factor
  ccl_f2d_no_extrapol = 404, //Do not extrapolate, just throw an exception
} ccl_f2d_extrap_growth_t;

//f2d interpolation types
typedef enum ccl_f2d_interp_t
{
  ccl_f2d_3 = 303, //Bicubic interpolation
} ccl_f2d_interp_t;

/**
 * Struct containing a 2D power spectrum
 */
typedef struct {
  double lkmin,lkmax; /**< Edges in log(k)*/
  double amin,amax; /**< Edges in a*/
  int is_factorizable; /**< Is this factorizable into k- and a-dependent functions? */
  int is_k_constant; /**< no k-dependence, just return 1*/
  int is_a_constant; /**< no a-dependence, just return 1*/
  int extrap_order_lok; /**< Order of extrapolating polynomial in log(k) for low k (0, 1 or 2)*/
  int extrap_order_hik; /**< Order of extrapolating polynomial in log(k) for high k (0, 1 or 2)*/
  ccl_f2d_extrap_growth_t extrap_linear_growth;  /**< Extrapolation type at high redshifts*/
  int is_log; /**< Do I hold the values of log(f(k,a))?*/
  double growth_factor_0; /**< Constant extrapolating growth factor*/
  int growth_exponent; /**< Power to which growth should be exponentiated*/
  gsl_spline *fk; /**< Spline holding the values of the k-dependent factor*/
  gsl_spline *fa; /**< Spline holding the values of the a-dependent factor*/
  gsl_spline2d *fka; /**< Spline holding the values of f(k,a)*/
} ccl_f2d_t;

/**
 * Create a ccl_f2d_t structure.
 * @param na number of elements in a_arr.
 * @param a_arr array of scale factor values at which the function is defined. The array should be ordered.
 * @param nk number of elements of lk_arr.
 * @param lk_arr array of logarithmic wavenumbers at which the function is defined (i.e. this array contains ln(k), NOT k). The array should be ordered.
 * @param fka_arr array of size na * nk containing the 2D function. The 2D ordering is such that fka_arr[ia*nk+ik] = f(k=exp(lk_arr[ik]),a=a_arr[ia]).
 * @param fk_arr array of size nk containing the k-dependent part of the function. Only relevant if is_factorizable is true.
 * @param fa_arr array of size na containing the a-dependent part of the function. Only relevant if is_factorizable is true.
 * @param is_factorizable if not 0, fk_arr and fa_arr will be used as 1-D arrays to construct a factorizable 2D function.
 * @param extrap_order_lok Order of the polynomial that extrapolates on wavenumbers smaller than the minimum of lk_arr. Allowed values: 0 (constant), 1 (linear extrapolation) and 2 (quadratic extrapolation). Extrapolation happens in ln(k).
 * @param extrap_order_hik Order of the polynomial that extrapolates on wavenumbers larger than the maximum of lk_arr. Allowed values: 0 (constant), 1 (linear extrapolation) and 2 (quadratic extrapolation). Extrapolation happens in ln(k).
 * @param extrap_linear_growth: ccl_f2d_extrap_growth_t value defining how the function with scale factors below the interpolation range. Allowed values: ccl_f2d_cclgrowth (scale with the CCL linear growth factor), ccl_f2d_constantgrowth (scale by multiplying the function at the earliest available scale factor by a constant number, defined by `growth_factor_0`), ccl_f2d_no_extrapol (throw an error if the function is ever evaluated outside the interpolation range in a). Note that, above the interpolation range (i.e. for low redshifts), the function will be assumed constant.
 * @param is_fka_log: if not zero, `fka_arr` contains ln(f(k,a)) instead of f(k,a). If the function is factorizable, then `fk_arr` holds ln(K(k)) and `fa_arr` holds ln(A(a)), where f(k,a)=K(k)*A(a).
 * @param growth_factor_0: custom growth function. Irrelevant if extrap_linear_growth!=ccl_f2d_constantgrowth.
 * @param growth_exponent: power to which the extrapolating growth factor should be exponentiated when extrapolating (e.g. usually 2 for linear power spectra).
 * @param interp_type: 2D interpolation method. Currently only ccl_f2d_3 is implemented (bicubic interpolation).
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
ccl_f2d_t *ccl_f2d_t_new(int na,double *a_arr,
			 int nk,double *lk_arr,
			 double *fka_arr,
			 double *fk_arr,
			 double *fa_arr,
			 int is_factorizable,
			 int extrap_order_lok,
			 int extrap_order_hik,
			 ccl_f2d_extrap_growth_t extrap_linear_growth,
			 int is_fka_log,
			 double growth_factor_0,
			 int growth_exponent,
			 ccl_f2d_interp_t interp_type,
			 int *status);

/**
 * Evaluate 2D function of k and a defined by ccl_f2d_t structure.
 * @param fka ccl_f2d_t structure defining f(k,a).
 * @param lk Natural logarithm of the wavenumber.
 * @param a Scale factor.
 * @param cosmo ccl_cosmology structure, only needed if evaluating f(k,a) at small scale factors outside the interpolation range, and if fka was initialized with extrap_linear_growth = ccl_f2d_cclgrowth.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
double ccl_f2d_t_eval(ccl_f2d_t *fka,double lk,double a,void *cosmo,
		      int *status);

/**
 * Evaluate logarithmic derivative of 2D function of k and a defined by ccl_f2d_t structure wrt k.
 * @param fka ccl_f2d_t structure defining f(k,a).
 * @param lk Natural logarithm of the wavenumber.
 * @param a Scale factor.
 * @param cosmo ccl_cosmology structure, only needed if evaluating f(k,a) at small scale factors outside the interpolation range, and if fka was initialized with extrap_linear_growth = ccl_f2d_cclgrowth.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
double ccl_f2d_t_dlogf_dlk_eval(ccl_f2d_t *f2d,double lk,double a,void *cosmo, int *status);


/**
 * F2D structure destructor.
 * Frees up all memory associated with a f2d structure.
 * @param fka Structure to be freed.
 */
void ccl_f2d_t_free(ccl_f2d_t *fka);

/**
 * Make a copy of a ccl_f2d_t structure.
 * @param f2d_o old ccl_f2d_t structure.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
ccl_f2d_t *ccl_f2d_t_copy(ccl_f2d_t *f2d_o, int *status);

CCL_END_DECLS

#endif
