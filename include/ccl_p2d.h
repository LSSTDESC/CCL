/* @file */

#ifndef __CCL_P2D_H_INCLUDED__
#define __CCL_P2D_H_INCLUDED__

#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

CCL_BEGIN_DECLS
  
/**
 * Evaluate power spectrum defined by ccl_p2d_t structure.
 * @param na number of elements in a_arr.
 * @param a_arr array of scale factor values at which the power spectrum is defined. The array should be ordered.
 * @param nk number of elements of lk_arr.
 * @param lk_arr array of logarithmic wavenumbers at which the power spectrum is defined (i.e. this array contains ln(k), NOT k). The array should be ordered.
 * @param pk_arr array of size na * nk containing the 2D power spectrum. The 2D ordering is such that pk_arr[ia*nk+ik] = P(k=exp(lk_arr[ik]),a=a_arr[ia]).
 * @param extrap_order_lok Order of the polynomial that extrapolates on wavenumbers smaller than the minimum of lk_arr. Allowed values: 0 (constant), 1 (linear extrapolation) and 2 (quadratic extrapolation). Extrapolation happens in ln(k).
 * @param extrap_order_hik Order of the polynomial that extrapolates on wavenumbers larger than the maximum of lk_arr. Allowed values: 0 (constant), 1 (linear extrapolation) and 2 (quadratic extrapolation). Extrapolation happens in ln(k).
 * @param extrap_linear_growth: ccl_p2d_extrap_growth_t value defining how the power spectrum is scaled on scale factors below the interpolation range. Allowed values: ccl_p2d_cclgrowth (scale with the CCL linear growth factor), ccl_p2d_customgrowth (scale with a custom function of redshift passed through `growth`), ccl_p2d_constantgrowth (scale by multiplying the power spectrum at the earliest available scale factor by a constant number, defined by `growth_factor_0`), ccl_p2d_no_extrapol (throw an error if the power spectrum is ever evaluated outside the interpolation range in a).
 * @param is_pk_log: if not zero, `pk_arr` contains ln(P(k,a)) instead of P(k,a).
 * @param growth: custom growth function. Irrelevant if extrap_linear_growth!=ccl_p2d_customgrowth.
 * @param growth_factor_0: custom growth function. Irrelevant if extrap_linear_growth!=ccl_p2d_constantgrowth.
 * @param interp_type: 2D interpolation method. Currently only ccl_p2d_3 is implemented (bicubic interpolation).
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
ccl_p2d_t *ccl_p2d_t_new(int na,double *a_arr,
			 int nk,double *lk_arr,
			 double *pk_arr,
			 int extrap_order_lok,
			 int extrap_order_hik,
			 ccl_p2d_extrap_growth_t extrap_linear_growth,
			 int is_pk_log,
			 double (*growth)(double),
			 double growth_factor_0,
			 ccl_p2d_interp_t interp_type,
			 int *status);

/**
 * Evaluate power spectrum defined by ccl_p2d_t structure.
 * @param psp ccl_p2d_t structure defining P(k,a).
 * @param lk Natural logarithm of the wavenumber.
 * @param a Scale factor.
 * @param cosmo ccl_cosmology structure, only needed if evaluating P(k,a) at small scale factors outside the interpolation range, and if psp was initialized with extrap_linear_growth = ccl_p2d_cclgrowth.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 */
double ccl_p2d_t_eval(ccl_p2d_t *psp,double lk,double a,ccl_cosmology *cosmo,
		      int *status);

/**
 * P2D structure destructor.
 * Frees up all memory associated with a p2d structure.
 * @param psp Structure to be freed.
 */
void ccl_p2d_t_free(ccl_p2d_t *psp);

CCL_END_DECLS

#endif

