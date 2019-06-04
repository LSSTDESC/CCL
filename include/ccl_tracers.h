/** @file */

#ifndef __CCL_CLTRACERS_H_INCLUDED__
#define __CCL_CLTRACERS_H_INCLUDED__

CCL_BEGIN_DECLS

#define CCL_FRAC_RELEVANT 5E-4

typedef struct {
  int der_bessel;
  int der_angles;
  ccl_f2d_t *transfer;
  ccl_f1d_t *kernel;
  double chi_min;
  double chi_max;
} ccl_cl_tracer_t;

ccl_cl_tracer_t *ccl_cl_tracer_t_new(ccl_cosmology *cosmo,
				     int der_bessel,
				     int der_angles,
				     int n_w,double *chi_w,double *w_w,
				     int na_ka,double *a_ka,
				     int nk_ka,double *lk_ka,
				     double *fka_arr,
				     double *fk_arr,
				     double *fa_arr,
				     int is_factorizable,
				     int is_k_powerlaw,
				     double k_powerlaw_exponent,
				     int extrap_order_lok,
				     int extrap_order_hik,
				     int *status);

void ccl_cl_tracer_t_free(ccl_cl_tracer_t *tr);

double ccl_cl_tracer_t_get_f_ell(ccl_cl_tracer_t *tr,double ell,int *status);

double ccl_cl_tracer_t_get_kernel(ccl_cl_tracer_t *tr,double chi,int *status);

double ccl_cl_tracer_t_get_transfer(ccl_cl_tracer_t *tr,double lk,double a,int *status);
				 
double ccl_cl_tracer_t_get_cl_contribution(ccl_cl_tracer_t *tr,
					   double ell,double chi,double lk,double a,
					   int *status);

CCL_END_DECLS

#endif
