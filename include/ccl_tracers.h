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
				     int is_fka_log,
				     int is_factorizable,
				     int extrap_order_lok,
				     int extrap_order_hik,
				     int *status);

void ccl_cl_tracer_t_free(ccl_cl_tracer_t *tr);

double ccl_cl_tracer_t_get_f_ell(ccl_cl_tracer_t *tr,double ell,int *status);

double ccl_cl_tracer_t_get_kernel(ccl_cl_tracer_t *tr,double chi,int *status);

double ccl_cl_tracer_t_get_transfer(ccl_cl_tracer_t *tr,double lk,double a,int *status);
				 
void ccl_get_number_counts_kernel(ccl_cosmology *cosmo,
				  int nz,double *z_arr,double *nz_arr,
				  int normalize_nz,
				  double *pchi_arr,int *status);
int ccl_get_nchi_lensing_kernel(int nz,double *z_arr,int *status);
void ccl_get_chis_lensing_kernel(ccl_cosmology *cosmo,
				 int nchi,double z_max,
				 double *chis,int *status);
void ccl_get_lensing_mag_kernel(ccl_cosmology *cosmo,
				int nz,double *z_arr,double *nz_arr,
				int normalize_nz,double z_max,
				int nz_s,double *zs_arr,double *sz_arr,
				int nchi,double *chi_arr,double *wL_arr,int *status);
void ccl_get_kappa_kernel(ccl_cosmology *cosmo,
			  double chi_source,
			  int nchi,double *chi_arr,
			  double *wchi,int *status);


typedef struct {
  int n_tracers;
  ccl_cl_tracer_t **ts;
} ccl_cl_tracer_collection_t;

ccl_cl_tracer_collection_t *ccl_cl_tracer_collection_t_new(int *status);

void ccl_cl_tracer_collection_t_free(ccl_cl_tracer_collection_t *trc);

void ccl_add_cl_tracer_to_collection(ccl_cl_tracer_collection_t *trc,ccl_cl_tracer_t *tr,int *status);

CCL_END_DECLS

#endif
