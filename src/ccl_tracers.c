#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "ccl.h"

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
				     int *status)
{
  ccl_cl_tracer_t *tr=NULL;

  if((der_angles<0) || (der_angles>2)) {
    *status=CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: ccl_cl_tracer_new: "
				     "der_angles must be between 0 and 2\n");
  }
  if((der_bessel<0) || (der_bessel>2)) {
    *status=CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: ccl_cl_tracer_new: "
				     "der_bessel must be between 0 and 2\n");
  }

  if(*status==0) {
    tr=malloc(sizeof(ccl_cl_tracer_t));
    if(tr==NULL)
      *status=CCL_ERROR_MEMORY;
  }

  if(*status==0) {
    tr->der_angles=der_angles;
    tr->der_bessel=der_bessel;
    tr->kernel=NULL; //Initialize these to NULL
    tr->transfer=NULL; //Initialize these to NULL
    tr->chi_min=0;
    tr->chi_max=1E15;
  }
  
  if(*status==0) {
    //Initialize radial kernel
    if((n_w>0) && (chi_w!=NULL) && (w_w!=NULL)) {
      tr->kernel=ccl_f1d_t_new(n_w,chi_w,w_w,0,0);
      if(tr->kernel==NULL) //CHECK IF THIS IS EXPECTED
	*status=CCL_ERROR_MEMORY;
    }
  }

  //Find kernel edges
  if(*status=0) {
    int ichi;
    double w_max=w_w[0];

    //Find maximum of radial kernel
    for(ichi=0;ichi<n_w;ichi++) {
      if(w_w[ichi]>=w_max)
	w_max=w_w[ichi];
    }

    //Multiply by fraction
    w_max*=CCL_FRAC_RELEVANT;

    // Initialize as the original edges in case we don't find an interval
    tr->chi_min=chi_w[0];
    tr->chi_max=chi_w[n_w-1];

    //Find minimum
    for(ichi=0;ichi<n_w;ichi++) {
      if(w_w[ichi]>=w_max) {
	tr->chi_min=chi_w[ichi];
	break;
      }
    }

    //Find maximum
    for(ichi=n_w-1;ichi>=0;ichi--) {
      if(w_w[ichi]>=w_max) {
	tr->chi_max=chi_w[ichi];
	break;
      }
    }
  }

  if(*status==0) {
    if((na_ka>0) && (nk_ka>0) &&
       (a_ka!=NULL) && (lk_ka!=NULL) &&
       ((fka_arr!=NULL) ||
	((fk_arr!=NULL) && (fa_arr!=NULL)))) {
      tr->transfer=ccl_f2d_t_new(na_ka,a_ka, //na, a_arr
				 nk_ka,lk_ka, //nk, lk_arr
				 fka_arr, //fka_arr
				 fk_arr, //fk_arr
				 fa_arr, //fa_arr
				 is_factorizable, //is factorizable
				 is_k_powerlaw, //is_k_powerlaw
				 k_powerlaw_exponent, //k_powerlaw_exponent
				 extrap_order_lok, //extrap_order_lok
				 extrap_order_hik, //extrap_order_hik
				 ccl_f2d_constantgrowth, //extrap_linear_growth
				 0, //is_fka_log
				 NULL, //growth (function)
				 1, //growth_factor_0 -> will assume constant transfer function
				 0, //growth_exponent
				 ccl_f2d_3, //interp_type
				 status);
      if(tr->transfer==NULL) //CHECK IF THIS IS EXPECTED
	*status=CCL_ERROR_MEMORY;
    }
  }
}

void ccl_cl_tracer_t_free(ccl_cl_tracer_t *tr)
{
  if(tr!=NULL) {
    if(tr->transfer!=NULL)
      ccl_f2d_t_free(tr->transfer);
    if(tr->kernel!=NULL)
      ccl_f1d_t_free(tr->kernel);
    free(tr);
  }
}

double ccl_cl_tracer_t_get_f_ell(ccl_cl_tracer_t *tr,double ell,int *status)
{
  if(tr!=NULL) {
    if(tr->der_angles==1)
      return ell*(ell+1.);
    else if(tr->der_angles==2) {
      if(ell>1)
	return sqrt((ell+2)*(ell+1)*ell*(ell-1));
      else
	return 0;
    }
    else
      return 1;
  }
  else
    return 1;
}

double ccl_cl_tracer_t_get_kernel(ccl_cl_tracer_t *tr,double chi,int *status)
{
  if(tr!=NULL) {
    if(tr->kernel!=NULL)
      ccl_f1d_t_eval(tr->kernel,chi);
    else
      return 1;
  }
  else
    return 1;
}

double ccl_cl_tracer_t_get_transfer(ccl_cl_tracer_t *tr,double lk,double a,int *status)
{
  if(tr!=NULL) {
    if(tr->transfer!=NULL)
      ccl_f2d_t_eval(tr->transfer,lk,a,NULL,status);
    else
      return 1;
  }
  else
    return 1;
}

double ccl_cl_tracer_t_get_cl_contribution(ccl_cl_tracer_t *tr,
					   double ell,double chi,double lk,double a,
					   int *status)
{
  double f_chi=ccl_cl_tracer_t_get_kernel(tr,chi,status);
  double f_ka=ccl_cl_tracer_t_get_transfer(tr,lk,a,status);
  double f_ell=ccl_cl_tracer_t_get_f_ell(tr,ell,status);
  double delta_lka=f_ka*f_chi*f_ell;

  if(*status==0)
    return delta_lka;
  else
    return -1;
}     
