#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include "ccl.h"

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
			 int *status)
{
  int s2dstatus;
  ccl_p2d_t *psp=malloc(sizeof(ccl_p2d_t));
  if(psp==NULL)
    *status = CCL_ERROR_MEMORY;

  if((extrap_order_lok>2) || (extrap_order_lok<0) || (extrap_order_hik>2) || (extrap_order_hik<0))
    *status=CCL_ERROR_INCONSISTENT;

  if((extrap_linear_growth!=ccl_p2d_cclgrowth) &&
     (extrap_linear_growth!=ccl_p2d_customgrowth) &&
     (extrap_linear_growth!=ccl_p2d_constantgrowth) &&
     (extrap_linear_growth!=ccl_p2d_no_extrapol))
    *status=CCL_ERROR_INCONSISTENT;
  
  if(*status==0) {
    psp->lkmin=lk_arr[0];
    psp->lkmax=lk_arr[nk-1];
    psp->amin=a_arr[0];
    psp->amax=a_arr[na-1];
    psp->extrap_order_lok=extrap_order_lok;
    psp->extrap_order_hik=extrap_order_hik;
    psp->extrap_linear_growth=extrap_linear_growth;
    psp->is_log=is_pk_log;
    psp->growth=growth;
    psp->growth_factor_0=growth_factor_0;
    psp->pk=NULL;
    if(fabs(psp->amax-1)>1E-4)
      *status=CCL_ERROR_SPLINE;
  }

  if(*status==0) {
    switch(interp_type) {
    case(ccl_p2d_3):
      psp->pk=gsl_spline2d_alloc(gsl_interp2d_bicubic,nk,na);
      break;
    default:
      psp->pk=NULL;
    }
    if(psp->pk==NULL)
      *status = CCL_ERROR_MEMORY;
  }

  if(*status==0) {
    s2dstatus=gsl_spline2d_init(psp->pk,lk_arr,a_arr,pk_arr,nk,na);
    if(s2dstatus)
      *status = CCL_ERROR_SPLINE;
  }

  return psp;
}

double ccl_p2d_t_eval(ccl_p2d_t *psp,double lk,double a,ccl_cosmology *cosmo,
		      int *status)
{
  double a_ev=a;
  int is_hiz= a<psp->amin;
  int is_loz= a>psp->amax;

  if(is_loz) { //Are we above the interpolation range in a?
    *status=CCL_ERROR_SPLINE_EV;
    return -1;
  }
  else if(is_hiz) { //Are we below the interpolation range in a?
    if(psp->extrap_linear_growth==ccl_p2d_no_extrapol) {
      *status=CCL_ERROR_SPLINE_EV;
      return NAN;
    }
    a_ev=psp->amin;
  }

  double pk_pre,pk_post;
  double lk_ev=lk;
  int is_hik= lk>psp->lkmax;
  int is_lok= lk<psp->lkmin;
  if(is_hik) //Are we above the interpolation range in k?
    lk_ev=psp->lkmax;
  else if(is_lok) //Are we below the interpolation range in k?
    lk_ev=psp->lkmin;

  //Evaluate spline
  int spstatus=gsl_spline2d_eval_e(psp->pk,lk_ev,a_ev,NULL,NULL,&pk_pre);
  if(spstatus) {
    *status=CCL_ERROR_SPLINE_EV;
    return -1;
  }

  //Now extrapolate in k if needed
  if(is_hik) {
    pk_post=pk_pre;
    if(psp->extrap_order_hik>0) {
      double pd;
      double dlk=lk-lk_ev;
      spstatus=gsl_spline2d_eval_deriv_x_e(psp->pk,lk_ev,a_ev,NULL,NULL,&pd);
      if(spstatus) {
	*status=CCL_ERROR_SPLINE_EV;
	return -1;
      }
      pk_post+=pd*dlk;
      if(psp->extrap_order_hik>1) {
	spstatus=gsl_spline2d_eval_deriv_xx_e(psp->pk,lk_ev,a_ev,NULL,NULL,&pd);
	if(spstatus) {
	  *status=CCL_ERROR_SPLINE_EV;
	  return -1;
	}
	pk_post+=pd*dlk*dlk*0.5;
      }
    }
  }
  else if(is_lok) {
    pk_post=pk_pre;
    pk_post=pk_pre;
    if(psp->extrap_order_lok>0) {
      double pd;
      double dlk=lk-lk_ev;
      spstatus=gsl_spline2d_eval_deriv_x_e(psp->pk,lk_ev,a_ev,NULL,NULL,&pd);
      if(spstatus) {
	*status=CCL_ERROR_SPLINE_EV;
	return -1;
      }
      pk_post+=pd*dlk;
      if(psp->extrap_order_lok>1) {
	spstatus=gsl_spline2d_eval_deriv_xx_e(psp->pk,lk_ev,a_ev,NULL,NULL,&pd);
	if(spstatus) {
	  *status=CCL_ERROR_SPLINE_EV;
	  return -1;
	}
	pk_post+=pd*dlk*dlk*0.5;
      }
    }
  }
  else
    pk_post=pk_pre;

  //Exponentiate if needed
  if(psp->is_log)
    pk_post=exp(pk_post);

  //Extrapolate in a if needed
  if(is_hiz) {
    double gz;
    if(psp->extrap_linear_growth==ccl_p2d_cclgrowth) //Use CCL's growth function
      gz=ccl_growth_factor(cosmo,a,status)/ccl_growth_factor(cosmo,a_ev,status);
    else if(psp->extrap_linear_growth==ccl_p2d_customgrowth) //Use internal growth function
      gz=psp->growth(a)/psp->growth(a_ev);
    else //Use constant growth factor
      gz=psp->growth_factor_0;
    pk_post*=gz*gz;
  }

  return pk_post;
}

void ccl_p2d_t_free(ccl_p2d_t *psp)
{
  if(psp!=NULL) {
    if(psp->pk!=NULL)
      gsl_spline2d_free(psp->pk);
    free(psp);
  }
}
