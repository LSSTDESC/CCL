#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include "ccl.h"

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
			 double (*growth)(double),
			 double growth_factor_0,
			 int growth_exponent,
			 ccl_f2d_interp_t interp_type,
			 int *status)
{
  int s2dstatus;
  ccl_f2d_t *f2d=malloc(sizeof(ccl_f2d_t));
  if(f2d==NULL)
    *status = CCL_ERROR_MEMORY;
  
  if(*status==0) {
    f2d->lkmin=lk_arr[0];
    f2d->lkmax=lk_arr[nk-1];
    f2d->amin=a_arr[0];
    f2d->amax=a_arr[na-1];
    f2d->is_factorizable=is_factorizable;
    f2d->extrap_order_lok=extrap_order_lok;
    f2d->extrap_order_hik=extrap_order_hik;
    f2d->extrap_linear_growth=extrap_linear_growth;
    f2d->is_log=is_fka_log;
    f2d->growth=growth;
    f2d->growth_factor_0=growth_factor_0;
    f2d->growth_exponent=growth_exponent;
    f2d->fka=NULL;
    f2d->fk=NULL;
    f2d->fa=NULL;
    if(fabs(f2d->amax-1)>1E-4)
      *status=CCL_ERROR_SPLINE;
  }

  if((extrap_order_lok>2) || (extrap_order_lok<0) || (extrap_order_hik>2) || (extrap_order_hik<0))
    *status=CCL_ERROR_INCONSISTENT;

  if((extrap_linear_growth!=ccl_f2d_cclgrowth) &&
     (extrap_linear_growth!=ccl_f2d_customgrowth) &&
     (extrap_linear_growth!=ccl_f2d_constantgrowth) &&
     (extrap_linear_growth!=ccl_f2d_no_extrapol))
    *status=CCL_ERROR_INCONSISTENT;

  if(*status==0) {
    switch(interp_type) {
    case(ccl_f2d_3):
      if(f2d->is_factorizable) {
	f2d->fk=gsl_spline_alloc(gsl_interp_cspline,nk);
	f2d->fa=gsl_spline_alloc(gsl_interp_cspline,na);
      }
      else
	f2d->fka=gsl_spline2d_alloc(gsl_interp2d_bicubic,nk,na);
      break;
    default:
      f2d->fk=NULL;
      f2d->fa=NULL;
      f2d->fka=NULL;
    }
    if((f2d->fka==NULL) && ((f2d->fk==NULL) || (f2d->fa==NULL)))
      *status = CCL_ERROR_MEMORY;
  }

  if(*status==0) {
    if(f2d->is_factorizable){
      s2dstatus=gsl_spline_init(f2d->fk,lk_arr,fk_arr,nk);
      s2dstatus|=gsl_spline_init(f2d->fa,a_arr,fa_arr,na);
    }
    else {
      s2dstatus=gsl_spline2d_init(f2d->fka,lk_arr,a_arr,fka_arr,nk,na);
    }
    if(s2dstatus)
      *status = CCL_ERROR_SPLINE;
  }

  return f2d;
}

double ccl_f2d_t_eval(ccl_f2d_t *f2d,double lk,double a,void *cosmo,
		      int *status)
{
  double a_ev=a;
  int is_hiz= a<f2d->amin;
  int is_loz= a>f2d->amax;

  if(is_loz) { //Are we above the interpolation range in a?
    *status=CCL_ERROR_SPLINE_EV;
    return NAN;
  }
  else if(is_hiz) { //Are we below the interpolation range in a?
    if(f2d->extrap_linear_growth==ccl_f2d_no_extrapol) {
      *status=CCL_ERROR_SPLINE_EV;
      return NAN;
    }
    a_ev=f2d->amin;
  }

  double fka_pre,fka_post;
  double lk_ev=lk;
  int is_hik= lk>f2d->lkmax;
  int is_lok= lk<f2d->lkmin;
  if(is_hik) //Are we above the interpolation range in k?
    lk_ev=f2d->lkmax;
  else if(is_lok) //Are we below the interpolation range in k?
    lk_ev=f2d->lkmin;

  //Evaluate spline
  int spstatus;
  if(f2d->is_factorizable) {
    double fk,fa;
    spstatus=gsl_spline_eval_e(f2d->fk,lk_ev,NULL,&fk);
    spstatus|=gsl_spline_eval_e(f2d->fa,a_ev,NULL,&fa);
    if(f2d->is_log)
      fka_pre = fk+fa;
    else
      fka_pre = fk*fa;
  }
  else
    spstatus=gsl_spline2d_eval_e(f2d->fka,lk_ev,a_ev,NULL,NULL,&fka_pre);
  if(spstatus) {
    *status=CCL_ERROR_SPLINE_EV;
    return NAN;
  }

  //Now extrapolate in k if needed
  if(is_hik) {
    fka_post=fka_pre;
    if(f2d->extrap_order_hik>0) {
      double pd;
      double dlk=lk-lk_ev;
      if(f2d->is_factorizable)
	spstatus=gsl_spline_eval_deriv_e(f2d->fk,lk_ev,NULL,&pd);
      else
	spstatus=gsl_spline2d_eval_deriv_x_e(f2d->fka,lk_ev,a_ev,NULL,NULL,&pd);
      if(spstatus) {
	*status=CCL_ERROR_SPLINE_EV;
	return NAN;
      }
      fka_post+=pd*dlk;
      if(f2d->extrap_order_hik>1) {
	if(f2d->is_factorizable)
	  spstatus=gsl_spline_eval_deriv2_e(f2d->fk,lk_ev,NULL,&pd);
	else
	  spstatus=gsl_spline2d_eval_deriv_xx_e(f2d->fka,lk_ev,a_ev,NULL,NULL,&pd);
	if(spstatus) {
	  *status=CCL_ERROR_SPLINE_EV;
	  return NAN;
	}
	fka_post+=pd*dlk*dlk*0.5;
      }
    }
  }
  else if(is_lok) {
    fka_post=fka_pre;
    if(f2d->extrap_order_lok>0) {
      double pd;
      double dlk=lk-lk_ev;
      if(f2d->is_factorizable)
	spstatus=gsl_spline_eval_deriv_e(f2d->fk,lk_ev,NULL,&pd);
      else
	spstatus=gsl_spline2d_eval_deriv_x_e(f2d->fka,lk_ev,a_ev,NULL,NULL,&pd);
      if(spstatus) {
	*status=CCL_ERROR_SPLINE_EV;
	return NAN;
      }
      fka_post+=pd*dlk;
      if(f2d->extrap_order_lok>1) {
	if(f2d->is_factorizable)
	  spstatus=gsl_spline_eval_deriv2_e(f2d->fk,lk_ev,NULL,&pd);
	else
	  spstatus=gsl_spline2d_eval_deriv_xx_e(f2d->fka,lk_ev,a_ev,NULL,NULL,&pd);
	if(spstatus) {
	  *status=CCL_ERROR_SPLINE_EV;
	  return NAN;
	}
	fka_post+=pd*dlk*dlk*0.5;
      }
    }
  }
  else
    fka_post=fka_pre;

  //Exponentiate if needed
  if(f2d->is_log)
    fka_post=exp(fka_post);

  //Extrapolate in a if needed
  if(is_hiz) {
    double gz;
    if(f2d->extrap_linear_growth==ccl_f2d_cclgrowth) {//Use CCL's growth function
      ccl_cosmology *csm=(ccl_cosmology *)cosmo;
      gz=ccl_growth_factor(csm,a,status)/ccl_growth_factor(csm,a_ev,status);
    }
    else if(f2d->extrap_linear_growth==ccl_f2d_customgrowth) //Use internal growth function
      gz=f2d->growth(a)/f2d->growth(a_ev);
    else //Use constant growth factor
      gz=f2d->growth_factor_0;
    fka_post*=pow(gz,f2d->growth_exponent);
  }

  return fka_post;
}

void ccl_f2d_t_free(ccl_f2d_t *f2d)
{
  if(f2d!=NULL) {
    if(f2d->fka!=NULL)
      gsl_spline2d_free(f2d->fka);
    if(f2d->fk!=NULL)
      gsl_spline_free(f2d->fk);
    if(f2d->fa!=NULL)
      gsl_spline_free(f2d->fa);
    free(f2d);
  }
}
