#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include "ccl.h"

ccl_f2d_t *ccl_f2d_t_copy(ccl_f2d_t *f2d_o, int *status)
{
  int s2dstatus=0;
  ccl_f2d_t *f2d = malloc(sizeof(ccl_f2d_t));
  if (f2d == NULL)
    *status = CCL_ERROR_MEMORY;

  if(*status==0) {
    f2d->lkmin = f2d_o->lkmin;
    f2d->lkmax = f2d_o->lkmax;
    f2d->amin = f2d_o->amin;
    f2d->amax = f2d_o->amax;
    f2d->is_factorizable = f2d_o->is_factorizable;
    f2d->is_k_constant = f2d_o->is_k_constant;
    f2d->is_a_constant = f2d_o->is_a_constant;
    f2d->extrap_linear_growth = f2d_o->extrap_linear_growth;
    f2d->extrap_order_lok = f2d_o->extrap_order_lok;
    f2d->extrap_order_hik = f2d_o->extrap_order_hik;
    f2d->is_log = f2d_o->is_log;
    f2d->growth_factor_0 = f2d_o->growth_factor_0;
    f2d->growth_exponent = f2d_o->growth_exponent;

    if(f2d_o->fk != NULL) {
      f2d->fk = gsl_spline_alloc(gsl_interp_cspline,
                                 f2d_o->fk->size);
      if(f2d->fk == NULL)
        *status = CCL_ERROR_MEMORY;

      if(*status==0) {
        s2dstatus |= gsl_spline_init(f2d->fk, f2d_o->fk->x,
                                     f2d_o->fk->y, f2d_o->fk->size);
        if(s2dstatus)
          *status = CCL_ERROR_SPLINE;
      }
    }
    else
      f2d->fk = NULL;
  }

  if(*status==0) {
    if(f2d_o->fa != NULL) {
      f2d->fa = gsl_spline_alloc(gsl_interp_cspline,
                                 f2d_o->fa->size);
      if(f2d->fa == NULL)
        *status = CCL_ERROR_MEMORY;

      if(*status==0) {
        s2dstatus |= gsl_spline_init(f2d->fa, f2d_o->fa->x,
                                     f2d_o->fa->y, f2d_o->fa->size);
        if(s2dstatus)
          *status = CCL_ERROR_SPLINE;
      }
    }
    else
      f2d->fa = NULL;
  }

  if(*status==0) {
    if(f2d_o->fka != NULL) {
      f2d->fka = gsl_spline2d_alloc(gsl_interp2d_bicubic,
                                    f2d_o->fka->interp_object.xsize,
                                    f2d_o->fka->interp_object.ysize);
      if(f2d->fka == NULL)
        *status = CCL_ERROR_MEMORY;

      if(*status==0) {
        s2dstatus |= gsl_spline2d_init(f2d->fka, f2d_o->fka->xarr,
                                       f2d_o->fka->yarr, f2d_o->fka->zarr,
                                       f2d_o->fka->interp_object.xsize,
                                       f2d_o->fka->interp_object.ysize);
        if(s2dstatus)
          *status = CCL_ERROR_SPLINE;
      }
    }
    else
      f2d->fka = NULL;
  }

  return f2d;
}
  
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
                         int *status) {
  int s2dstatus=0;
  ccl_f2d_t *f2d = malloc(sizeof(ccl_f2d_t));
  if (f2d == NULL)
    *status = CCL_ERROR_MEMORY;

  if (*status == 0) {
    is_factorizable = is_factorizable || (a_arr == NULL) || (lk_arr == NULL) || (fka_arr == NULL);
    f2d->is_factorizable = is_factorizable;
    f2d->is_k_constant = ((lk_arr == NULL) || ((fka_arr == NULL) && (fk_arr == NULL)));
    f2d->is_a_constant = ((a_arr == NULL) || ((fka_arr == NULL) && (fa_arr == NULL)));
    f2d->extrap_order_lok = extrap_order_lok;
    f2d->extrap_order_hik = extrap_order_hik;
    f2d->extrap_linear_growth = extrap_linear_growth;
    f2d->is_log = is_fka_log;
    f2d->growth_factor_0 = growth_factor_0;
    f2d->growth_exponent = growth_exponent;
    f2d->fka = NULL;
    f2d->fk = NULL;
    f2d->fa = NULL;

    if (!(f2d->is_k_constant)) { //If it's not constant
      f2d->lkmin = lk_arr[0];
      f2d->lkmax = lk_arr[nk-1];
    }
    if (!(f2d->is_a_constant)) {
      f2d->amin = a_arr[0];
      f2d->amax = a_arr[na-1];
    }
  }

  if ((extrap_order_lok > 2) || (extrap_order_lok < 0) || (extrap_order_hik > 2) || (extrap_order_hik < 0))
    *status = CCL_ERROR_INCONSISTENT;

  if ((extrap_linear_growth != ccl_f2d_cclgrowth) &&
      (extrap_linear_growth != ccl_f2d_constantgrowth) &&
      (extrap_linear_growth != ccl_f2d_no_extrapol))
    *status = CCL_ERROR_INCONSISTENT;

  if(*status == 0) {
    switch(interp_type) {
    case(ccl_f2d_3):
      if (f2d->is_factorizable) {
        // Do not allocate spline if constant
        if(f2d->is_k_constant)
          f2d->fk = NULL;
        else { //Otherwise allocate and check
          f2d->fk = gsl_spline_alloc(gsl_interp_cspline, nk);
          if(f2d->fk == NULL)
            *status = CCL_ERROR_MEMORY;
        }

        // Do not allocate spline if constant
        if (f2d->is_a_constant)
          f2d->fa = NULL;
        else { //Otherwise allocate and check
          f2d->fa = gsl_spline_alloc(gsl_interp_cspline, na);
          if (f2d->fa == NULL)
            *status = CCL_ERROR_MEMORY;
        }
      }
      else {
        // Do not allocate spline if constant
        if ((f2d->is_k_constant) || (f2d->is_a_constant))
          f2d->fka = NULL;
        else { //Otherwise allocate and check
          f2d->fka = gsl_spline2d_alloc(gsl_interp2d_bicubic, nk, na);
          if (f2d->fka == NULL)
            *status = CCL_ERROR_MEMORY;
        }
      }
      break;

    default:
      f2d->fk = NULL;
      f2d->fa = NULL;
      f2d->fka = NULL;
    }
  }

  if (*status == 0) {
    if (f2d->is_factorizable) {
      if (f2d->fk != NULL)
        s2dstatus |= gsl_spline_init(f2d->fk, lk_arr, fk_arr, nk);
      if (f2d->fa != NULL)
        s2dstatus |= gsl_spline_init(f2d->fa, a_arr, fa_arr, na);
    }
    else {
      if (f2d->fka != NULL)
        s2dstatus=gsl_spline2d_init(f2d->fka, lk_arr, a_arr, fka_arr, nk, na);
    }
    if (s2dstatus)
      *status = CCL_ERROR_SPLINE;
  }

  return f2d;
}

double ccl_f2d_t_eval(ccl_f2d_t *f2d,double lk,double a,void *cosmo, int *status) {
  int is_hiz, is_loz;
  double a_ev = a;
  if (f2d->is_a_constant) {
    is_hiz = 0;
    is_loz = 0;
  }
  else {
    is_hiz = a < f2d->amin;
    is_loz = a > f2d->amax;
    if (is_loz) { // Are we above the interpolation range in a?
      if (f2d->extrap_linear_growth == ccl_f2d_no_extrapol) {
        *status=CCL_ERROR_SPLINE_EV;
        return NAN;
      }
      a_ev = f2d->amax;
    }
    else if (is_hiz) { // Are we below the interpolation range in a?
      if (f2d->extrap_linear_growth == ccl_f2d_no_extrapol) {
        *status=CCL_ERROR_SPLINE_EV;
        return NAN;
      }
      a_ev = f2d->amin;
    }
  }

  int is_hik, is_lok;
  double fka_pre, fka_post;
  double lk_ev = lk;
  if (f2d->is_k_constant) {
    is_hik = 0;
    is_lok = 0;
  }
  else {
    is_hik = lk > f2d->lkmax;
    is_lok = lk < f2d->lkmin;
    if (is_hik) // Are we above the interpolation range in k?
      lk_ev = f2d->lkmax;
    else if (is_lok) // Are we below the interpolation range in k?
      lk_ev = f2d->lkmin;
  }

  // Evaluate spline
  int spstatus=0;
  if (f2d->is_factorizable) {
    double fk, fa;
    if (f2d->fk == NULL) {
      if (f2d->is_log)
        fk = 0;
      else
        fk = 1;
    }
    else
      spstatus |= gsl_spline_eval_e(f2d->fk, lk_ev, NULL, &fk);

    if (f2d->fa == NULL) {
      if (f2d->is_log)
        fa = 0;
      else
        fa = 1;
    }
    else
      spstatus |= gsl_spline_eval_e(f2d->fa, a_ev, NULL, &fa);
      if (f2d->is_log)
        fka_pre = fk+fa;
      else
        fka_pre = fk*fa;
  }
  else {
    if (f2d->fka == NULL) {
      if (f2d->is_log)
        fka_pre = 0;
      else
        fka_pre = 1;
    }
    else
      spstatus = gsl_spline2d_eval_e(f2d->fka, lk_ev, a_ev, NULL, NULL, &fka_pre);
  }

  if (spstatus) {
    *status = CCL_ERROR_SPLINE_EV;
    return NAN;
  }

  // Now extrapolate in k if needed
  if (is_hik) {
    fka_post = fka_pre;
    if (f2d->extrap_order_hik > 0) {
      double pd;
      double dlk = lk-lk_ev;
      if (f2d->is_factorizable)
        spstatus = gsl_spline_eval_deriv_e(f2d->fk, lk_ev, NULL, &pd);
      else
        spstatus = gsl_spline2d_eval_deriv_x_e(f2d->fka, lk_ev, a_ev, NULL, NULL, &pd);
      if (spstatus) {
        *status = CCL_ERROR_SPLINE_EV;
        return NAN;
      }
      fka_post += pd*dlk;
      if (f2d->extrap_order_hik > 1) {
        if (f2d->is_factorizable)
          spstatus = gsl_spline_eval_deriv2_e(f2d->fk, lk_ev, NULL, &pd);
        else
          spstatus = gsl_spline2d_eval_deriv_xx_e(f2d->fka, lk_ev, a_ev, NULL, NULL, &pd);
        if (spstatus) {
          *status=CCL_ERROR_SPLINE_EV;
          return NAN;
        }
        fka_post += pd*dlk*dlk*0.5;
      }
    }
  }
  else if (is_lok) {
    fka_post = fka_pre;
    if (f2d->extrap_order_lok > 0) {
      double pd;
      double dlk = lk-lk_ev;
      if (f2d->is_factorizable)
        spstatus = gsl_spline_eval_deriv_e(f2d->fk, lk_ev, NULL, &pd);
      else
        spstatus = gsl_spline2d_eval_deriv_x_e(f2d->fka, lk_ev, a_ev, NULL, NULL, &pd);
      if (spstatus) {
        *status = CCL_ERROR_SPLINE_EV;
        return NAN;
      }
      fka_post += pd*dlk;

      if (f2d->extrap_order_lok > 1) {
        if (f2d->is_factorizable)
          spstatus = gsl_spline_eval_deriv2_e(f2d->fk, lk_ev, NULL, &pd);
        else
          spstatus = gsl_spline2d_eval_deriv_xx_e(f2d->fka, lk_ev, a_ev, NULL, NULL, &pd);
        if (spstatus) {
          *status = CCL_ERROR_SPLINE_EV;
          return NAN;
        }
        fka_post += pd*dlk*dlk*0.5;
      }
    }
  }
  else
    fka_post = fka_pre;

  // Exponentiate if needed
  if (f2d->is_log)
    fka_post = exp(fka_post);

  // Extrapolate in a if needed
  if (is_hiz) {
    double gz;
    if (f2d->extrap_linear_growth == ccl_f2d_cclgrowth) { // Use CCL's growth function
      ccl_cosmology *csm = (ccl_cosmology *)cosmo;
      if (!csm->computed_growth) {
        *status = CCL_ERROR_GROWTH_INIT;
        ccl_cosmology_set_status_message(
          csm,
          "ccl_f2d.c: ccl_f2d_t_eval(): growth factor splines have not been precomputed!");
        return NAN;
      }
      gz = (
        ccl_growth_factor(csm, a, status) /
        ccl_growth_factor(csm, a_ev, status));
    }
    else // Use constant growth factor
      gz = f2d->growth_factor_0;

    fka_post *= pow(gz, f2d->growth_exponent);
  }

  return fka_post;
}

void ccl_f2d_t_free(ccl_f2d_t *f2d)
{
  if(f2d != NULL) {
    if(f2d->fka != NULL)
      gsl_spline2d_free(f2d->fka);
    if(f2d->fk != NULL)
      gsl_spline_free(f2d->fk);
    if(f2d->fa != NULL)
      gsl_spline_free(f2d->fa);
    free(f2d);
  }
}
