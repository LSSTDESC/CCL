#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include "ccl.h"


ccl_a_finder *ccl_a_finder_new(int na, double *a_arr)
{
  if(na<=0)
    return NULL;

  ccl_a_finder *finda=malloc(sizeof(ccl_a_finder));
  if(finda == NULL)
    return NULL;

  finda->ia_last=0;
  finda->amin = a_arr[0];
  finda->amax = a_arr[na-1];
  finda->na = na;
  finda->a_arr = malloc(na*sizeof(double));
  if(finda->a_arr==NULL) {
    free(finda);
    return NULL;
  }

  memcpy(finda->a_arr, a_arr, na*sizeof(double));

  return finda;
}

void ccl_a_finder_free(ccl_a_finder *finda)
{
  if(finda!=NULL) {
    if(finda->na>0)
      free(finda->a_arr);
    free(finda);
  }
}
    
int ccl_find_a_index(ccl_a_finder *finda, double a)
{
  int ia_0;

  if(a>=finda->amax)
    ia_0=finda->na-1;
  else if(a<=finda->amin)
    ia_0=0;
  else {
    int gotit=0;
    ia_0=finda->ia_last;
    while(!gotit) {
      if(ia_0==0) {
        if(a<finda->a_arr[1])
          gotit=1;
        else
          ia_0++;
      }
      else if(ia_0==finda->na-1) {
        if(a>=finda->a_arr[ia_0-1])
          gotit=1;
        ia_0--;
      }
      else {
        if(a<finda->a_arr[ia_0])
          ia_0--;
        else {
          if(a>=finda->a_arr[ia_0+1])
            ia_0++;
          else
            gotit=1;
        }
      }
    }
  }

  finda->ia_last = ia_0;
  return ia_0;
}

ccl_f3d_t *ccl_f3d_t_copy(ccl_f3d_t *f3d_o, int *status)
{
  int ia, s2dstatus=0;
  ccl_f3d_t *f3d = malloc(sizeof(ccl_f3d_t));
  if (f3d == NULL)
    *status = CCL_ERROR_MEMORY;

  if(*status==0) {
    f3d->lkmin = f3d_o->lkmin;
    f3d->lkmax = f3d_o->lkmax;
    f3d->na = f3d_o->na;
    f3d->is_product = f3d_o->is_product;
    f3d->extrap_linear_growth = f3d_o->extrap_linear_growth;
    f3d->extrap_order_lok = f3d_o->extrap_order_lok;
    f3d->extrap_order_hik = f3d_o->extrap_order_hik;
    f3d->is_log = f3d_o->is_log;
    f3d->growth_factor_0 = f3d_o->growth_factor_0;
    f3d->growth_exponent = f3d_o->growth_exponent;

    f3d->a_arr = malloc(f3d->na*sizeof(double));
    if(f3d->a_arr == NULL)
      *status = CCL_ERROR_MEMORY;
  }

  if(*status==0) {
    memcpy(f3d->a_arr, f3d_o->a_arr, f3d->na*sizeof(double));

    if(f3d_o->fka_1 != NULL)
      f3d->fka_1 = ccl_f2d_t_copy(f3d_o->fka_1, status);
    else
      f3d->fka_1 = NULL;
  }

  if(*status==0) {
    if(f3d_o->fka_2 != NULL)
      f3d->fka_2 = ccl_f2d_t_copy(f3d_o->fka_2, status);
    else
      f3d->fka_2 = NULL;
  }

  if(*status==0) {
    if(f3d_o->tkka != NULL) {
      f3d->tkka = malloc(f3d->na*sizeof(gsl_spline2d));
      if (f3d->tkka == NULL)
        *status = CCL_ERROR_MEMORY;

      if(*status == 0) {
        s2dstatus = 0;
        for(ia=0; ia<f3d->na; ia++) {
          f3d->tkka[ia] = gsl_spline2d_alloc(gsl_interp2d_bicubic,
                                             f3d_o->tkka[ia]->interp_object.xsize,
                                             f3d_o->tkka[ia]->interp_object.ysize);
          if(f3d->tkka[ia] == NULL)
            *status = CCL_ERROR_MEMORY;

          if(*status==0) {
            s2dstatus |= gsl_spline2d_init(f3d->tkka[ia],
                                           f3d_o->tkka[ia]->xarr,
                                           f3d_o->tkka[ia]->yarr,
                                           f3d_o->tkka[ia]->zarr,
                                           f3d_o->tkka[ia]->interp_object.xsize,
                                           f3d_o->tkka[ia]->interp_object.ysize);
            if(s2dstatus)
              *status = CCL_ERROR_SPLINE;
          }
        }
      }
    }
    else
      f3d->tkka = NULL;
  }

  return f3d;
}
  
ccl_f3d_t *ccl_f3d_t_new(int na,double *a_arr,
                         int nk,double *lk_arr,
                         double *tkka_arr,
                         double *fka1_arr,
                         double *fka2_arr,
                         int is_product,
                         int extrap_order_lok,
                         int extrap_order_hik,
                         ccl_f2d_extrap_growth_t extrap_linear_growth,
                         int is_tkka_log,
                         double growth_factor_0,
                         int growth_exponent,
                         ccl_f2d_interp_t interp_type,
                         int *status) {
  int ia, s2dstatus;
  ccl_f3d_t *f3d = malloc(sizeof(ccl_f3d_t));
  if (f3d == NULL)
    *status = CCL_ERROR_MEMORY;

  if (*status == 0) {
    is_product = is_product || (tkka_arr == NULL);
    f3d->is_product = is_product;
    f3d->extrap_order_lok = extrap_order_lok;
    f3d->extrap_order_hik = extrap_order_hik;
    f3d->extrap_linear_growth = extrap_linear_growth;
    f3d->is_log = is_tkka_log;
    f3d->growth_factor_0 = growth_factor_0;
    f3d->growth_exponent = growth_exponent;
    f3d->fka_1 = NULL;
    f3d->fka_2 = NULL;
    f3d->tkka = NULL;

    f3d->lkmin = lk_arr[0];
    f3d->lkmax = lk_arr[nk-1];
    f3d->na = na;
    f3d->a_arr = malloc(na*sizeof(double));
    if(f3d->a_arr == NULL)
      *status = CCL_ERROR_MEMORY;
  }

  if (*status == 0)
    memcpy(f3d->a_arr, a_arr, na*sizeof(double));

  if ((extrap_order_lok > 1) || (extrap_order_lok < 0) ||
      (extrap_order_hik > 1) || (extrap_order_hik < 0))
    *status = CCL_ERROR_INCONSISTENT;

  if ((extrap_linear_growth != ccl_f2d_cclgrowth) &&
      (extrap_linear_growth != ccl_f2d_constantgrowth) &&
      (extrap_linear_growth != ccl_f2d_no_extrapol))
    *status = CCL_ERROR_INCONSISTENT;

  if (f3d->is_product) {
    if ((fka1_arr == NULL) || (fka2_arr == NULL))
      *status = CCL_ERROR_INCONSISTENT;
  }
  else {
    if (tkka_arr == NULL)
      *status = CCL_ERROR_INCONSISTENT;
  }
      
  if(*status == 0) {
    if (f3d->is_product) {
      f3d->fka_1 = ccl_f2d_t_new(na, a_arr,
                                 nk, lk_arr,
                                 fka1_arr, NULL, NULL, 0,
                                 extrap_order_lok, extrap_order_hik,
                                 extrap_linear_growth,
                                 is_tkka_log,
                                 growth_factor_0, growth_exponent/2,
                                 interp_type, status);
      f3d->fka_2 = ccl_f2d_t_new(na, a_arr,
                                 nk, lk_arr,
                                 fka2_arr, NULL, NULL, 0,
                                 extrap_order_lok, extrap_order_hik,
                                 extrap_linear_growth,
                                 is_tkka_log,
                                 growth_factor_0, growth_exponent/2,
                                 interp_type, status);
    }
    else {
      switch(interp_type) {
      case(ccl_f2d_3):
        f3d->tkka = malloc(na*sizeof(gsl_spline2d));
        if (f3d->tkka == NULL)
          *status = CCL_ERROR_MEMORY;
        if(*status == 0) {
          for(ia=0; ia<na; ia++) {
            double *tkk = &(tkka_arr[ia*nk*nk]);
            f3d->tkka[ia] = gsl_spline2d_alloc(gsl_interp2d_bicubic, nk, nk);
            if (f3d->tkka[ia] == NULL) {
              *status = CCL_ERROR_MEMORY;
              break;
            }
            s2dstatus = gsl_spline2d_init(f3d->tkka[ia],
                                          lk_arr, lk_arr, tkk, nk, nk);
            if (s2dstatus) {
              *status = CCL_ERROR_SPLINE;
              break;
            }
          }
        }
        break;
      default:
        f3d->tkka = NULL;
      }
    }
  }

  return f3d;
}

double ccl_f3d_t_eval(ccl_f3d_t *f3d,double lk1,double lk2,double a,ccl_a_finder *finda,
                      void *cosmo, int *status) {
  double tkka_post;

  double a_ev = a;
  int is_hiz = a < f3d->a_arr[0];
  int is_loz = a > f3d->a_arr[f3d->na-1];
  if (is_loz) { // Are we above the interpolation range in a?
    if (f3d->extrap_linear_growth == ccl_f2d_no_extrapol) {
      *status=CCL_ERROR_SPLINE_EV;
      return NAN;
    }
    a_ev = f3d->a_arr[f3d->na-1];
  }
  else if (is_hiz) { // Are we below the interpolation range in a?
    if (f3d->extrap_linear_growth == ccl_f2d_no_extrapol) {
      *status=CCL_ERROR_SPLINE_EV;
      return NAN;
    }
    a_ev = f3d->a_arr[0];
  }

  if (f3d->is_product) {
    double fka1 = ccl_f2d_t_eval(f3d->fka_1, lk1, a_ev, cosmo, status);
    double fka2 = ccl_f2d_t_eval(f3d->fka_2, lk2, a_ev, cosmo, status);
    if(*status != 0)
      return NAN;
    tkka_post = fka1*fka2;
  }
  else {
    double lk1_ev = lk1;
    int is_hik1 = lk1 > f3d->lkmax;
    int is_lok1 = lk1 < f3d->lkmin;
    int extrap_k1 = (is_hik1 & (f3d->extrap_order_hik > 0)) ||  (is_lok1 & (f3d->extrap_order_lok > 0));
    if (is_hik1) // Are we above the interpolation range in k?
      lk1_ev = f3d->lkmax;
    else if (is_lok1) // Are we below the interpolation range in k?
      lk1_ev = f3d->lkmin;

    double lk2_ev = lk2;
    int is_hik2 = lk2 > f3d->lkmax;
    int is_lok2 = lk2 < f3d->lkmin;
    int extrap_k2 = (is_hik2 & (f3d->extrap_order_hik > 0)) ||  (is_lok2 & (f3d->extrap_order_lok > 0));
    if (is_hik2) // Are we above the interpolation range in k?
      lk2_ev = f3d->lkmax;
    else if (is_lok2) // Are we below the interpolation range in k?
      lk2_ev = f3d->lkmin;

    int ia = ccl_find_a_index(finda, a_ev);

    if(*status == 0) {
      int spstatus = 0;
      double tkka, dtkka1, dtkka2;
      spstatus |= gsl_spline2d_eval_e(f3d->tkka[ia], lk1_ev, lk2_ev,
                                      NULL, NULL, &tkka);
      if(extrap_k1)
        spstatus |= gsl_spline2d_eval_deriv_x_e(f3d->tkka[ia], lk1_ev, lk2_ev,
                                                NULL, NULL, &dtkka1);
      if(extrap_k2)
        spstatus |= gsl_spline2d_eval_deriv_y_e(f3d->tkka[ia], lk1_ev, lk2_ev,
                                                NULL, NULL, &dtkka2);
      if(ia < f3d->na-1) {
        double tkka_p1, dtkka1_p1, dtkka2_p1;
        double h = (a_ev-f3d->a_arr[ia])/(f3d->a_arr[ia+1]-f3d->a_arr[ia]);

        spstatus |= gsl_spline2d_eval_e(f3d->tkka[ia+1], lk1_ev, lk2_ev,
                                        NULL, NULL, &tkka_p1);
        if(!spstatus)
          tkka = tkka*(1-h) + tkka_p1*h;

        if(extrap_k1) {
          spstatus |= gsl_spline2d_eval_deriv_x_e(f3d->tkka[ia+1], lk1_ev, lk2_ev,
                                                  NULL, NULL, &dtkka1_p1);
          if(!spstatus)
            dtkka1 = dtkka1*(1-h) + dtkka1_p1*h;
        }

        if(extrap_k2) {
          spstatus |= gsl_spline2d_eval_deriv_y_e(f3d->tkka[ia+1], lk1_ev, lk2_ev,
                                                  NULL, NULL, &dtkka2_p1);
          if(!spstatus)
            dtkka2 = dtkka2*(1-h) + dtkka2_p1*h;
        }
      }
      if(spstatus) {
        *status = CCL_ERROR_SPLINE_EV;
        return NAN;
      }

      // Extrapolate if needed
      if(extrap_k1)
        tkka += dtkka1 * (lk1 - lk1_ev);
      if(extrap_k2)
        tkka += dtkka2 * (lk2 - lk2_ev);
      tkka_post = tkka;
    }

    // Exponentiate if needed
    if (f3d->is_log)
      tkka_post = exp(tkka_post);
  }

  // Extrapolate in a if needed
  if (is_hiz) {
    double gz;
    if (f3d->extrap_linear_growth == ccl_f2d_cclgrowth) { // Use CCL's growth function
      ccl_cosmology *csm = (ccl_cosmology *)cosmo;
      if (!csm->computed_growth) {
        *status = CCL_ERROR_GROWTH_INIT;
        ccl_cosmology_set_status_message(
          csm,
          "ccl_f3d.c: ccl_f3d_t_eval(): growth factor splines have not been precomputed!");
        return NAN;
      }
      gz = (
        ccl_growth_factor(csm, a, status) /
        ccl_growth_factor(csm, a_ev, status));
    }
    else // Use constant growth factor
      gz = f3d->growth_factor_0;

    tkka_post *= pow(gz, f3d->growth_exponent);
  }

  return tkka_post;
}

void ccl_f3d_t_free(ccl_f3d_t *f3d)
{
  if(f3d != NULL) {
    if(f3d->fka_1 != NULL)
      ccl_f2d_t_free(f3d->fka_1);
    if(f3d->fka_2 != NULL)
      ccl_f2d_t_free(f3d->fka_2);
    if(f3d->tkka != NULL) {
      int ia;
      for(ia=0; ia<f3d->na; ia++)
        gsl_spline2d_free(f3d->tkka[ia]);
      free(f3d->tkka);
    }
    if(f3d->na > 0)
      free(f3d->a_arr);
    free(f3d);
  }
}

ccl_a_finder *ccl_a_finder_new_from_f3d(ccl_f3d_t *f3d)
{
  return ccl_a_finder_new(f3d->na, f3d->a_arr);
}
