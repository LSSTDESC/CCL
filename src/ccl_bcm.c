#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

#include "ccl.h"

/* BCM correction */
// See Schneider & Teyssier (2015) for details of the model.
double ccl_bcm_model_fka(ccl_cosmology * cosmo, double k, double a, int *status) {
  double fkz;
  double b0;
  double bfunc, bfunc4;
  double kg;
  double gf,scomp;
  double kh;
  double z;

  z = 1./a - 1.;
  kh = k / cosmo->params.h;
  b0 = 0.105*cosmo->params.bcm_log10Mc - 1.27;
  bfunc = b0 / (1. + pow(z/2.3, 2.5));
  bfunc4 = (1-bfunc) * (1-bfunc) * (1-bfunc) * (1-bfunc);
  kg = 0.7 * bfunc4 * pow(cosmo->params.bcm_etab, -1.6);
  gf = bfunc / (1 + pow(kh/kg, 3.)) + 1. - bfunc; //k in h/Mpc
  scomp = 1 + (kh / cosmo->params.bcm_ks) * (kh / cosmo->params.bcm_ks); //k in h/Mpc
  fkz = gf * scomp;
  return fkz;
}


void ccl_bcm_correct(ccl_cosmology *cosmo, ccl_f2d_t *psp, int *status)
{
  size_t nk, na;
  double *x, *z, *y2d=NULL;

  //Find lk array
  if(psp->fk != NULL) {
    nk = psp->fk->size;
    x = psp->fk->x;
  }
  else {
    nk = psp->fka->interp_object.xsize;
    x = psp->fka->xarr;
  }

  //Find a array
  if(psp->fa != NULL) {
    na = psp->fa->size;
    z = psp->fa->x;
  }
  else {
    na = psp->fka->interp_object.ysize;
    z = psp->fka->yarr;
  }

  //Allocate pka array
  y2d = malloc(nk * na * sizeof(double));
  if (y2d == NULL) {
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
                                     "ccl_bcm.c: ccl_bcm_correct(): "
                                     "memory allocation\n");
  }

  if (*status == 0) {
    for (int j = 0; j<na; j++) {
      for (int i=0; i<nk; i++) {
        if (*status == 0) {
          double pk = ccl_f2d_t_eval(psp, x[i], z[j], cosmo, status);
          double fbcm = ccl_bcm_model_fka(cosmo, exp(x[i]), z[j], status);
          if(psp->is_log)
            y2d[j*nk + i] = log(pk*fbcm);
          else
            y2d[j*nk + i] = pk*fbcm;
        }
      }
    }
  }

  if (*status == 0) {
    gsl_spline2d *fka = gsl_spline2d_alloc(gsl_interp2d_bicubic, nk, na);

    if (fka == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
                                       "ccl_bcm.c: ccl_bcm_correct(): "
                                       "memory allocation\n");
    }
    if(*status == 0) {
      int spstatus = gsl_spline2d_init(fka, x, z, y2d, nk, na);
      if(spstatus) {
        *status = CCL_ERROR_MEMORY;
        ccl_cosmology_set_status_message(cosmo,
                                         "ccl_bcm.c: ccl_bcm_correct(): "
                                         "Error initializing spline\n");
      }
    }
    if(*status == 0) {
      if(psp->fa != NULL)
        gsl_spline_free(psp->fa);
      if(psp->fk != NULL)
        gsl_spline_free(psp->fk);
      if(psp->fka != NULL)
        gsl_spline2d_free(psp->fka);
      psp->fka = fka;
      psp->is_factorizable = 0;
      psp->is_k_constant = 0;
      psp->is_a_constant = 0;
    }
    else
      gsl_spline2d_free(fka);
  }

  free(y2d);
}
