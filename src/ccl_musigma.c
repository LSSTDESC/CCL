#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include "ccl.h"


/*
 * Spline the linear power spectrum for mu-Sigma MG cosmologies.
 * @param cosmo Cosmological parameters
 ^ @param psp The linear power spectrum to spline.
 * @param status, integer indicating the status
 */
void ccl_cosmology_spline_linpower_musigma(ccl_cosmology* cosmo, ccl_f2d_t *psp,
                                           int mg_rescale, int* status) {

  if (*status == 0) {
    // If scale-independent mu / Sigma modified gravity is in use
    // and mu ! = 0 : get the unnormalized growth factor in MG and for
    // corresponding GR case, to rescale CLASS power spectrum
    if ((fabs(cosmo->params.mu_0) > 1e-14) && mg_rescale) {
      // Set up another cosmology which is exactly the same as the
      // current one but with mu_0 and Sigma_0=0, for scaling P(k)

      // Get a list of the three neutrino masses already calculated
      size_t na;
      double *aa;
      double norm_pk;
      double mnu_list[3] = {0, 0, 0};
      ccl_parameters params_GR;
      params_GR.m_nu = NULL;
      params_GR.z_mgrowth = NULL;
      params_GR.df_mgrowth = NULL;
      ccl_cosmology * cosmo_GR = NULL;
      double *D_mu = NULL;
      double *D_GR = NULL;

      for (int i=0; i< cosmo->params.N_nu_mass; i=i+1)
        mnu_list[i] = cosmo->params.m_nu[i];

      if (isfinite(cosmo->params.A_s)) {
        norm_pk = cosmo->params.A_s;
      }
      else if (isfinite(cosmo->params.sigma8)) {
        norm_pk = cosmo->params.sigma8;
      }
      else {
        *status = CCL_ERROR_PARAMETERS;
        strcpy(
               cosmo->status_message,
               "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): "
               "neither A_s nor sigma8 defined.\n");
      }

      if (*status == 0) {
        params_GR = ccl_parameters_create(
          cosmo->params.Omega_c, cosmo->params.Omega_b, cosmo->params.Omega_k,
          cosmo->params.Neff, mnu_list, cosmo->params.N_nu_mass,
          cosmo->params.w0, cosmo->params.wa, cosmo->params.h,
          norm_pk, cosmo->params.n_s,
          cosmo->params.bcm_log10Mc, cosmo->params.bcm_etab,
          cosmo->params.bcm_ks, 0., 0., cosmo->params.nz_mgrowth,
          cosmo->params.z_mgrowth, cosmo->params.df_mgrowth, status);

        if (*status) {
          *status = CCL_ERROR_PARAMETERS;
          strcpy(
            cosmo->status_message,
            "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): could not make MG params.\n");
        }
      }

      // Get array of scale factors
      if(*status == 0) {
        if(psp->fa != NULL) {
          na = psp->fa->size;
          aa = psp->fa->x;
        }
        else if(psp->fka != NULL) {
          na = psp->fka->interp_object.ysize;
          aa = psp->fka->yarr;
        }
        else {
          *status = CCL_ERROR_SPLINE;
          strcpy(cosmo->status_message,
                 "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): "
                 "input pk2d has no splines.\n");
        }
      }

      if (*status == 0) {
        cosmo_GR = ccl_cosmology_create(params_GR, cosmo->config);
        D_mu = malloc(na * sizeof(double));
        D_GR = malloc(na * sizeof(double));

        if (cosmo_GR == NULL || D_mu == NULL || D_GR == NULL) {
          *status = CCL_ERROR_MEMORY;
          ccl_cosmology_set_status_message(
            cosmo,
            "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): memory allocation\n");
        }
      }

      if (*status == 0) {
        ccl_cosmology_compute_growth(cosmo_GR, status);

        if (*status) {
          *status = CCL_ERROR_PARAMETERS;
          strcpy(
            cosmo->status_message,
            "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): could not init GR growth.\n");
        }
      }

      if (*status == 0) {
        for (int i=0; i<na; i++) {
          D_mu[i] = ccl_growth_factor_unnorm(cosmo, aa[i], status);
          D_GR[i] = ccl_growth_factor_unnorm(cosmo_GR, aa[i], status);
        }

        if (*status) {
          *status = CCL_ERROR_PARAMETERS;
          strcpy(
            cosmo->status_message,
            "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): could not make MG and GR growth.\n");
        }
      }

      if (*status == 0) {
        size_t nk=0;
        if (psp->fka != NULL)
          nk=psp->fka->interp_object.xsize;

        for (int i=0; i<na; i++) {
          double rescale_factor=D_mu[i]/D_GR[i];
          if(psp->is_log)
            rescale_factor = 2*log(rescale_factor);
          else
            rescale_factor = rescale_factor*rescale_factor;
          if (psp->fa != NULL) {
            if(psp->is_log)
              psp->fa->y[i] += rescale_factor;
            else
              psp->fa->y[i] *= rescale_factor;
          }
          else {
            for(int j=0; j<nk; j++) {
              if(psp->is_log)
                psp->fka->zarr[i*nk+j] += rescale_factor;
              else
                psp->fka->zarr[i*nk+j] *= rescale_factor;
            }
          }
        }

        if(psp->fa != NULL) {
          gsl_spline *fa = gsl_spline_alloc(gsl_interp_cspline,
                                            psp->fa->size);
          if(fa == NULL) {
            *status == CCL_ERROR_MEMORY;
            ccl_cosmology_set_status_message(
                                             cosmo,
            "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): memory allocation\n");
          }
          if(*status==0) {
            int spstatus = gsl_spline_init(fa, psp->fa->x,
                                           psp->fa->y, psp->fa->size);
            if(spstatus) {
              *status == CCL_ERROR_MEMORY;
              ccl_cosmology_set_status_message(
                                               cosmo,
                                               "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): "
                                               "Error initializing spline\n");
            }
          }
          if(*status==0) {
            gsl_spline_free(psp->fa);
            psp->fa=fa;
          }
        }
        else {
          gsl_spline2d *fka = gsl_spline2d_alloc(gsl_interp2d_bicubic,
                                                 psp->fka->interp_object.xsize,
                                                 psp->fka->interp_object.ysize); 
          if(fka == NULL) {
            *status == CCL_ERROR_MEMORY;
            ccl_cosmology_set_status_message(
                                             cosmo,
                                             "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): "
                                             "memory allocation\n");
          }
          if(*status==0) {
            int spstatus = gsl_spline2d_init(fka, psp->fka->xarr,
                                             psp->fka->yarr, psp->fka->zarr,
                                             psp->fka->interp_object.xsize,
                                             psp->fka->interp_object.ysize);
            if(spstatus) {
              *status == CCL_ERROR_MEMORY;
              ccl_cosmology_set_status_message(
                                               cosmo,
                                               "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): "
                                               "Error initializing spline\n");
            }
          }
          if(*status==0) {
            gsl_spline2d_free(psp->fka);
            psp->fka=fka;
          }
        }
      }
      free(D_mu);
      free(D_GR);
      ccl_parameters_free(&params_GR);
      ccl_cosmology_free(cosmo_GR);
    }
  }

  // if desried, renomalize to a given sigma8
  if (*status == 0) {
    if (isfinite(cosmo->params.sigma8) && (!isfinite(cosmo->params.A_s))) {
      size_t na, nk;
      double sigma8, renorm=1;

      cosmo->computed_linear_power = true;
      sigma8 = ccl_sigma8(cosmo, psp, status);
      cosmo->computed_linear_power = false;

      if (*status == 0) {
        renorm = cosmo->params.sigma8/sigma8;
        if (psp->is_log)
          renorm = 2*log(renorm);
        else
          renorm = renorm*renorm;

        if (psp->fa != NULL) {
          na = psp->fa->size;
          for (int i=0; i<na; i++) {
            if (psp->fa != NULL) {
              if (psp->is_log)
                psp->fa->y[i] += renorm;
              else
                psp->fa->y[i] *= renorm;
            }
          }
        }
        else {
          na = psp->fka->interp_object.ysize;
          nk = psp->fka->interp_object.xsize;
          for(int i=0; i<na*nk; i++) {
            if (psp->is_log)
              psp->fka->zarr[i] += renorm;
            else
              psp->fka->zarr[i] *= renorm;
          }
        }
      }

      if (*status == 0) {
        if(psp->fa != NULL) {
          gsl_spline *fa = gsl_spline_alloc(gsl_interp_cspline,
                                            psp->fa->size);
          if(fa == NULL) {
            *status == CCL_ERROR_MEMORY;
            ccl_cosmology_set_status_message(
                                             cosmo,
            "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): memory allocation\n");
          }
          if(*status==0) {
            int spstatus = gsl_spline_init(fa, psp->fa->x,
                                           psp->fa->y, psp->fa->size);
            if(spstatus) {
              *status == CCL_ERROR_MEMORY;
              ccl_cosmology_set_status_message(
                                               cosmo,
                                               "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): "
                                               "Error initializing spline\n");
            }
          }
          if(*status==0) {
            gsl_spline_free(psp->fa);
            psp->fa=fa;
          }
        }
        else {
          gsl_spline2d *fka = gsl_spline2d_alloc(gsl_interp2d_bicubic,
                                                 psp->fka->interp_object.xsize,
                                                 psp->fka->interp_object.ysize); 
          if(fka == NULL) {
            *status == CCL_ERROR_MEMORY;
            ccl_cosmology_set_status_message(
                                             cosmo,
                                             "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): "
                                             "memory allocation\n");
          }
          if(*status==0) {
            int spstatus = gsl_spline2d_init(fka, psp->fka->xarr,
                                             psp->fka->yarr, psp->fka->zarr,
                                             psp->fka->interp_object.xsize,
                                             psp->fka->interp_object.ysize);
            if(spstatus) {
              *status == CCL_ERROR_MEMORY;
              ccl_cosmology_set_status_message(
                                               cosmo,
                                               "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): "
                                               "Error initializing spline\n");
            }
          }
          if(*status==0) {
            gsl_spline2d_free(psp->fka);
            psp->fka=fka;
          }
        }
      }
    }
  }
}
