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
void ccl_cosmology_spline_linpower_musigma(ccl_cosmology* cosmo, ccl_f2d_t *psp, int rescaled_mg_flag, int* status) {
  double kmin, kmax, ndecades, amin, amax, ic, sigma8, log_sigma8;
  int nk, na, s;
  double *lk = NULL, *aa = NULL, *lpk_ln = NULL, *lpk_nl = NULL;
  double norm_pk;
  double *mnu_list = NULL;
  ccl_parameters params_GR;
  params_GR.m_nu = NULL;
  params_GR.z_mgrowth = NULL;
  params_GR.df_mgrowth = NULL;
  ccl_cosmology * cosmo_GR = NULL;
  double *D_mu = NULL;
  double *D_GR = NULL;

  if (*status == 0) {
    //calculations done - now allocate CCL splines
    kmin = 2*exp(psp->lkmin);
    kmax = cosmo->spline_params.K_MAX_SPLINE;
    //Compute nk from number of decades and N_K = # k per decade
    ndecades = log10(kmax) - log10(kmin);
    nk = (int)ceil(ndecades*cosmo->spline_params.N_K);
    amin = cosmo->spline_params.A_SPLINE_MINLOG_PK;
    amax = cosmo->spline_params.A_SPLINE_MAX;
    na = cosmo->spline_params.A_SPLINE_NA_PK+cosmo->spline_params.A_SPLINE_NLOG_PK-1;

    // The lk array is initially k, but will later
    // be overwritten with log(k)
    lk = ccl_log_spacing(kmin, kmax, nk);
    if (lk == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
          cosmo,
          "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): memory allocation\n");
    }
  }

  if (*status == 0) {
    aa = ccl_linlog_spacing(
        amin, cosmo->spline_params.A_SPLINE_MIN_PK,
        amax, cosmo->spline_params.A_SPLINE_NLOG_PK,
        cosmo->spline_params.A_SPLINE_NA_PK);
    if (aa == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): memory allocation\n");
    }
  }

  if (*status == 0) {
    lpk_ln = malloc(nk * na * sizeof(double));
    if (lpk_ln == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): memory allocation\n");
    }
  }

  if (*status == 0) {
    lpk_nl = malloc(nk * na * sizeof(double));
    if(lpk_nl == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): memory allocation\n");
    }
  }

  if (*status == 0) {
    // After this loop lk will contain log(k),
    // lpk_ln will contain log(P_lin), all in Mpc, not Mpc/h units!
    double psout_l;
    s = 0;

    // If scale-independent mu / Sigma modified gravity is in use
    // and mu ! = 0 : get the unnormalized growth factor in MG and for
    // corresponding GR case, to rescale CLASS power spectrum
    if (fabs(cosmo->params.mu_0) > 1e-14) {
      // Set up another cosmology which is exactly the same as the
      // current one but with mu_0 and Sigma_0=0, for scaling P(k)

      // Get a list of the three neutrino masses already calculated
      mnu_list = malloc(3*sizeof(double));
      if (mnu_list == NULL) {
        *status = CCL_ERROR_MEMORY;
        ccl_cosmology_set_status_message(
          cosmo,
          "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): memory allocation\n");
      }

      if (*status == 0) {
        for (int i=0; i< cosmo->params.N_nu_mass; i=i+1) {
          mnu_list[i] = cosmo->params.m_nu[i];
        }
        if (cosmo->params.N_nu_mass < 3) {
          for (int j=cosmo->params.N_nu_mass; j<3; j=j+1) {
            mnu_list[j] = 0.;
          }
        }

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
            "ccl_power.c: ccl_cosmology_spline_linpower_musigma(): neither A_s nor sigma8 defined.\n");
        }
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
        for (int i=0; i<nk; i++) {
          lk[i] = log(lk[i]);
          for (int j = 0; j < na; j++) {
            //The 2D interpolation routines access the function values pk_{k_ia_j} with the following ordering:
            //pk_ij = pk[j*N_k + i]
            //with i = 0,...,N_k-1 and j = 0,...,N_a-1.
            psout_l = ccl_f2d_t_eval(psp, lk[i], aa[j], cosmo, status);
	    if (rescaled_mg_flag == 0) {
            lpk_ln[j*nk+i] = log(psout_l) ;
	    }
	    else {
            lpk_ln[j*nk+i] = log(psout_l) + 2 * log(D_mu[j]) - 2 * log(D_GR[j]);
	    }

          }
        }
      }
    }
    else {
      // This is the normal GR case.
      for (int i=0; i<nk; i++) {
        lk[i] = log(lk[i]);
        for (int j = 0; j<na; j++) {
          //The 2D interpolation routines access the function values pk_{k_ia_j} with the following ordering:
          //pk_ij = pk[j*N_k + i]
          //with i = 0,...,N_k-1 and j = 0,...,N_a-1.
          psout_l = ccl_f2d_t_eval(psp, lk[i], aa[j], cosmo, status);
          lpk_ln[j*nk+i] = log(psout_l);
        }
      }
    }
  }

  if (*status == 0) {
    cosmo->data.p_lin = ccl_f2d_t_new(
      na, aa, nk, lk, lpk_ln, NULL, NULL, 0,
      1, 2, ccl_f2d_cclgrowth, 1, NULL, 0, 2,
      ccl_f2d_3,status);
  }

  // if desried, renomalize to a given sigma8
  if (isfinite(cosmo->params.sigma8) && (!isfinite(cosmo->params.A_s))) {
    if (*status == 0) {
      cosmo->computed_linear_power = true;
      sigma8 = ccl_sigma8(cosmo, status);
      cosmo->computed_linear_power = false;
    }

    if (*status == 0) {
      // Calculate normalization factor using computed value of sigma8, then
      // recompute P(k, a) using this normalization
      log_sigma8 = 2*(log(cosmo->params.sigma8) - log(sigma8));
      for(int j = 0; j<na*nk; j++)
        lpk_ln[j] += log_sigma8;
    }

    if (*status == 0) {
      // Free the previous P(k,a) spline, and allocate a new one to store the
      // properly-normalized P(k,a)
      ccl_f2d_t_free(cosmo->data.p_lin);
      cosmo->data.p_lin = ccl_f2d_t_new(
        na, aa, nk, lk, lpk_ln, NULL, NULL, 0,
        1, 2, ccl_f2d_cclgrowth, 1, NULL, 0, 2,
        ccl_f2d_3,status);
    }
  }

  free(D_mu);
  free(D_GR);
  free(mnu_list);
  ccl_parameters_free(&params_GR);
  ccl_cosmology_free(cosmo_GR);
  free(lk);
  free(aa);
  free(lpk_nl);
  free(lpk_ln);
}
