#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include "ccl.h"


/* --------- ROUTINE: ccl_mu_MG ---------
INPUT: cosmology object, scale factor, wavenumber for scale
TASK: Compute mu(a,k) where mu is one of the the parameterizating functions
of modifications to GR in the quasistatic approximation.
*/

double ccl_mu_MG(ccl_cosmology * cosmo, double a, double k, int *status)
{
    double s1_k, s2_k, hnorm;
	// This function can be extended to include other
	// redshift and scale z-dependences for mu in the future
    if (k==0.0) {
        s1_k = cosmo->params.c1_mg;
    }
    else {
      hnorm = ccl_h_over_h0(cosmo, a, status);
	    s2_k = (cosmo->params.lambda_mg*(hnorm*cosmo->params.H0)/k/(ccl_constants.CLIGHT/1000));
	    s1_k = (1.0+cosmo->params.c1_mg*s2_k*s2_k)/(1.0+s2_k*s2_k);
	}
	return cosmo->params.mu_0 * ccl_omega_x(cosmo, a, ccl_species_l_label, status)/cosmo->params.Omega_l*s1_k;
}

/* --------- ROUTINE: ccl_Sig_MG ---------
INPUT: cosmology object, scale factor, wavenumber for scale
TASK: Compute Sigma(a,k) where Sigma is one of the the parameterizating functions
of modifications to GR in the quasistatic approximation.
*/

double ccl_Sig_MG(ccl_cosmology * cosmo, double a, double k, int *status)
{
    double s1_k, s2_k, hnorm;
	// This function can be extended to include other
	// redshift and scale dependences for Sigma in the future.
    if (k==0.0) {
        s1_k = cosmo->params.c2_mg;
    }
    else {
      hnorm = ccl_h_over_h0(cosmo, a, status);
	    s2_k = cosmo->params.lambda_mg*(hnorm*cosmo->params.H0)/k/(ccl_constants.CLIGHT/1000);
        s1_k = (1.0+cosmo->params.c2_mg*s2_k*s2_k)/(1.0+s2_k*s2_k);

	}
	return cosmo->params.sigma_0 * ccl_omega_x(cosmo, a, ccl_species_l_label, status)/cosmo->params.Omega_l*s1_k;
}

/*
 * Spline the linear power spectrum for mu-Sigma MG cosmologies.
 * @param cosmo Cosmological parameters
 ^ @param psp The linear power spectrum to spline.
 * @param status, integer indicating the status
 */
void ccl_rescale_musigma_s8(ccl_cosmology* cosmo, ccl_f2d_t *psp,
                            int mg_rescale, int* status) {

  int do_mg = mg_rescale && (fabs(cosmo->params.mu_0) > 1e-14);
  int do_s8 = isfinite(cosmo->params.sigma8) && (!isfinite(cosmo->params.A_s));

  if (!do_mg && !do_s8)
    return;

  size_t na;
  double *aa;
  double rescale_extra_musig = 1;
  double *rescale_factor = NULL;

  if (*status == 0) {
    // Get array of scale factors
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
      ccl_cosmology_set_status_message(cosmo,
             "ccl_musigma.c: ccl_rescale_musigma_s8(): "
             "input pk2d has no splines.\n");
    }
  }

  // Alloc rescale_factor
  if(*status == 0) {
    rescale_factor = malloc(na * sizeof(double));
    if(rescale_factor == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
             "ccl_musigma.c: ccl_rescale_musigma_s8(): memory allocation\n");
    }
  }

  // set to 1 by default
  if(*status == 0) {
    for(int i=0; i<na; i++)
      rescale_factor[i]=1;
  }

  // If scale-independent mu / Sigma modified gravity is in use
  // and mu ! = 0 : get the unnormalized growth factor in MG and for
  // corresponding GR case, to rescale CLASS power spectrum
  if(do_mg) {
    // Set up another cosmology which is exactly the same as the
    // current one but with mu_0 and Sigma_0=0, for scaling P(k)

    // Get a list of the three neutrino masses already calculated
    double norm_pk;
    double mnu_list[3] = {0, 0, 0};
    ccl_parameters params_GR;
    params_GR.m_nu = NULL;
    params_GR.z_mgrowth = NULL;
    params_GR.df_mgrowth = NULL;
    ccl_cosmology * cosmo_GR = NULL;

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
      ccl_cosmology_set_status_message(cosmo,
             "ccl_musigma.c: ccl_rescale_musigma_s8(): "
             "neither A_s nor sigma8 defined.\n");
    }

    if (*status == 0) {
      params_GR = ccl_parameters_create(
        cosmo->params.Omega_c, cosmo->params.Omega_b, cosmo->params.Omega_k,
        cosmo->params.Neff, mnu_list, cosmo->params.N_nu_mass,
        cosmo->params.w0, cosmo->params.wa, cosmo->params.h,
        norm_pk, cosmo->params.n_s,
        cosmo->params.bcm_log10Mc, cosmo->params.bcm_etab,
        cosmo->params.bcm_ks, 0., 0., 1., 1., 0.,cosmo->params.nz_mgrowth,
        cosmo->params.z_mgrowth, cosmo->params.df_mgrowth, status);

      if (*status) {
        *status = CCL_ERROR_PARAMETERS;
        ccl_cosmology_set_status_message(cosmo,
               "ccl_musigma.c: ccl_rescale_musigma_s8(): "
               "could not make MG params.\n");
      }
    }

    if (*status == 0) {
      cosmo_GR = ccl_cosmology_create(params_GR, cosmo->config);

      if (cosmo_GR == NULL) {
        *status = CCL_ERROR_MEMORY;
        ccl_cosmology_set_status_message(cosmo,
               "ccl_musigma.c: ccl_rescale_musigma_s8(): "
               "error initializing cosmology\n");
      }
    }

    if (*status == 0) {
     ccl_cosmology_compute_growth(cosmo_GR, status);


      if (*status) {
        *status = CCL_ERROR_PARAMETERS;
        ccl_cosmology_set_status_message(cosmo,
               "ccl_musigma.c: ccl_rescale_musigma_s8(): "
               "could not init GR growth.\n");
      }
    }

    // Populate rescale_factor
    if (*status == 0) {
      for (int i=0; i<na; i++) {
        double D_mu = ccl_growth_factor_unnorm(cosmo, aa[i], status);
        double D_GR = ccl_growth_factor_unnorm(cosmo_GR, aa[i], status);
        double renorm = D_mu/D_GR;
        rescale_factor[i] *= renorm*renorm;
      }
      rescale_extra_musig = rescale_factor[na-1];

      if (*status) {
        *status = CCL_ERROR_PARAMETERS;
        ccl_cosmology_set_status_message(cosmo,
               "ccl_musigma.c: ccl_rescale_musigma_s8(): "
               "could not make MG and GR growth.\n");
      }
    }

    ccl_parameters_free(&params_GR);
    ccl_cosmology_free(cosmo_GR);
  }

  if(do_s8) {
    if (*status == 0) {
      double renorm = cosmo->params.sigma8/ccl_sigma8(cosmo, psp, status);
      renorm *= renorm;
      renorm /= rescale_extra_musig;

      for (int i=0; i<na; i++)
        rescale_factor[i] *= renorm;
    }
  }

  if (*status == 0) {
    size_t nk=0;
    if (psp->fka != NULL)
      nk=psp->fka->interp_object.xsize;

    // Rescale
    for (int i=0; i<na; i++) {
      if (psp->fa != NULL) {
        if(psp->is_log)
          psp->fa->y[i] += log(rescale_factor[i]);
        else
          psp->fa->y[i] *= rescale_factor[i];
      }
      else {
        for(int j=0; j<nk; j++) {
          if(psp->is_log)
            psp->fka->zarr[i*nk+j] += log(rescale_factor[i]);
          else
            psp->fka->zarr[i*nk+j] *= rescale_factor[i];
        }
      }
    }

    //Respline
    if(psp->fa != NULL) {
      gsl_spline *fa = gsl_spline_alloc(gsl_interp_cspline,
                                        psp->fa->size);
      if(fa == NULL) {
        *status = CCL_ERROR_MEMORY;
        ccl_cosmology_set_status_message(cosmo,
                                         "ccl_musigma.c: ccl_rescale_musigma_s8(): "
                                         "memory allocation\n");
      }
      if(*status==0) {
        int spstatus = gsl_spline_init(fa, psp->fa->x,
                                       psp->fa->y, psp->fa->size);
        if(spstatus) {
          *status = CCL_ERROR_MEMORY;
          ccl_cosmology_set_status_message(cosmo,
                                           "ccl_musigma.c: ccl_rescale_musigma_s8(): "
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
        *status = CCL_ERROR_MEMORY;
        ccl_cosmology_set_status_message(cosmo,
                                         "ccl_musigma.c: ccl_rescale_musigma_s8(): "
                                         "memory allocation\n");
      }
      if(*status==0) {
        int spstatus = gsl_spline2d_init(fka, psp->fka->xarr,
                                         psp->fka->yarr, psp->fka->zarr,
                                         psp->fka->interp_object.xsize,
                                         psp->fka->interp_object.ysize);
        if(spstatus) {
          *status = CCL_ERROR_MEMORY;
          ccl_cosmology_set_status_message(cosmo,
                                           "ccl_musigma.c: ccl_rescale_musigma_s8(): "
                                           "Error initializing spline\n");
        }
      }
      if(*status==0) {
        gsl_spline2d_free(psp->fka);
        psp->fka=fka;
      }
    }
  }

  free(rescale_factor);
}
