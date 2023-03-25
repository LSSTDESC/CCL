#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>

#include "ccl.h"

/*
 * We use the remapping procedure in https://arxiv.org/pdf/1601.07230.pdf
 * to map w0-wa models onto wa == 0 models.
 * The functions below implement this procedure.
*/

// I snuck a private function from background.c into here. :P
// The default routines create a full spline but we just need one value.
// This may seem like over optimizing, but in testing initializing halofit
// is an order of magnitude or more slower if we don't do this.
void compute_chi(double a, ccl_cosmology *cosmo, double * chi, int * stat);

static double zdrag_eh(ccl_parameters *params) {
  // eqn 4 of Eisenstein & Hu 1998
  double OMh2 = (params->Omega_c + params->Omega_b) * params->h * params->h;
  double OBh2 = params->Omega_b * params->h * params->h;
  double b1 = 0.313 * pow(OMh2, -0.419) * (1 + 0.607*pow(OMh2, 0.674));
  double b2 = 0.238 * pow(OMh2, 0.223);
  return 1291 * pow(OMh2, 0.251) * (1 + b1*pow(OBh2, b2)) / (1 + 0.659*pow(OMh2, 0.828));
}

struct hf_model_match_data {
  double chi_drag;
  double a;
  ccl_cosmology *cosmo;
  int *status;
};

static ccl_cosmology *create_w0eff_cosmo(double w0eff, ccl_cosmology *cosmo, int *status) {
  // create a cosmology with the same parameters as the input except w0-wa. Instead
  // the cosmology is created with w0 = w0eff.
  ccl_parameters params_w0eff;
  double mnu[3];
  int i;

  for(i=0; i<3; ++i)
    mnu[i] = 0;
  for(i=0; i<cosmo->params.N_nu_mass; ++i)
    mnu[i] = cosmo->params.m_nu[i];

  params_w0eff = ccl_parameters_create(
    cosmo->params.Omega_c, cosmo->params.Omega_b, cosmo->params.Omega_k,
    cosmo->params.Neff, mnu, cosmo->params.N_nu_mass,
    w0eff, 0, cosmo->params.h, cosmo->params.A_s, cosmo->params.sigma8,
    cosmo->params.n_s, cosmo->params.T_CMB, cosmo->params.Omega_g, cosmo->params.T_ncdm,
    cosmo->params.bcm_log10Mc, cosmo->params.bcm_etab,
    cosmo->params.bcm_ks, cosmo->params.mu_0, cosmo->params.sigma_0,
    cosmo->params.c1_mg, cosmo->params.c2_mg, cosmo->params.lambda_mg,
    cosmo->params.nz_mgrowth,
    cosmo->params.z_mgrowth, cosmo->params.df_mgrowth, status);

    if(*status != 0)
      return NULL;

    return ccl_cosmology_create(params_w0eff, cosmo->config);
}

static double w0eff_func(double w0eff, void *p) {
  // function used to compare the distance to the CMB in a test cosmology to
  // to the value in the original cosmology
  // returns chi_eff - chi
  struct hf_model_match_data *hfd = (struct hf_model_match_data*)p;
  ccl_cosmology *cosmo_w0eff = NULL;
  double chi_drag_w0eff, tmp, zdrag_w0eff;

  // make the equivalent cosmology
  cosmo_w0eff = create_w0eff_cosmo(w0eff, hfd->cosmo, hfd->status);
  if (cosmo_w0eff == NULL) {
    *(hfd->status) = CCL_ERROR_MEMORY;
    return NAN;
  }

  // get the comoving distance to zdrag
  zdrag_w0eff = zdrag_eh(&(cosmo_w0eff->params));
  compute_chi(1.0 / (1.0 + zdrag_w0eff), cosmo_w0eff, &tmp, hfd->status);
  chi_drag_w0eff = tmp;
  compute_chi(hfd->a, cosmo_w0eff, &tmp, hfd->status);
  chi_drag_w0eff -= tmp;
  if (*(hfd->status) != 0) {
    return NAN;
  }

  ccl_parameters_free(&(cosmo_w0eff->params));
  ccl_cosmology_free(cosmo_w0eff);
  return chi_drag_w0eff - hfd->chi_drag;
}

static double get_w0eff(double a, struct hf_model_match_data data) {
  // For a given input w0-wa cosmology, this function solves for the value of
  // w0eff such that the comoving distance from a to the CMB in a cosmology
  // with the same parameters, but with w0, wa = w0eff, 0, is the same as the
  // original cosmology.
  double w0eff, w0eff_low = -2.0, w0eff_high = -0.35;
  double flow, fhigh;
  int itr, max_itr = 1000, gsl_status;
  const gsl_root_fsolver_type *T;
  gsl_root_fsolver *s;
  gsl_function F;

  data.a = a;
  data.chi_drag = ccl_comoving_radial_distance(data.cosmo, 1.0 / (1.0 + zdrag_eh(&(data.cosmo->params))), data.status);
  data.chi_drag -= ccl_comoving_radial_distance(data.cosmo, a, data.status);
  if(*(data.status) != 0) {
    ccl_cosmology_set_status_message(
      data.cosmo,
      "ccl_halofit.c: get_w0eff(): "
      "could not compute chi_drag for cosmology\n");
    return NAN;
  }

  F.function = &w0eff_func;
  F.params = &data;

  // we have to bound the root, otherwise return -1
  // we will fiil in any -1's in the calling routine
  flow = w0eff_func(w0eff_low, &data);
  fhigh = w0eff_func(w0eff_high, &data);
  if (flow * fhigh > 0) {
    return -1;
  }

  T = gsl_root_fsolver_brent;
  s = gsl_root_fsolver_alloc(T);
  if (s == NULL) {
    *(data.status) = CCL_ERROR_MEMORY;
  }
  else {
    gsl_root_fsolver_set(s, &F, w0eff_low, w0eff_high);

    itr = 0;
    do {
      itr++;
      gsl_status = gsl_root_fsolver_iterate(s);
      if (gsl_status == GSL_EBADFUNC)
        break;

      w0eff = gsl_root_fsolver_root(s);
      w0eff_low = gsl_root_fsolver_x_lower(s);
      w0eff_high = gsl_root_fsolver_x_upper(s);

      gsl_status = gsl_root_test_interval(
        w0eff_low, w0eff_high,
        1e-6,
        1e-6);
    } while (gsl_status == GSL_CONTINUE && itr < max_itr);

    gsl_root_fsolver_free(s);

    if (gsl_status != GSL_SUCCESS || itr >= max_itr) {
      ccl_raise_gsl_warning(
        gsl_status, "ccl_halofit.c: get_w0eff(): error in root finding for the halofit matching cosmology\n");
      *(data.status) |= gsl_status;
    }
  }

  return w0eff;
}

/* helper data and functions for integrals

 the integral is

 \int dlnk \Delta^{2}(k) \exp(-k^{2}R^{2})

 for halofit, we also need the first and second derivatives of this
 integral
*/
struct hf_int_data {
  double r;
  double r2;
  double a;
  ccl_cosmology *cosmo;
  ccl_f2d_t *plin;
  int *status;
  gsl_integration_cquad_workspace *workspace;
};

static double gauss_norm_int_func(double lnk, void *p) {
  struct hf_int_data *hfd = (struct hf_int_data*)p;
  double k = exp(lnk);
  double k2 = k*k;

  return (
    ccl_f2d_t_eval(hfd->plin, lnk, hfd->a, hfd->cosmo, hfd->status) *
    k*k2/2.0/M_PI/M_PI *
    exp(-k2 * (hfd->r2)));
}

static double onederiv_gauss_norm_int_func(double lnk, void *p) {
  struct hf_int_data *hfd = (struct hf_int_data*)p;
  double k = exp(lnk);
  double k2 = k*k;

  return (
    ccl_f2d_t_eval(hfd->plin, lnk, hfd->a, hfd->cosmo, hfd->status) *
    k*k2/2.0/M_PI/M_PI *
    exp(-k2 * (hfd->r2)) *
    (-k2 * 2.0 * (hfd->r)));
}

static double twoderiv_gauss_norm_int_func(double lnk, void *p) {
  struct hf_int_data *hfd = (struct hf_int_data*)p;
  double k = exp(lnk);
  double k2 = k*k;

  return (
    ccl_f2d_t_eval(hfd->plin, lnk, hfd->a, hfd->cosmo, hfd->status) *
    k*k2/2.0/M_PI/M_PI *
    exp(-k2 * (hfd->r2)) *
    (-2.0*k2 + 4.0*k2*k2 * (hfd->r2)));
}

// function whose root is \sigma^2{rsigma, a} = 1
static double rsigma_func(double rsigma, void *p) {
  struct hf_int_data *hfd = (struct hf_int_data*)p;
  double result, lnkmin, lnkmax;
  gsl_function F;
  int gsl_status;

  lnkmin = hfd->plin->lkmin;
  lnkmax = fmax(hfd->plin->lkmax, log(30/rsigma));
  hfd->r = rsigma;
  hfd->r2 = rsigma * rsigma;
  F.function = &gauss_norm_int_func;
  F.params = (void *)hfd;
  gsl_status = gsl_integration_cquad(
    &F, lnkmin, lnkmax,
    0.0, hfd->cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL,
    hfd->workspace, &result, NULL, NULL);

  if (gsl_status != GSL_SUCCESS) {
    ccl_raise_gsl_warning(
      gsl_status,
      "ccl_halofit.c: rsigma_func(): error in integration "
      "for finding the halofit non-linear scale\n");
    *(hfd->status) |= gsl_status;
  }

  return result - 1.0;
}

static double lnrsigma_func(double lnrsigma, void *p) {
  return rsigma_func(exp(lnrsigma), p);
}

static double get_rsigma(double a, struct hf_int_data data) {
  double rsigma, rlow = log(1e-64), rhigh = log(1e16);
  double flow, fhigh;
  int itr, max_itr = 1000, gsl_status;
  const gsl_root_fsolver_type *T;
  gsl_root_fsolver *s;
  gsl_function F;

  data.a = a;
  F.function = &lnrsigma_func;
  F.params = &data;

  // we have to bound the root, otherwise return -1
  // we will fiil in any -1's in the calling routine
  flow = lnrsigma_func(rlow, &data);
  fhigh = lnrsigma_func(rhigh, &data);
  if (flow * fhigh > 0) {
    return -1;
  }

  T = gsl_root_fsolver_brent;
  s = gsl_root_fsolver_alloc(T);
  if (s == NULL) {
    *(data.status) = CCL_ERROR_MEMORY;
  }
  else {
    gsl_root_fsolver_set(s, &F, rlow, rhigh);

    itr = 0;
    do {
      itr++;
      gsl_status = gsl_root_fsolver_iterate(s);
      if (gsl_status == GSL_EBADFUNC)
        break;

      rsigma = gsl_root_fsolver_root(s);
      rlow = gsl_root_fsolver_x_lower(s);
      rhigh = gsl_root_fsolver_x_upper(s);

      gsl_status = gsl_root_test_interval(
        rlow, rhigh,
        data.cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL,
        data.cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL);
    } while (gsl_status == GSL_CONTINUE && itr < max_itr);

    gsl_root_fsolver_free(s);

    if (gsl_status != GSL_SUCCESS || itr >= max_itr) {
      ccl_raise_gsl_warning(
        gsl_status, "ccl_halofit.c: get_rsigma(): error in root finding for the halofit non-linear scale\n");
      *(data.status) |= gsl_status;
    }
  }
  rsigma = exp(rsigma);

  return rsigma;
}

/*
 * Allocate a new struct for storing halofit data
 * @param cosmo Cosmological data
 * @param int, status of computations
 */
halofit_struct* ccl_halofit_struct_new(ccl_cosmology *cosmo,
                                       ccl_f2d_t *plin, int *status) {
  size_t n_a;
  int i, gsl_status;
  double lnkmin, lnkmax;
  double *a_vec = NULL;
  double *vals = NULL;
  double *vals_om = NULL;
  double *vals_de = NULL;
  halofit_struct *hf = NULL;
  struct hf_int_data data;
  gsl_function F;
  gsl_integration_cquad_workspace *workspace = NULL;
  double result;
  double sigma2, rsigma, dsigma2drsigma;
  struct hf_model_match_data data_w0eff;
  ccl_cosmology *cosmo_w0eff = NULL;

  // compute spline point locations and integral bounds
  // note that the spline point locations in `a` determine a radius by
  // solving sigma2(R, a) = 1
  // it is this radius that is needed for the subsequent splines of
  // the derivatives.
  lnkmin = plin->lkmin;
  lnkmax = plin->lkmax;
  if(plin->fa != NULL) {
    n_a = plin->fa->size;
    a_vec = plin->fa->x;
  }
  else if(plin->fka != NULL) {
    n_a = plin->fka->interp_object.ysize;
    a_vec = plin->fka->yarr;
  }
  else {
    *status = CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_halofit.c: ccl_halofit_struct_new(): "
           "input pk2d has no splines.\n");
  }

  if(*status == 0) {
    ///////////////////////////////////////////////////////
    // memory allocation
    hf = (halofit_struct*)malloc(sizeof(halofit_struct));
    if (hf == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
                                       cosmo,
                                       "ccl_halofit.c: ccl_halofit_struct_new(): "
                                       "memory could not be allocated for halofit_struct\n");
    } else {
      hf->rsigma = NULL;
      hf->sigma2 = NULL;
      hf->n_eff = NULL;
      hf->C = NULL;
      hf->weff = NULL;
      hf->omeff = NULL;
      hf->deeff = NULL;
    }
  }

  if (*status == 0) {
    workspace = gsl_integration_cquad_workspace_alloc(cosmo->gsl_params.N_ITERATION);
    if (workspace == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): "
        "memory could not be allocated for cquad workspace\n");
    }
  }

  if (*status == 0) {
    vals = (double*)malloc(sizeof(double) * n_a);
    if (vals == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): "
        "memory could not be allocated for results vector\n");
    }
  }

  if (*status == 0) {
    vals_om = (double*)malloc(sizeof(double) * n_a);
    if (vals_om == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): "
        "memory could not be allocated for OmegaM results vector\n");
    }
  }

  if (*status == 0) {
    vals_de = (double*)malloc(sizeof(double) * n_a);
    if (vals_de == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): "
        "memory could not be allocated for OmegaDE results vector\n");
    }
  }

  ////////////////////////////////////////////////////////
  // if wa != 0, then we need to find an equivalent
  // cosmology with wa = 0
  data_w0eff.cosmo = cosmo;
  data_w0eff.status = status;

  if (*status == 0) {
    if (cosmo->params.wa != 0) {
      for(i=0; i<n_a; ++i) {
        vals[i] = get_w0eff(a_vec[i], data_w0eff);
        if (*status != 0) {
          *status = CCL_ERROR_ROOT;
          ccl_cosmology_set_status_message(
            cosmo,
            "ccl_halofit.c: ccl_halofit_struct_new(): "
            "could not solve for effective value of w0 for w0-wa cosmology\n");
          break;
        }

        // now get omeff and deff
        cosmo_w0eff = create_w0eff_cosmo(vals[i], cosmo, status);
        if (cosmo_w0eff == NULL) {
          *status = CCL_ERROR_MEMORY;
          ccl_cosmology_set_status_message(
            cosmo,
            "ccl_halofit.c: ccl_halofit_struct_new(): "
            "could not allocat memory for effective w0 for w0-wa cosmology\n");
          break;
        }

        vals_om[i] = ccl_omega_x(cosmo_w0eff, a_vec[i], ccl_species_m_label, status) +
          ccl_omega_x(cosmo_w0eff, a_vec[i], ccl_species_nu_label, status);
        vals_de[i] = ccl_omega_x(cosmo_w0eff, a_vec[i], ccl_species_l_label, status);

        ccl_parameters_free(&(cosmo_w0eff->params));
        ccl_cosmology_free(cosmo_w0eff);
        cosmo_w0eff = NULL;

        if (*status != 0) {
          ccl_cosmology_set_status_message(
            cosmo,
            "ccl_halofit.c: ccl_halofit_struct_new(): "
            "could not compute OmegaM and OmegaDE for cosmology\n");
          break;
        }
      }
    } else {
      for(i=0; i<n_a; ++i) {
        vals[i] = cosmo->params.w0;
        vals_om[i] = ccl_omega_x(cosmo, a_vec[i], ccl_species_m_label, status) +
          ccl_omega_x(cosmo, a_vec[i], ccl_species_nu_label, status);
        vals_de[i] = ccl_omega_x(cosmo, a_vec[i], ccl_species_l_label, status);
        if (*status != 0) {
          ccl_cosmology_set_status_message(
            cosmo,
            "ccl_halofit.c: ccl_halofit_struct_new(): "
            "could not compute OmegaM and OmegaDE for cosmology\n");
          break;
        }
      }
    }
  }

  // spline the weff values
  if (*status == 0) {
    hf->weff = gsl_spline_alloc(gsl_interp_akima, n_a);
    if (hf->weff == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): "
        "memory could not be allocated for weff spline\n");
    }
  }

  if (*status == 0) {
    gsl_status = gsl_spline_init(hf->weff, a_vec, vals, n_a);
    if (gsl_status != GSL_SUCCESS) {
      *status = CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): could not build weff spline\n");
    }
  }

  // spline the omeff values
  if (*status == 0) {
    hf->omeff = gsl_spline_alloc(gsl_interp_akima, n_a);
    if (hf->omeff == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): "
        "memory could not be allocated for omeff spline\n");
    }
  }

  if (*status == 0) {
    gsl_status = gsl_spline_init(hf->omeff, a_vec, vals_om, n_a);
    if (gsl_status != GSL_SUCCESS) {
      *status = CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): could not build omeff spline\n");
    }
  }

  // spline the deeff values
  if (*status == 0) {
    hf->deeff = gsl_spline_alloc(gsl_interp_akima, n_a);
    if (hf->deeff == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): "
        "memory could not be allocated for deeff spline\n");
    }
  }

  if (*status == 0) {
    gsl_status = gsl_spline_init(hf->deeff, a_vec, vals_de, n_a);
    if (gsl_status != GSL_SUCCESS) {
      *status = CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): could not build deeff spline\n");
    }
  }

  ///////////////////////////////////////////////////////
  // find the nonlinear scale at each scale factor
  if (*status == 0) {
    // setup for integrations
    data.status = status;
    data.cosmo = cosmo;
    data.plin = plin;
    data.workspace = workspace;

    for (i=0; i<n_a; ++i) {
      vals[i] = get_rsigma(a_vec[i], data);
      if ((*status != 0) || (vals[i] <= 0)) {
        *status = CCL_ERROR_ROOT;
        ccl_cosmology_set_status_message(
          cosmo,
          "ccl_halofit.c: ccl_halofit_struct_new(): "
          "could not solve for non-linear scale for halofit at scale factor %f\n", a_vec[i]);
        break;
      }
    }
  }

  // spline the non-linear scales
  if (*status == 0) {
    hf->rsigma = gsl_spline_alloc(gsl_interp_akima, n_a);
    if (hf->rsigma == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): "
        "memory could not be allocated for Rsigma spline\n");
    }
  }

  if (*status == 0) {
    gsl_status = gsl_spline_init(hf->rsigma, a_vec, vals, n_a);
    if (gsl_status != GSL_SUCCESS) {
      *status = CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): could not build Rsigma spline\n");
    }
  }

  ///////////////////////////////////////////////////////
  // now compute sigma2(R) at each spline point
  // and spline that
  // this should be close to 1 OFC, but better to use the exact value
  if (*status == 0) {
    for (i=0; i<n_a; ++i) {
      data.a = a_vec[i];
      vals[i] = rsigma_func(gsl_spline_eval(hf->rsigma, a_vec[i], NULL), &data) + 1;
      if (*status != 0) {
        ccl_cosmology_set_status_message(
          cosmo,
          "ccl_halofit.c: ccl_halofit_struct_new(): could not eval "
          "points for sigma2(R) spline\n");
        break;
      }
    }
  }

  if (*status == 0) {
    hf->sigma2 = gsl_spline_alloc(gsl_interp_akima, n_a);
    if (hf->sigma2 == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): "
        "memory could not be allocated for sigma2(R) spline\n");
    }
  }

  if (*status == 0) {
    gsl_status = gsl_spline_init(hf->sigma2, a_vec, vals, n_a);
    if (gsl_status != GSL_SUCCESS) {
      *status = CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): could not build sigma2(R) spline\n");
    }
  }

  ///////////////////////////////////////////////////////
  // now compute the effective spectral index
  if (*status == 0) {
    F.function = &onederiv_gauss_norm_int_func;
    F.params = &data;

    for (i=0; i<n_a; ++i) {
      rsigma = gsl_spline_eval(hf->rsigma, a_vec[i], NULL);
      sigma2 = gsl_spline_eval(hf->sigma2, a_vec[i], NULL);

      data.a = a_vec[i];
      data.r = rsigma;
      data.r2 = rsigma * rsigma;

      gsl_status = gsl_integration_cquad(
        &F,
        lnkmin,
        fmax(lnkmax, log(30/rsigma)),
        0.0, cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL,
        workspace, &result, NULL, NULL);

      if (gsl_status != GSL_SUCCESS) {
        *status = CCL_ERROR_INTEG;
        ccl_raise_gsl_warning(
          gsl_status,
          "ccl_power.c: ccl_halofit_struct_new(): could not eval "
          "points for n_eff spline\n");
        ccl_cosmology_set_status_message(
          cosmo,
          "ccl_halofit.c: ccl_halofit_struct_new(): could not eval "
          "points for n_eff spline\n");
        break;
      }

      // this is n_eff but expressed in terms of linear derivs
      // see eqn A5 of Takahashi et al.
      vals[i] = -rsigma / sigma2 * result - 3.0;
    }
  }

  if (*status == 0) {
    hf->n_eff = gsl_spline_alloc(gsl_interp_akima, n_a);
    if (hf->n_eff == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): "
        "memory could not be allocated for n_eff spline\n");
    }
  }

  if (*status == 0) {
    gsl_status = gsl_spline_init(hf->n_eff, a_vec, vals, n_a);
    if (gsl_status != GSL_SUCCESS) {
      *status = CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): could not build n_eff spline\n");
    }
  }

  ///////////////////////////////////////////////////////
  // now compute the curvature C
  if (*status == 0) {
    F.function = &twoderiv_gauss_norm_int_func;
    F.params = &data;

    for (i=0; i<n_a; ++i) {
      rsigma = gsl_spline_eval(hf->rsigma, a_vec[i], NULL);
      sigma2 = gsl_spline_eval(hf->sigma2, a_vec[i], NULL);

      // we need to solve for the deriv we need from n_eff here
      dsigma2drsigma = gsl_spline_eval(hf->n_eff, a_vec[i], NULL);
      dsigma2drsigma = (dsigma2drsigma + 3.0) / (-rsigma / sigma2);

      data.a = a_vec[i];
      data.r = rsigma;
      data.r2 = rsigma * rsigma;

      gsl_status = gsl_integration_cquad(
        &F,
        lnkmin,
        fmax(lnkmax, log(30/rsigma)),
        0.0, cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL,
        workspace, &result, NULL, NULL);

      if (gsl_status != GSL_SUCCESS) {
        *status = CCL_ERROR_INTEG;
        ccl_raise_gsl_warning(
          gsl_status,
          "ccl_power.c: ccl_halofit_struct_new(): could not eval "
          "points for C spline\n");
        ccl_cosmology_set_status_message(
          cosmo,
          "ccl_halofit.c: ccl_halofit_struct_new(): could not eval "
          "points for C spline\n");
        break;
      }

      // this is C but expressed in terms of linear derivs
      // see eqn A5 of Takahashi et al.
      vals[i] = (
        -1.0 * (
          result * rsigma * rsigma / sigma2 +
          dsigma2drsigma * rsigma / sigma2 -
          dsigma2drsigma * dsigma2drsigma * rsigma * rsigma / sigma2 / sigma2));
    }
  }

  if (*status == 0) {
    hf->C = gsl_spline_alloc(gsl_interp_akima, n_a);
    if (hf->C == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): "
        "memory could not be allocated for C spline\n");
    }
  }

  if (*status == 0) {
    gsl_status = gsl_spline_init(hf->C, a_vec, vals, n_a);
    if (gsl_status != GSL_SUCCESS) {
      *status = CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): could not build C spline\n");
    }
  }

  // free stuff on the way out
  // if the status is non-zero, then we should free any data already
  // accumulated
  if (*status != 0) {
    ccl_halofit_struct_free(hf);
    hf = NULL;
  }
  gsl_integration_cquad_workspace_free(workspace);
  free(vals);
  free(vals_om);
  free(vals_de);
  if (cosmo_w0eff != NULL) {
    ccl_parameters_free(&(cosmo_w0eff->params));
    ccl_cosmology_free(cosmo_w0eff);
  }

  return hf;
}

/*
 * Free a halofit struct
 * @param hf, pointer to halofit struct to free
 */
void ccl_halofit_struct_free(halofit_struct *hf) {
  if (hf != NULL) {
    if (hf->rsigma != NULL)
      gsl_spline_free(hf->rsigma);

    if (hf->sigma2 != NULL)
      gsl_spline_free(hf->sigma2);

    if (hf->n_eff != NULL)
      gsl_spline_free(hf->n_eff);

    if (hf->C != NULL)
      gsl_spline_free(hf->C);

    if (hf->weff != NULL)
      gsl_spline_free(hf->weff);

    if (hf->omeff != NULL)
      gsl_spline_free(hf->omeff);

    if (hf->deeff != NULL)
      gsl_spline_free(hf->deeff);

    free(hf);
  }
}

/**
 * Computes the halofit non-linear power spectrum
 * @param cosmo: cosmology object containing parameters
 * @param lk: natural logarithm of wavenumber in units of Mpc^{-1}
 * @param a: scale factor normalised to a=1 today
 * @param status: Status flag: 0 if there are no errors, non-zero otherwise
 * @param hf: halofit splines for evaluating the power spectrum
 * @return halofit_matter_power: halofit power spectrum, P(k), units of Mpc^{3}
 */
double ccl_halofit_power(ccl_cosmology *cosmo, ccl_f2d_t *plin,
                         double lk, double a, halofit_struct *hf, int *status) {
  double rsigma, neff, C;
  double ksigma, weffa, omegaMz, omegaDEwz, kh;
  double PkL, PkNL, f1, f2, f3, an, bn, cn, gamman, alphan, betan, nun, mun, y, fy;
  double DeltakL, DeltakL_tilde, DeltakQ, DeltakH, DeltakHprime, DeltakNL;
  double Qnu, fnu;
  double f1a, f2a, f3a, f1b, f2b, f3b, fb_frac;
  double neff2, neff3, neff4;
  double kh2, y2;
  double delta2_norm, om_nu;
  double k=exp(lk);

  // all eqns are from Takahashi et al. unless stated otherwise
  // eqns A4 - A5
  rsigma = gsl_spline_eval(hf->rsigma, a, NULL);
  neff = gsl_spline_eval(hf->n_eff, a, NULL);
  C = gsl_spline_eval(hf->C, a, NULL);

  weffa = cosmo->params.w0;
  omegaMz = ccl_omega_x(cosmo, a, ccl_species_m_label, status);
  omegaDEwz = ccl_omega_x(cosmo, a, ccl_species_l_label, status);

  // not using these to match CLASS better - might be a bug in CLASS
  // weffa = gsl_spline_eval(hf->weff, a, NULL);
  // omegaMz = gsl_spline_eval(hf->omeff, a, NULL);
  // omegaDEwz = gsl_spline_eval(hf->deeff, a, NULL);

  ksigma = 1.0 / rsigma;
  neff2 = neff * neff;
  neff3 = neff2 * neff;
  neff4 = neff3 * neff;

  delta2_norm = k*k*k/2.0/M_PI/M_PI;

  // compute the present day neutrino massive neutrino fraction
  // uses all neutrinos even if they are moving fast
  om_nu = cosmo->params.sum_nu_masses / 93.14 / cosmo->params.h / cosmo->params.h;
  fnu = om_nu / (cosmo->params.Omega_m);

  // eqns A6 - A13 of Takahashi et al.
  an = pow(
    10.0,
    1.5222 + 2.8553*neff + 2.3706*neff2 + 0.9903*neff3 +
    0.2250*neff4 - 0.6038*C + 0.1749*omegaDEwz*(1.0 + weffa));
  bn = pow(10.0, -0.5642 + 0.5864*neff + 0.5716*neff2 - 1.5474*C + 0.2279*omegaDEwz*(1.0 + weffa));
  cn = pow(10.0, 0.3698 + 2.0404*neff + 0.8161*neff2 + 0.5869*C);
  gamman = 0.1971 - 0.0843*neff + 0.8460*C;
  alphan = fabs(6.0835 + 1.3373*neff - 0.1959*neff2 - 5.5274*C);
  betan = 2.0379 - 0.7354*neff + 0.3157*neff2 + 1.2490*neff3 + 0.3980*neff4 - 0.1682*C;
  mun = 0.0;
  nun = pow(10.0, 5.2105 + 3.6902*neff);

  // eqns C17 and C18 for Smith et al.
  if (fabs(1.0 - omegaMz) > 0.01) {
    f1a = pow(omegaMz, -0.0732);
    f2a = pow(omegaMz, -0.1423);
    f3a = pow(omegaMz, 0.0725);
    f1b = pow(omegaMz, -0.0307);
    f2b = pow(omegaMz, -0.0585);
    f3b = pow(omegaMz, 0.0743);
    fb_frac = omegaDEwz / (1.0 - omegaMz);
    f1 = fb_frac * f1b + (1.0 - fb_frac) * f1a;
    f2 = fb_frac * f2b + (1.0 - fb_frac) * f2a;
    f3 = fb_frac * f3b + (1.0 - fb_frac) * f3a;
  } else {
    f1 = 1.0;
    f2 = 1.0;
    f3 = 1.0;
  }

  // correction to betan from Bird et al., eqn A10
  betan += (fnu * (1.081 + 0.395*neff2));

  // eqns A1 - A3
  PkL = ccl_f2d_t_eval(plin, lk, a, cosmo, status);
  y = k/ksigma;
  y2 = y * y;
  fy = y/4.0 + y2/8.0;
  DeltakL = PkL * delta2_norm;

  // correction to DeltakL from Bird et al., eqn A9
  kh = k / cosmo->params.h;
  kh2 = kh * kh;
  DeltakL_tilde = DeltakL * (1.0 + fnu * (47.48 * kh2) / (1.0 + 1.5 * kh2));
  DeltakQ = DeltakL * pow(1.0 + DeltakL_tilde, betan) / (1.0 + alphan*DeltakL_tilde) * exp(-fy);

  DeltakHprime = an * pow(y, 3.0*f1) / (1.0 + bn*pow(y, f2) + pow(cn*f3*y, 3.0 - gamman));
  DeltakH = DeltakHprime / (1.0 + mun/y + nun/y2);

  // correction to DeltakH from Bird et al., eqn A6-A7
  Qnu = fnu * (0.977 - 18.015 * (cosmo->params.Omega_m - 0.3));
  DeltakH *= (1.0 + Qnu);

  DeltakNL = DeltakQ + DeltakH;
  PkNL = DeltakNL / delta2_norm;

  // we check the status once
  if(*status != 0)
    return NAN;
  else
    return PkNL;
}
