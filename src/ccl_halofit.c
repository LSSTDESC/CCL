#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>

#include "ccl.h"

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
  int *status;
  gsl_integration_cquad_workspace *workspace;
};

static double gauss_norm_int_func(double lnk, void *p) {
  struct hf_int_data *hfd = (struct hf_int_data*)p;
  double k = exp(lnk);
  double k2 = k*k;

  return (
    ccl_linear_matter_power(hfd->cosmo, k, hfd->a, hfd->status) *
    k*k2/2.0/M_PI/M_PI *
    exp(-k2 * (hfd->r2)));
}

static double onederiv_gauss_norm_int_func(double lnk, void *p) {
  struct hf_int_data *hfd = (struct hf_int_data*)p;
  double k = exp(lnk);
  double k2 = k*k;

  return (
    ccl_linear_matter_power(hfd->cosmo, k, hfd->a, hfd->status) *
    k*k2/2.0/M_PI/M_PI *
    exp(-k2 * (hfd->r2)) *
    (-k2 * 2.0 * (hfd->r)));
}

static double twoderiv_gauss_norm_int_func(double lnk, void *p) {
  struct hf_int_data *hfd = (struct hf_int_data*)p;
  double k = exp(lnk);
  double k2 = k*k;

  return (
    ccl_linear_matter_power(hfd->cosmo, k, hfd->a, hfd->status) *
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

  lnkmin = hfd->cosmo->data.p_lin->lkmin;
  lnkmax = hfd->cosmo->data.p_lin->lkmax;
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
      "for finding the halofit non-linear scale");
    *(hfd->status) |= gsl_status;
  }

  return result - 1.0;
}

static double get_rsigma(double a, struct hf_int_data data) {
  double rsigma, rlow = 0.0001, rhigh = 100.0;
  int itr, max_itr = 1000, gsl_status;
  const gsl_root_fsolver_type *T;
  gsl_root_fsolver *s;
  gsl_function F;

  data.a = a;
  F.function = &rsigma_func;
  F.params = &data;

  T = gsl_root_fsolver_brent;
  s = gsl_root_fsolver_alloc(T);
  gsl_root_fsolver_set(s, &F, rlow, rhigh);

  itr = 0;
  do {
    itr++;
    gsl_status = gsl_root_fsolver_iterate(s);
    *(data.status) |= gsl_status;
    if (*(data.status) != 0)
      break;

    rsigma = gsl_root_fsolver_root(s);
    rlow = gsl_root_fsolver_x_lower(s);
    rhigh = gsl_root_fsolver_x_upper(s);

    gsl_status = gsl_root_test_interval(
      rlow, rhigh,
      data.cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL,
      data.cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL);
    *(data.status) |= gsl_status;
    if (*(data.status) != 0)
      break;

  } while (gsl_status == GSL_CONTINUE && itr < max_itr);

  gsl_root_fsolver_free(s);

  if (gsl_status != GSL_SUCCESS || itr >= max_itr) {
    ccl_raise_gsl_warning(
      gsl_status, "ccl_halofit.c: get_rsigma: error in root finding for the halofit non-linear scale");
    *(data.status) |= gsl_status;
  }

  return rsigma;
}

/*
 * Allocate a new struct for storing halofit data
 * @param cosmo Cosmological data
 * @param int, status of computations
 */
halofit_struct* ccl_halofit_struct_new(ccl_cosmology *cosmo, int *status) {
  int n_a, i, gsl_status;
  double amin, amax;
  double lnkmin, lnkmax;
  double *a_vec = NULL;
  double *vals = NULL;
  halofit_struct *hf = NULL;
  struct hf_int_data data;
  gsl_function F;
  gsl_integration_cquad_workspace *workspace = NULL;
  double result;
  double sigma2, rsigma, dsigma2drsigma;

  // compute spline point locations and integral bounds
  // note that the spline point locations in `a` determine a radius by
  // solving sigma2(R, a) = 1
  // it is this radius that is needed for the subsequent splines of
  // the derivatives.
  amin = cosmo->data.p_lin->amin;
  amax = cosmo->data.p_lin->amax;
  n_a = cosmo->spline_params.A_SPLINE_NA_PK + cosmo->spline_params.A_SPLINE_NLOG_PK - 1;
  lnkmin = cosmo->data.p_lin->lkmin;
  lnkmax = cosmo->data.p_lin->lkmax;

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
    a_vec = ccl_linlog_spacing(
      amin, cosmo->spline_params.A_SPLINE_MIN_PK,
      amax, cosmo->spline_params.A_SPLINE_NLOG_PK,
      cosmo->spline_params.A_SPLINE_NA_PK);
    if (a_vec == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_halofit.c: ccl_halofit_struct_new(): "
        "memory could not be allocated for scale factor vector\n");
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

  ///////////////////////////////////////////////////////
  // first find the nonlinear scale at each scale factor
  if (*status == 0) {
    // setup for integrations
    data.status = status;
    data.cosmo = cosmo;
    data.workspace = workspace;

    for (i=0; i<n_a; ++i) {
      vals[i] = get_rsigma(a_vec[i], data);
      if (*status != 0) {
        *status = CCL_ERROR_ROOT;
        ccl_cosmology_set_status_message(
          cosmo,
          "ccl_halofit.c: ccl_halofit_struct_new(): "
          "could not solve for non-linear scale for halofit\n");
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
        lnkmin, lnkmax,
        0.0, cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL,
        workspace, &result, NULL, NULL);

      if (gsl_status != GSL_SUCCESS) {
        *status = CCL_ERROR_INTEG;
        ccl_raise_gsl_warning(
          gsl_status,
          "ccl_power.c: ccl_halofit_struct_new(): could not eval "
          "points for n_eff spline");
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
        lnkmin, lnkmax,
        0.0, cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL,
        workspace, &result, NULL, NULL);

      if (gsl_status != GSL_SUCCESS) {
        *status = CCL_ERROR_INTEG;
        ccl_raise_gsl_warning(
          gsl_status,
          "ccl_power.c: ccl_halofit_struct_new(): could not eval "
          "points for C spline");
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
  if (*status != 0)
    ccl_halofit_struct_free(hf);
  gsl_integration_cquad_workspace_free(workspace);
  free(a_vec);
  free(vals);

  return hf;
}

/*
 * Free a halofit struct
 * @param hf, pointer to halofit struct to free
 */
void ccl_halofit_struct_free(halofit_struct *hf) {
  if (hf != NULL) {
    if (hf->rsigma != NULL) {
      gsl_spline_free(hf->rsigma);
      hf->rsigma = NULL;
    }

    if (hf->sigma2 != NULL) {
      gsl_spline_free(hf->sigma2);
      hf->sigma2 = NULL;
    }

    if (hf->n_eff != NULL) {
      gsl_spline_free(hf->n_eff);
      hf->n_eff = NULL;
    }

    if (hf->C != NULL) {
      gsl_spline_free(hf->C);
      hf->C = NULL;
    }

    free(hf);
  }
}
