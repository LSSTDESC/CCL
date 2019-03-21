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
      "for finding the halofit non-linear scale\n");
    *(hfd->status) |= gsl_status;
  }

  return result - 1.0;
}

static double get_rsigma(double a, struct hf_int_data data) {
  double rsigma, rlow = 1e-2, rhigh = 1e2;
  double flow, fhigh;
  int itr, max_itr = 1000, gsl_status;
  const gsl_root_fsolver_type *T;
  gsl_root_fsolver *s;
  gsl_function F;

  data.a = a;
  F.function = &rsigma_func;
  F.params = &data;

  // we have to bound the root, otherwise return -1
  // we will fiil in any -1's in the calling routine
  flow = rsigma_func(rlow, &data);
  fhigh = rsigma_func(rhigh, &data);
  if (flow * fhigh > 0) {
    return -1;
  }

  T = gsl_root_fsolver_brent;
  s = gsl_root_fsolver_alloc(T);
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
      gsl_status, "ccl_halofit.c: get_rsigma: error in root finding for the halofit non-linear scale\n");
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

    // now go backwards and fill any -1's
    // any scale factor above 0.15 should work ok, so set an error if not
    for (i=0; i<n_a; ++i) {
      if (a_vec[i] >= 0.15 && vals[i] == -1) {
        *status = CCL_ERROR_ROOT;
        ccl_cosmology_set_status_message(
          cosmo,
          "ccl_halofit.c: ccl_halofit_struct_new(): "
          "could not solve for non-linear scale for halofit\n");
        break;
      }
    }
    if (*status == 0) {
      for (i=n_a-2; i>=0; --i) {
        if (vals[i] == -1)
          vals[i] = vals[i+1];
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
        lnkmin, lnkmax,
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
    if (hf->rsigma != NULL)
      gsl_spline_free(hf->rsigma);

    if (hf->sigma2 != NULL)
      gsl_spline_free(hf->sigma2);

    if (hf->n_eff != NULL)
      gsl_spline_free(hf->n_eff);

    if (hf->C != NULL)
      gsl_spline_free(hf->C);

    free(hf);
  }
}

/**
 * Computes the halofit non-linear power spectrum
 * @param cosmo: cosmology object containing parameters
 * @param k: wavenumber in units of Mpc^{-1}
 * @param a: scale factor normalised to a=1 today
 * @param status: Status flag: 0 if there are no errors, non-zero otherwise
 * @param hf: halofit splines for evaluating the power spectrum
 * @return halofit_matter_power: halofit power spectrum, P(k), units of Mpc^{3}
 */
double ccl_halofit_power(ccl_cosmology *cosmo, double k, double a, halofit_struct *hf, int *status) {
  double rsigma, neff, C;
  double ksigma, weffa, omegaMz, omegaDEwz;
  double PkL, PkNL, f1, f2, f3, an, bn, cn, gamman, alphan, betan, nun, mun, y, fy;
  double DeltakL, DeltakQ, DeltakH, DeltakHprime, DeltakNL;

  // eqns A4 - A5
  rsigma = gsl_spline_eval(hf->rsigma, a, NULL);
  neff = gsl_spline_eval(hf->n_eff, a, NULL);
  C = gsl_spline_eval(hf->C, a, NULL);

  ksigma = 1.0 / rsigma;

  // for w0-wa we use the effective w formalism for w0-wa
  // we taylor expand log(a) = log(1 + x) ~ x - x * x / 2
  // near a = 1 or x = 0 for numerical reasons
  if (a < 1.0 - 1e-12)
    weffa = cosmo->params.w0 + cosmo->params.wa - cosmo->params.wa * (a - 1.0) / log(a);
  else
    weffa = cosmo->params.w0 + cosmo->params.wa - cosmo->params.wa  / (1.0 - fabs(a - 1.0) / 2);
  omegaMz = ccl_omega_x(cosmo, a, ccl_species_m_label, status);
  omegaDEwz = ccl_omega_x(cosmo, a, ccl_species_l_label, status);

  // eqns A6 - A13 of Takahashi et al.
  an = pow(
    10.0,
    1.5222 + 2.8553*neff + 2.3706*neff*neff + 0.9903*neff*neff*neff +
    0.2250*neff*neff*neff*neff - 0.6038*C + 0.1749*omegaDEwz*(1.0 + weffa));
  bn = pow(10.0, -0.5642 + 0.5864*neff + 0.5716*neff*neff - 1.5474*C + 0.2279*omegaDEwz*(1.0 + weffa));
  cn = pow(10.0, 0.3698 + 2.0404*neff + 0.8161*neff*neff + 0.5869*C);
  gamman = 0.1971 - 0.0843*neff + 0.8460*C;
  alphan = fabs(6.0835 + 1.3373*neff - 0.1959*neff*neff - 5.5274*C);
  betan = 2.0379 - 0.7354*neff + 0.3157*neff*neff + 1.2490*neff*neff*neff + 0.3980*neff*neff*neff*neff - 0.1682*C;
  mun = 0.0;
  nun = pow(10.0, 5.2105 + 3.6902*neff);

  // eqns A14
  f1 = pow(omegaMz,-0.0307);
  f2 = pow(omegaMz,-0.0585);
  f3 = pow(omegaMz,0.0743);

  // eqns A1 - A3
  PkL = ccl_linear_matter_power(cosmo, k, a, status);
  y = k/ksigma;
  fy = y/4.0 + y*y/8.0;
  DeltakL = PkL*k*k*k/2.0/M_PI/M_PI;

  DeltakQ = DeltakL * pow(1.0 + DeltakL, betan) / (1.0 + alphan*DeltakL) * exp(-1.0*fy);

  DeltakHprime = an * pow(y, 3.0*f1) / (1.0 + bn*pow(y, f2) + pow(cn*f3*y, 3.0 - gamman));
  DeltakH = DeltakHprime / (1.0 + mun/y + nun/y/y);

  DeltakNL = DeltakQ + DeltakH;
  PkNL = DeltakNL / (k*k*k/2.0/M_PI/M_PI);

  // we check the status once
  if(*status != 0)
    return NAN;
  else
    return PkNL;
}
