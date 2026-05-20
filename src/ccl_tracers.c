#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>

#include "ccl.h"


ccl_cl_tracer_collection_t *ccl_cl_tracer_collection_t_new(int *status) {
  ccl_cl_tracer_collection_t *trc = NULL;
  trc = malloc(sizeof(ccl_cl_tracer_collection_t));
  if (trc == NULL)
    *status = CCL_ERROR_MEMORY;

  if (*status == 0) {
    trc->n_tracers = 0;
    // Currently CCL_MAX_TRACERS_PER_COLLECTION is hard-coded to 100.
    // It should be enough for any practical application with minimal memory overhead
    trc->ts = malloc(CCL_MAX_TRACERS_PER_COLLECTION*sizeof(ccl_cl_tracer_t *));
    if (trc->ts == NULL) {
      *status = CCL_ERROR_MEMORY;
      free(trc);
      trc = NULL;
    }
  }

  return trc;
}

void ccl_cl_tracer_collection_t_free(ccl_cl_tracer_collection_t *trc) {
  if (trc != NULL) {
    if (trc->ts != NULL)
      free(trc->ts);
    free(trc);
  }
}

void ccl_add_cl_tracer_to_collection(ccl_cl_tracer_collection_t *trc,
                                     ccl_cl_tracer_t *tr, int *status) {
  if (trc->n_tracers >= CCL_MAX_TRACERS_PER_COLLECTION) {
    *status = CCL_ERROR_MEMORY;
    return;
  }
  trc->ts[trc->n_tracers] = tr;
  trc->n_tracers++;
}

//Integrand for N(z) integrator
static double nz_integrand(double z, void *pars) {
  ccl_f1d_t *nz_f = (ccl_f1d_t *)pars;

  return ccl_f1d_t_eval(nz_f,z);
}

// Gets area of N(z) curve
static double get_nz_norm(ccl_cosmology *cosmo, ccl_f1d_t *nz_f,
                          double z0, double zf, int *status) {
  double nz_norm = -1, nz_enorm;

  if(cosmo->gsl_params.NZ_NORM_SPLINE_INTEGRATION) {
    gsl_interp_accel *accel = NULL;
    accel = gsl_interp_accel_alloc();
    if(accel == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo, "ccl_tracers.c: get_nz_norm(): out of memory");
    }
    if(*status == 0) {
      *status = gsl_spline_eval_integ_e(nz_f->spline, z0, zf, accel, &nz_norm);
      if(*status) {
        *status = CCL_ERROR_SPLINE_EV;
        nz_norm = NAN;
        ccl_cosmology_set_status_message(
          cosmo, "ccl_tracers.c: get_nz_norm(): Spline integration failed.");
      }
    }
    gsl_interp_accel_free(accel);
    return nz_norm;
  }
  else {
    // Use GSL integration routine
    gsl_function F;
    gsl_integration_workspace *w = NULL;
    F.function = &nz_integrand;
    F.params = nz_f;

    w = gsl_integration_workspace_alloc(cosmo->gsl_params.N_ITERATION);

    if (w == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo, "ccl_tracers.c: get_nz_norm(): out of memory");
    }
    else {
      int gslstatus = gsl_integration_qag(
        &F, z0, zf, 0,
        cosmo->gsl_params.INTEGRATION_EPSREL,
        cosmo->gsl_params.N_ITERATION,
        cosmo->gsl_params.INTEGRATION_GAUSS_KRONROD_POINTS,
        w, &nz_norm, &nz_enorm);

      if (gslstatus != GSL_SUCCESS) {
        *status = CCL_ERROR_INTEG;
        ccl_cosmology_set_status_message(
          cosmo,
          "ccl_tracers.c: get_nz_norm(): "
          "integration error when normalizing N(z)\n");
      }
    }

    gsl_integration_workspace_free(w);

    return nz_norm;
  }
}

void ccl_get_number_counts_kernel(ccl_cosmology *cosmo,
                                  int nz, double *z_arr, double *nz_arr,
                                  int normalize_nz,
                                  double *pchi_arr, int *status) {
  // Returns dn/dchi normalized to unit area from an unnormalized dn/dz.
  // Prepare N(z) spline
  ccl_f1d_t *nz_f = NULL;

  nz_f = ccl_f1d_t_new(nz, z_arr, nz_arr, 0, 0,
		       ccl_f1d_extrap_const,
		       ccl_f1d_extrap_const, status);
  if (nz_f == NULL) {
    *status = CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_tracers.c: ccl_get_number_counts_kernel(): "
      "error initializing spline\n");
  }

  // Get N(z) normalization
  double i_nz_norm = -1;
  if (*status == 0) {
    if (normalize_nz)
      i_nz_norm = 1./get_nz_norm(cosmo, nz_f, z_arr[0], z_arr[nz-1], status);
    else
      i_nz_norm = 1;
  }

  if (*status == 0) {
    // Populate arrays
    for(int ichi=0; ichi < nz; ichi++) {
      double a = 1./(1+z_arr[ichi]);
      double h = cosmo->params.h*ccl_h_over_h0(cosmo,a,status)/ccl_constants.CLIGHT_HMPC;
      // H(z) * dN/dz * 1/Ngal
      pchi_arr[ichi] = h*nz_arr[ichi]*i_nz_norm;
    }
  }

  ccl_f1d_t_free(nz_f);
}

//3 H0^2 Omega_M / 2
static double get_lensing_prefactor(ccl_cosmology *cosmo,int *status) {
  double hub = cosmo->params.h/ccl_constants.CLIGHT_HMPC;
  return 1.5*hub*hub*cosmo->params.Omega_m;
}

typedef struct {
  ccl_cosmology *cosmo;
  double z_max;
  double z_end;
  double chi_end;
  double i_nz_norm;
  ccl_f1d_t *nz_f;
  ccl_f1d_t *sz_f;
  int *status;
} integ_lensing_pars;

// Integrand for lensing kernel.
// Returns N(z) * (1 - 5*s(z)/2) * (chi(z)-chi) / chi(z)
static double lensing_kernel_integrand(ccl_cosmology *cosmo, double chi, double chi_end, double pz, double qz, int *status) {
  if (chi == 0)
    return pz * qz;
  else {
    return (
      pz * qz *
      ccl_sinn(cosmo, chi - chi_end, status) /
      ccl_sinn(cosmo, chi, status));
  }
}

// Integrand for lensing kernel for GSL integration routines.
static double lensing_kernel_integrand_gsl(double z, void *pars) {
  integ_lensing_pars *p = (integ_lensing_pars *)pars;
  double pz = ccl_f1d_t_eval(p->nz_f, z);
  double qz;
  if (p->sz_f == NULL) // No magnification factor
    qz = 1;
  else // With magnification factor
    qz = (1 - 2.5*ccl_f1d_t_eval(p->sz_f, z));

  if (z == 0)
    return pz * qz;
  else {
    double chi = ccl_comoving_radial_distance(p->cosmo, 1./(1+z), p->status);
    return lensing_kernel_integrand(p->cosmo, chi, p->chi_end, pz, qz, p->status);
  }
}

// Returns
// Integral[ p(z) * (1-5s(z)/2) * chi_end * (chi(z)-chi_end)/chi(z) , {z',z_end,z_max} ]
static double lensing_kernel_integrate_qag_wrapper(ccl_cosmology *cosmo,
                                       integ_lensing_pars *pars,
                                       gsl_integration_workspace *w, double *error) {
  int gslstatus = 0;
  double result;
  gsl_function F;
  F.function = &lensing_kernel_integrand_gsl;
  F.params = pars;
  gslstatus = gsl_integration_qag(
    &F, pars->z_end, pars->z_max, 0,
    cosmo->gsl_params.INTEGRATION_EPSREL,
    cosmo->gsl_params.N_ITERATION,
    cosmo->gsl_params.INTEGRATION_GAUSS_KRONROD_POINTS,
    w, &result, error);
  
  if ((gslstatus != GSL_SUCCESS
       && gslstatus != GSL_EROUND && gslstatus != GSL_EMAXITER)
      || (*(pars->status))) {
    ccl_raise_gsl_warning(gslstatus, "ccl_tracers.c: lensing_kernel_integrate(): gsl_integration_qag failed.");
    *(pars->status) = CCL_ERROR_INTEG;
    return -1;
  }

  return result * pars->i_nz_norm * pars->chi_end;
}

// Computes the lensing kernel integral using GSL quadrature integration:
// 3 * H0^2 * Omega_M / 2 / a *
// Integral[ p(z) * (1-5s(z)/2) * chi_end * (chi(z)-chi_end)/chi(z) ,
//          {z',z_end,z_max} ]
static void integrate_lensing_kernel_gsl(ccl_cosmology *cosmo, double z_max, double nz_norm,
                                  ccl_f1d_t *nz_f, ccl_f1d_t *sz_f,
                                  int nchi, double* chi_arr, double* wL_arr,
                                  int* status) {
  double* wL_err_arr = malloc(nchi*sizeof(double));
  if(wL_err_arr == NULL) {
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_tracers.c: integrate_lensing_kernel_gsl(): error allocating memory\n");
  }
  if(*status == 0) {
    #pragma omp parallel default(none) \
                          shared(cosmo, z_max, nz_norm, sz_f, nz_f, \
                                nchi, chi_arr, wL_arr, wL_err_arr, status)
    {
      double chi, a, z, mgfac, lens_prefac, wL_err;
      int ichi, local_status;
      integ_lensing_pars *ipar = NULL;
      gsl_integration_workspace *w = NULL;

      local_status = *status;

      lens_prefac = get_lensing_prefactor(cosmo, &local_status);
      if (local_status == 0) {
        ipar = malloc(sizeof(integ_lensing_pars));
        w = gsl_integration_workspace_alloc(cosmo->gsl_params.N_ITERATION);

        if ((ipar == NULL) || (w == NULL)) {
          local_status = CCL_ERROR_MEMORY;
        }
      }

      if (local_status == 0) {
        ipar->cosmo = cosmo;
        ipar->z_max = z_max;
        ipar->i_nz_norm = nz_norm;
        ipar->sz_f = sz_f;
        ipar->nz_f = nz_f;
        ipar->status = &local_status;
      }

      //Populate arrays
      #pragma omp for
      for (ichi=0; ichi < nchi; ichi++) {
        if (local_status == 0) {
          chi = chi_arr[ichi];
          a = ccl_scale_factor_of_chi(cosmo, chi, &local_status);
          z = 1./a-1;
          ipar->z_end = z;
          ipar->chi_end = chi;
          wL_arr[ichi] = lensing_kernel_integrate_qag_wrapper(cosmo, ipar, w, &wL_err)*(1+z)*lens_prefac;
          wL_err_arr[ichi] = wL_err*(1+z)*lens_prefac;
          local_status = *(ipar->status);
        } else {
          wL_arr[ichi] = NAN;
        }
      } //end omp for

      gsl_integration_workspace_free(w);
      free(ipar);

      if (local_status) {
        #pragma omp atomic write
        *status = CCL_ERROR_INTEG;
      }
    } //end omp parallel

    if(*status) {
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_tracers.c: integrate_lensing_kernel_gsl(): error in computing lensing kernel.\n");
    } else {
      // Check if error estimates are sufficiently small compared to the peak of the lensing kernel.
      // First find the maxium value of the lensing kernel wL_max
      double wL_max = 0.0;
      for(int i=0; i < nchi; i++) {
        if(fabs(wL_arr[i]) > wL_max) {
          wL_max = fabs(wL_arr[i]);
        }
      }
      // Now check that the integration errors are smaller than wL_max*rel_tol
      if(wL_max > 0.0) {
        double rel_tol = cosmo->gsl_params.INTEGRATION_EPSREL;
        for(int i=0; i < nchi; i++) {
          if(wL_err_arr[i] > wL_max*rel_tol) {
            *status = CCL_ERROR_INTEG;
            ccl_cosmology_set_status_message(
              cosmo,
              "ccl_tracers.c: integrate_lensing_kernel_gsl(): error in computing lensing kernel. "
              "Integration error at chi %g larger than tolerance of %g: %g\n",
              chi_arr[i],  wL_max*rel_tol, wL_err_arr[i]);
            break;
          }
        }
      }
    }
  }
  free(wL_err_arr);
}

// Computes the lensing kernel integral using spline integration:
// 3 * H0^2 * Omega_M / 2 / a *
// Integral[ p(z) * (1-5s(z)/2) * chi_end * (chi(z)-chi_end)/chi(z) ,
//          {z',z_end,z_max} ]
static void integrate_lensing_kernel_spline(ccl_cosmology *cosmo,
                                              int nz, double* z_arr, double* nz_arr, double nz_norm,
                                              ccl_f1d_t* sz_f,
                                              int nchi, double* chi_arr, double* wL_arr,
                                              int* status) {
  double* chi_of_z_array = malloc(nz*sizeof(double));
  double* qz_array = malloc(nz*sizeof(double));
  if(chi_of_z_array == NULL || qz_array == NULL) {
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_tracers.c: lensing_kernel_integrate_spline(): error allocating memory\n");
  }

  if(*status == 0) {
    // Fill chi and qz arrays
    for(int i=0; i<nz; i++) {
      double a = 1./(1+z_arr[i]);
      chi_of_z_array[i] = ccl_comoving_radial_distance(cosmo, a, status);
      if(sz_f != NULL) {
        qz_array[i] = 1-2.5*ccl_f1d_t_eval(sz_f, z_arr[i]);
      } else {
        qz_array[i] = 1.0;
      }
    }
  }

  if(*status == 0) {
    #pragma omp parallel default(none) \
                        shared(cosmo, nz, z_arr, nz_arr, nz_norm, \
                               chi_of_z_array, qz_array, \
                               nchi, chi_arr, wL_arr, status, gsl_interp_akima)
    {
      int local_status = *status;
      double lens_prefac = get_lensing_prefactor(cosmo, &local_status);
      double result = 0.0;
      double chi, chi_end, a, z_end;

      double *integrand_array = malloc(nz*sizeof(double));
      if(integrand_array == NULL) {
        local_status = CCL_ERROR_MEMORY;
        ccl_cosmology_set_status_message(
          cosmo,
          "ccl_tracers.c: lensing_kernel_integrate_spline(): error allocating memory\n");
      }
      if(local_status == 0) {
        #pragma omp for
        for(int ichi=0; ichi<nchi; ichi++) {
          chi_end = chi_arr[ichi];
          a = ccl_scale_factor_of_chi(cosmo, chi_end, &local_status);
          z_end = 1./a-1;

          // We don't need to start at 0 but finding the index corresponding to chi_end would make things more complicated
          int i_chi_end = 0;
          for(int i=0; i<nz; i++) {
            if(chi_of_z_array[i] < chi_end) {
              integrand_array[i] = 0.0;
              i_chi_end = i+1;
            } else {
	      integrand_array[i] = lensing_kernel_integrand(cosmo,
							    chi_of_z_array[i],
							    chi_end, nz_arr[i],
							    qz_array[i],
							    &local_status);
            }
          }
          if(local_status) {
            ccl_raise_warning(CCL_ERROR_INTEG, "ccl_tracers.c: integrate_lensing_kernel_spline(): error in lensing_kernel_integrand.\n");
          } else {
            ccl_integ_spline(1, nz, z_arr, &integrand_array, z_arr[i_chi_end], z_arr[nz-1], &result, gsl_interp_akima, &local_status);
          }

          // Correct for the missing interval (chi_end, chi_of_z_array[i_chi_end]) in the integral,
          // where i_chi_end is the smallest index such that chi_end < chi_of_z_array[i_chi_end].
          // Trapezoidal rule: \int_a^b f(x) dx \approx (b-a)(f(a) + f(b))/2
          // Here a=z_end, b=zarr[i_chi_end], and f(x) = lensing_kernel_integrand
          // Since lensing_kernel_integrand(cosmo, chi_end, chi_end, ...) = 0, we have f(a) = 0
          // Only do this if z_end is greater than the support of the provided n(z), to avoid
          // inaccurate results due to using the trapezoidal rule for large intervals of z_arr[0] - z_end.
          if(z_end > z_arr[0]) {
            double trapz = 0.5 * (z_arr[i_chi_end] - z_end) * integrand_array[i_chi_end];
            result += trapz;
          }

          if(local_status == 0) {
            wL_arr[ichi] = result * lens_prefac * nz_norm * chi_end / a;
          } else {
            wL_arr[ichi] = NAN;
            ccl_raise_warning(CCL_ERROR_INTEG, "ccl_tracers.c: integrate_lensing_kernel_spline(): error in ccl_integ_spline.\n");
          }
        }
      }
      free(integrand_array);
      if(local_status) {
        #pragma omp atomic write
        *status = CCL_ERROR_INTEG;
      }
    } //end omp parallel
    if(*status) {
      ccl_cosmology_set_status_message(
      cosmo,
      "ccl_tracers.c: integrate_lensing_kernel_spline(): error in computing lensing kernel.\n");
    }  
  }

  free(chi_of_z_array);
  free(qz_array);
}

//Returns number of divisions on which
//the lensing kernel should be calculated
int ccl_get_nchi_lensing_kernel(int nz, double *z_arr, int *status) {
  double dz = -1;
  //Compute redshift step
  dz = (z_arr[nz-1]-z_arr[0])/(nz-1);

  //How many steps to z=0?
  return (int)(z_arr[nz-1]/dz+0.5);
}

//Return array with the values of chi at
//the which the lensing kernel will be
//calculated.
void ccl_get_chis_lensing_kernel(ccl_cosmology *cosmo,
                                 int nchi, double z_max,
                                 double *chis, int *status) {
  double dz = z_max/nchi;
  for(int ichi=0; ichi < nchi; ichi++) {
    double z = dz*ichi+1E-15;
    double a = 1./(1+z);
    chis[ichi] = ccl_comoving_radial_distance(cosmo, a, status);
  }
}

//Returns array with lensing kernel:
//3 * H0^2 * Omega_M / 2 / a *
// Integral[ p(z) * (1-5s(z)/2) * chi_end * (chi(z)-chi_end)/chi(z) ,
//          {z',z_end,z_max} ]
void ccl_get_lensing_mag_kernel(ccl_cosmology *cosmo,
                                int nz, double *z_arr, double *nz_arr,
                                int normalize_nz, double z_max,
                                int nz_s, double *zs_arr, double *sz_arr,
                                int nchi, double *chi_arr, double *wL_arr,
                                int *status) {
  ccl_f1d_t *nz_f = NULL;
  ccl_f1d_t *sz_f = NULL;

  // Prepare N(z) spline
  nz_f = ccl_f1d_t_new(nz, z_arr, nz_arr, 0, 0,
		       ccl_f1d_extrap_const,
		       ccl_f1d_extrap_const, status);
  if (nz_f == NULL) {
    *status = CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_tracers.c: ccl_get_lensing_mag_kernel(): error initializing spline\n");
  }

  // Get N(z) normalization
  double i_nz_norm = -1;
  if (*status == 0) {
    if (normalize_nz)
      i_nz_norm = 1./get_nz_norm(cosmo, nz_f, z_arr[0], z_arr[nz-1], status);
    else
      i_nz_norm = 1.;
  }

  // Prepare magnification bias spline if needed
  if (*status == 0) {
    if ((nz_s > 0) && (zs_arr != NULL) && (sz_arr != NULL)) {
      sz_f = ccl_f1d_t_new(nz_s, zs_arr, sz_arr, sz_arr[0], sz_arr[nz_s-1],
			   ccl_f1d_extrap_const,
			   ccl_f1d_extrap_const, status);
      if (sz_f == NULL) {
        *status = CCL_ERROR_SPLINE;
        ccl_cosmology_set_status_message(
          cosmo,
          "ccl_tracers.c: ccl_get_lensing_mag_kernel(): error initializing spline\n");
      }
    }
  }

  if(*status == 0) {
    if(cosmo->gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION) {
      integrate_lensing_kernel_spline(cosmo,
                                      nz, z_arr, nz_arr, i_nz_norm,
                                      sz_f, nchi, chi_arr, wL_arr, status);
    } else {
      integrate_lensing_kernel_gsl(cosmo, z_max, i_nz_norm,
                                  nz_f, sz_f,
                                  nchi, chi_arr, wL_arr,
                                  status);
    }
    if(*status) {
      ccl_raise_warning(
        CCL_ERROR_INTEG,
        "ccl_tracers.c: ccl_get_lensing_mag_kernel(): failed to compute lensing kernel with quadrature. "
        "Trying spline integration. This indicates that the n(z) is pathological.\n");
    }
  }

  ccl_f1d_t_free(nz_f);
  ccl_f1d_t_free(sz_f);
}

// Returns kernel for CMB lensing
// 3H0^2Om/2 * chi * (chi_s - chi) / chi_s / a
void ccl_get_kappa_kernel(ccl_cosmology *cosmo, double chi_source,
                          int nchi, double *chi_arr,
                          double *wchi, int *status) {
  double lens_prefac = get_lensing_prefactor(cosmo, status) / ccl_sinn(cosmo, chi_source, status);

  for (int ichi=0; ichi < nchi; ichi++) {
    double chi = chi_arr[ichi];
    double a = ccl_scale_factor_of_chi(cosmo, chi, status);
    if (chi != 0.0)
      wchi[ichi] = lens_prefac*(ccl_sinn(cosmo,chi_source-chi,status))*chi*chi/a/ccl_sinn(cosmo, chi, status);
    else
      wchi[ichi] = lens_prefac*(ccl_sinn(cosmo,chi_source-chi,status))*chi/a;
  }
}

ccl_cl_tracer_t *ccl_cl_tracer_t_new(ccl_cosmology *cosmo,
                                     int der_bessel,
                                     int der_angles,
                                     int n_w, double *chi_w, double *w_w,
                                     int na_ka, double *a_ka,
                                     int nk_ka, double *lk_ka,
                                     double *fka_arr,
                                     double *fk_arr,
                                     double *fa_arr,
                                     int is_fka_log,
                                     int is_factorizable,
                                     int extrap_order_lok,
                                     int extrap_order_hik,
                                     int *status) {
  ccl_cl_tracer_t *tr = NULL;

  // Check der_bessel and der_angles are sensible
  if ((der_angles < 0) || (der_angles > 2)) {
    *status = CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_tracers.c: ccl_cl_tracer_new(): der_angles must be between 0 and 2\n");
  }
  if ((der_bessel < -1) || (der_bessel > 2)) {
    *status = CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_tracers.c: ccl_cl_tracer_new(): der_bessel must be between -1 and 2\n");
  }

  if (*status == 0) {
    tr = malloc(sizeof(ccl_cl_tracer_t));
    if (tr == NULL)
      *status = CCL_ERROR_MEMORY;
  }

  // Initialize everythin
  if (*status == 0) {
    tr->der_angles = der_angles;
    tr->der_bessel = der_bessel;
    tr->kernel = NULL; // Initialize these to NULL
    tr->transfer = NULL; // Initialize these to NULL
    tr->chi_min = 0;
    tr->chi_max = 1E15;
  }

  if (*status == 0) {
    // Initialize radial kernel
    if ((n_w > 0) && (chi_w != NULL) && (w_w != NULL)) {
      tr->kernel = ccl_f1d_t_new(n_w,chi_w,w_w,0,0,
				 ccl_f1d_extrap_const,
				 ccl_f1d_extrap_const, status);
      if (tr->kernel == NULL)
        *status=CCL_ERROR_MEMORY;
    }
  }

  // Find kernel edges
  if (*status == 0) {
    // If no radial kernel, set limits to zero and maximum distance
    if (tr->kernel == NULL) {
      tr->chi_min = 0;
      tr->chi_max = ccl_comoving_radial_distance(cosmo, cosmo->spline_params.A_SPLINE_MIN, status);
    }
    else {
      int ichi;
      double w_max = fabs(w_w[0]);

      // Find maximum of radial kernel
      for (ichi=0; ichi < n_w; ichi++) {
        if (fabs(w_w[ichi]) >= w_max)
          w_max = fabs(w_w[ichi]);
      }

      // Multiply by fraction
      w_max *= CCL_FRAC_RELEVANT;

      // Initialize as the original edges in case we don't find an interval
      tr->chi_min = chi_w[0];
      tr->chi_max = chi_w[n_w-1];

      // Find minimum
      for (ichi=0; ichi < n_w-1; ichi++) {
        if (fabs(w_w[ichi+1]) >= w_max) {
          tr->chi_min = chi_w[ichi];
          break;
        }
      }

      // Find maximum
      for (ichi=n_w-1; ichi >= 1; ichi--) {
        if (fabs(w_w[ichi-1]) >= w_max) {
          tr->chi_max = chi_w[ichi];
          break;
        }
      }
    }
  }

  if (*status == 0) {
    if ((fka_arr != NULL) || (fk_arr != NULL) || (fa_arr != NULL)) {
      tr->transfer = ccl_f2d_t_new(
        na_ka,a_ka, // na, a_arr
        nk_ka,lk_ka, // nk, lk_arr
        fka_arr, // fka_arr
        fk_arr, // fk_arr
        fa_arr, // fa_arr
        is_factorizable, // is factorizable
        extrap_order_lok, // extrap_order_lok
        extrap_order_hik, // extrap_order_hik
        ccl_f2d_constantgrowth, // extrap_linear_growth
        is_fka_log, // is_fka_log
        1, // growth_factor_0 -> will assume constant transfer function
        0, // growth_exponent
        ccl_f2d_3, // interp_type
        status);
      if (tr->transfer == NULL)
        *status=CCL_ERROR_MEMORY;
    }
  }

  return tr;
}

void ccl_cl_tracer_t_free(ccl_cl_tracer_t *tr) {
  if (tr != NULL) {
    if (tr->transfer != NULL)
      ccl_f2d_t_free(tr->transfer);
    if (tr->kernel != NULL)
      ccl_f1d_t_free(tr->kernel);
    free(tr);
  }
}

double ccl_cl_tracer_t_get_f_ell(ccl_cl_tracer_t *tr, double ell, int *status) {
  if (tr != NULL) {
    if (tr->der_angles == 1)
      return ell*(ell+1.);
    else if (tr->der_angles == 2) {
      if (ell <= 1) // This is identically 0
        return 0;
      else if (ell <= 10) // Use full expression in this case
        return sqrt((ell+2)*(ell+1)*ell*(ell-1));
      else {
        double lp1h = ell+0.5;
        double lp1h2 = lp1h*lp1h;
        if (ell <= 1000)  // This is accurate to 5E-5 for l>10
          return lp1h2*(1-1.25/lp1h2);
        else // This is accurate to 1E-6 for l>1000
          return lp1h2;
      }
    }
    else
      return 1;
  }
  else
    return 1;
}

double ccl_cl_tracer_t_get_kernel(ccl_cl_tracer_t *tr, double chi, int *status) {
  if (tr != NULL) {
    if (tr->kernel != NULL)
      return ccl_f1d_t_eval(tr->kernel, chi);
    else
      return 1;
  }
  else
    return 1;
}

double ccl_cl_tracer_t_get_transfer(ccl_cl_tracer_t *tr,
                                    double lk, double a, int *status) {
  if (tr != NULL) {
    if (tr->transfer != NULL)
      return ccl_f2d_t_eval(tr->transfer, lk, a, NULL, status);
    else
      return 1;
  }
  else
    return 1;
}
