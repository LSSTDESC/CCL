/** @file */
#ifndef __CCL_HALOFIT_H_INCLUDED__
#define __CCL_HALOFIT_H_INCLUDED__

#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline.h>

CCL_BEGIN_DECLS

typedef struct halofit_struct {
  gsl_spline *rsigma;
  gsl_spline *sigma2;
  gsl_spline *n_eff;
  gsl_spline *C;
  gsl_spline *weff;
  gsl_spline *omeff;
  gsl_spline *deeff;
} halofit_struct;

/*
 * Allocate a new struct for storing halofit data
 * @param cosmo Cosmological data
 * @return int, status of computations
 */
halofit_struct* ccl_halofit_struct_new(ccl_cosmology *cosmo,
                                       ccl_f2d_t *plin, int *status);

/*
 * Free a halofit struct
 * @param hf, pointer to halofit struct to free
 */
void ccl_halofit_struct_free(halofit_struct *hf);

/**
 * Computes the halofit non-linear power spectrum
 * @param cosmo: cosmology object containing parameters
 * @param k: wavenumber in units of Mpc^{-1}
 * @param a: scale factor normalised to a=1 today
 * @param status: Status flag: 0 if there are no errors, non-zero otherwise
 * @param hf: halofit splines for evaluating the power spectrum
 * @return halofit_matter_power: halofit power spectrum, P(k), units of Mpc^{3}
 */
double ccl_halofit_power(ccl_cosmology *cosmo, ccl_f2d_t *plin,
                         double k, double a, halofit_struct *hf, int *status);

CCL_END_DECLS

#endif
