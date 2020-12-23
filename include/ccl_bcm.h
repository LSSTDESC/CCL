/** @file */
#ifndef __CCL_BCM_H_INCLUDED__
#define __CCL_BCM_H_INCLUDED__

CCL_BEGIN_DECLS

/**
 * Correction for the impact of baryonic physics on the matter power spectrum.
 * Returns f(k,a) [dimensionless] for given cosmology, using the method specified for the baryonic transfer function.
 * f(k,a) is the fractional change in the nonlinear matter power spectrum from the Baryon Correction Model (BCM) of Schenider & Teyssier (2015). The parameters of the model are passed as part of the cosmology class.
 * @param cosmo Cosmology parameters and configurations, including baryonic parameters.
 * @param k Fourier mode, in [1/Mpc] units
 * @param a scale factor, normalized to 1 for today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return f(k,a).
 */
double ccl_bcm_model_fka(ccl_cosmology * cosmo, double k, double a, int *status);

void ccl_bcm_correct(ccl_cosmology *cosmo, ccl_f2d_t *psp, int *status);

CCL_END_DECLS

#endif
