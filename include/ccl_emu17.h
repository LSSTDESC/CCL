#ifndef __CCL_EMU17_INCLUDED__
#define __CCL_EMU17_INCLUDED__

#define A_MIN_EMU 1./3.
#define K_MAX_EMU 5.0
#define K_MIN_EMU 1.0000000474974513E-003

CCL_BEGIN_DECLS
/**
 * Emulator power spectrum
 * Obtain P(k,z) [Mpc^3] for a given set of input parameters.
 * @param xstarin vector of input parameters for the emulator, including redshift. 
 * @param Pkemu output P(k,z) power spectrum
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * @param cosmo Cosmology parameters and configurations (only relevant for storing status)
 */
void ccl_pkemu(double *xstarin, double **Pkemu, int *status, ccl_cosmology* cosmo);

CCL_END_DECLS
#endif
