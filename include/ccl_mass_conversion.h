/** @file */
#ifndef __CCL_M_CONV_H_INCLUDED__
#define __CCL_M_CONV_H_INCLUDED__

CCL_BEGIN_DECLS

/**
 * Get concentration for a new mass definition.
 * @param cosmo Cosmological parameters
 * @param delta_old overdensity parameter in the input mass definition.
 * @param nc number of concentration values to translate.
 * @param c_old input concentration.
 * @param delta_new overdensity parameter in the output mass definition.
 * @param c_new output concentration.
 * @param status Status flat. 0 if everything went well.
 */
void ccl_convert_concentration(ccl_cosmology *cosmo,
			       double delta_old, int nc, double c_old[],
			       double delta_new, double c_new[],int *status);

CCL_END_DECLS

#endif
