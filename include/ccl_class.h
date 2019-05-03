/** @file */
#ifndef __CCL_CLASS_H_INCLUDED__
#define __CCL_CLASS_H_INCLUDED__

CCL_BEGIN_DECLS

/*
 * Compute the power spectrum using CLASS
 * @param cosmo Cosmological parameters
 * @param status, integer indicating the status
 */
void ccl_cosmology_compute_linpower_class(ccl_cosmology* cosmo, int* status);

CCL_END_DECLS

#endif
