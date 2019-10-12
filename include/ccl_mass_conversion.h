/** @file */
#ifndef __CCL_M_CONV_H_INCLUDED__
#define __CCL_M_CONV_H_INCLUDED__

CCL_BEGIN_DECLS

void ccl_get_new_concentration(ccl_cosmology *cosmo,
			       double delta_old, int nc, double c_old[],
			       double delta_new, double c_new[],int *status);

CCL_END_DECLS

#endif
