/** @file */
#ifndef __CCL_BCM_H_INCLUDED__
#define __CCL_BCM_H_INCLUDED__

CCL_BEGIN_DECLS

/* BCM correction */
// See Schneider & Teyssier (2015) for details of the model.
double ccl_bcm_model_fka(ccl_cosmology* cosmo, double k, double a, int* status);

CCL_END_DECLS

#endif
