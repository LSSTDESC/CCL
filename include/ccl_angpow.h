/** @file */
#ifdef __cplusplus
extern "C" {
#endif
  
#pragma once


void ccl_angular_cls_angpow(ccl_cosmology *ccl_cosmo,CCL_ClWorkspace *w,
				   CCL_ClTracer *clt1,CCL_ClTracer *clt2,
				   double *cl_out,int * status);

#ifdef __cplusplus
}
#endif
