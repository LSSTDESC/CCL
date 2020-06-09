/** @file */
#ifndef __CCL_CLS_H_INCLUDED__
#define __CCL_CLS_H_INCLUDED__

CCL_BEGIN_DECLS


/**
 * Computes Limber power spectrum for two different tracers at a given ell.
 * @param cosmo Cosmological parameters
 * @param trc1 a ccl_cl_tracer_collection_t containing a bunch of individual contributions.
 * @param trc2 a ccl_cl_tracer_collection_t containing a bunch of individual contributions.
 * @param psp the p2d_t object representing the 3D power spectrum to integrate over. Pass null to use the non-linear matter power spectrum.
 * @param nl_out number of multipoles on which the power spectrum will be calculated.
 * @param l_out multipole values on which the power spectrum will be calculated.
 * @param cl_out will hold the calculated power spectrum values.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 */
void ccl_angular_cls_limber(ccl_cosmology *cosmo,
       ccl_cl_tracer_collection_t *trc1,
       ccl_cl_tracer_collection_t *trc2,
       ccl_f2d_t *psp,
       int nl_out, double *l_out, double *cl_out,
       ccl_integration_t integration_method,
       int *status);
/**
 * Computes non-Limber power spectrum for two different tracers at a given ell.
 * @param cosmo Cosmological parameters
 * @param trc1 a ccl_cl_tracer_collection_t containing a bunch of individual contributions.
 * @param trc2 a ccl_cl_tracer_collection_t containing a bunch of individual contributions.
 * @param psp the p2d_t object representing the 3D power spectrum to integrate over. Pass null to use the non-linear matter power spectrum.
 * @param nl_out number of multipoles on which the power spectrum will be calculated.
 * @param l_out multipole values on which the power spectrum will be calculated.
 * @param cl_out will hold the calculated power spectrum values.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 */
void ccl_angular_cls_nonlimber(ccl_cosmology *cosmo,
         ccl_cl_tracer_collection_t *trc1,
         ccl_cl_tracer_collection_t *trc2,
         ccl_f2d_t *psp,
         int nl_out,int *l_out,double *cl_out,
         int *status);

CCL_END_DECLS
#endif
