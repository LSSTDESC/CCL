/** @file */
#ifndef __CCL_CLS_H_INCLUDED__
#define __CCL_CLS_H_INCLUDED__

CCL_BEGIN_DECLS


/**
 * Computes Limber power spectrum for two different tracers at a given ell.
 * @param cosmo Cosmological parameters
 * @param trc1 a ccl_cl_tracer_collection_t containing a bunch of individual contributions.
 * @param trc2 a ccl_cl_tracer_collection_t containing a bunch of individual contributions.
 * @param psp the p2d_t object representing the 3D power spectrum to integrate over.
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
 * @param psp the p2d_t object representing the 3D power spectrum to integrate over.
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

/**
 * Computes non-Gaussian Limber power spectrum covariance for four different
 * tracers at a given (ell1,ell2) pair.
 * @param cosmo Cosmological parameters
 * @param trc1 a ccl_cl_tracer_collection_t containing a bunch of individual contributions.
 * @param trc2 a ccl_cl_tracer_collection_t containing a bunch of individual contributions.
 * @param trc3 a ccl_cl_tracer_collection_t containing a bunch of individual contributions.
 * @param trc4 a ccl_cl_tracer_collection_t containing a bunch of individual contributions.
 * @param tsp the t3d_t object representing the 3D connected trispectrum to integrate over.
 * @param nl1_out number of multipoles on which the covariance will be calculated along the first dimension.
 * @param l1_out multipole values on which the first dimension of the covariance will be calculated.
 * @param nl2_out number of multipoles on which the covariance will be calculated along the second dimension.
 * @param l2_out multipole values on which the second dimension of the covariance will be calculated.
 * @param cov_out will hold the calculated power spectrum values.
 * @param integration_method method for integration over chi (spline or QAG/QUAD).
 * @param chi_exponent exponent of the 1/chi^alpha factor in the Limber integral.
 * @param kernel_extra additional chi-dependent multiplicative factor entering the Limber integral.
 * @param prefactor_extra final constant factor multiplying the whole covariance.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 */
void ccl_angular_cl_covariance(ccl_cosmology *cosmo,
                               ccl_cl_tracer_collection_t *trc1,
                               ccl_cl_tracer_collection_t *trc2,
                               ccl_cl_tracer_collection_t *trc3,
                               ccl_cl_tracer_collection_t *trc4,
                               ccl_f3d_t *tsp,
                               int nl1_out, double *l1_out,
                               int nl2_out, double *l2_out,
                               double *cov_out,
                               ccl_integration_t integration_method,
                               int chi_exponent, ccl_f1d_t *kernel_extra,
                               double prefactor_extra, int *status);
CCL_END_DECLS
#endif
