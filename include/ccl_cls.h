/** @file */
#ifndef __CCL_CLS_H_INCLUDED__
#define __CCL_CLS_H_INCLUDED__

CCL_BEGIN_DECLS

double ccl_angular_cl_limber(ccl_cosmology *cosmo,
			     ccl_cl_tracer_collection_t *trc1,
			     ccl_cl_tracer_collection_t *trc2,
			     ccl_f2d_t *psp,double l,int *status);
double ccl_angular_cls_nonlimber(ccl_cosmology *cosmo,
				 double l_logstep,int l_linstep,
				 ccl_cl_tracer_collection_t *trc1,
				 ccl_cl_tracer_collection_t *trc2,
				 int nl_out,int *l_out,double *cl_out,
				 int *status);

CCL_END_DECLS


#endif
