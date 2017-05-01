#pragma once

int ccl_tracer_corr(ccl_cosmology *cosmo, int n_theta, double **theta, CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel,bool taper_cl,double *taper_cl_limits, double **corr_func);

int ccl_tracer_corr_fftlog(ccl_cosmology *cosmo, int n_theta, double **theta,
		     CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel,
		     bool taper_cl,double *taper_cl_limits,double **corr_func,
		     double (*angular_cl)(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,
					  CCL_ClTracer *clt2, int * status) );

int ccl_tracer_corr_legendre(ccl_cosmology *cosmo, int n_theta, double **theta,
                     CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel,
                     bool taper_cl,double *taper_cl_limits,double **corr_func,
                     double (*angular_cl)(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,
                                          CCL_ClTracer *clt2, int * status) );
