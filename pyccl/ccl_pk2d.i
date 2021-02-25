%module ccl_pk2d

%{
/* put additional #include here */
%}

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* lkarr, int nk)};
%apply (double* IN_ARRAY1, int DIM1) {(double* aarr, int na)};
%apply (double* IN_ARRAY1, int DIM1) {(double* pkarr, int npk)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int ndout, double* doutput)};

%include "../include/ccl_f2d.h"
%include "../include/ccl_core.h"

%inline %{
ccl_f2d_t *set_pk2d_new_from_arrays(double* lkarr,int nk,
				    double* aarr,int na,
				    double* pkarr,int npk,
				    int order_lok,int order_hik,
				    int is_logp,
				    int *status)
{
  ccl_f2d_t *psp=ccl_f2d_t_new(na,aarr,nk,lkarr,pkarr,NULL,NULL,0,
			       order_lok,order_hik,ccl_f2d_cclgrowth,
			       is_logp,0,2,ccl_f2d_3,status);
  return psp;
}

void get_pk_spline_a(ccl_cosmology *cosmo,int ndout,double* doutput,int *status)
{
  ccl_get_pk_spline_a_array(cosmo,ndout,doutput,status);
}

void get_pk_spline_lk(ccl_cosmology *cosmo,int ndout,double* doutput,int *status)
{
  ccl_get_pk_spline_lk_array(cosmo,ndout,doutput,status);
}

double pk2d_eval_single(ccl_f2d_t *psp,double lk,double a,ccl_cosmology *cosmo,int *status)
{
  return ccl_f2d_t_eval(psp,lk,a,cosmo,status);
}

void pk2d_eval_multi(ccl_f2d_t *psp,double* lkarr,int nk,
		     double a,ccl_cosmology *cosmo,
		     int ndout,double *doutput,int *status)
{
  for(int ii=0;ii<ndout;ii++)
    doutput[ii]=ccl_f2d_t_eval(psp,lkarr[ii],a,cosmo,status);
}
%}
