%module ccl_p2d

%{
/* put additional #include here */
%}

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* lkarr, int nk)};
%apply (double* IN_ARRAY1, int DIM1) {(double* aarr, int na)};
%apply (double* IN_ARRAY1, int DIM1) {(double* pkarr, int npk)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int ndout, double* doutput)};

%include "../include/ccl_p2d.h"
%include "../include/ccl_core.h"
%include "../include/ccl_params.h"

%inline %{
ccl_p2d_t *set_p2d_new_from_arrays(double* lkarr,int nk,
				   double* aarr,int na,
				   double* pkarr,int npk,
				   int *status)
{
  ccl_p2d_t *psp=ccl_p2d_t_new(na,aarr,nk,lkarr,pkarr,1,2,ccl_p2d_cclgrowth,1,NULL,0,ccl_p2d_3,status);
  return psp;
}

void get_pk_spline_a(int ndout,double* doutput,int *status)
{
  ccl_get_pk_spline_a_array(ndout,doutput,status);
}

void get_pk_spline_lk(int ndout,double* doutput,int *status)
{
  ccl_get_pk_spline_lk_array(ndout,doutput,status);
}
 
%}
