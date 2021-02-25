%module ccl_tk3d

%{
/* put additional #include here */
%}

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* lkarr, int nk)};
%apply (double* IN_ARRAY1, int DIM1) {(double* aarr, int na)};
%apply (double* IN_ARRAY1, int DIM1) {(double* pk1arr, int npk1)};
%apply (double* IN_ARRAY1, int DIM1) {(double* pk2arr, int npk2)};
%apply (double* IN_ARRAY1, int DIM1) {(double* tkkarr, int ntkk)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int ndout, double* doutput)};

%include "../include/ccl_f2d.h"
%include "../include/ccl_core.h"

%inline %{
ccl_f3d_t *tk3d_new_from_arrays(double* lkarr,int nk,
                                double* aarr,int na,
                                double* tkkarr,int ntkk,
                                int order_lok,int order_hik,
                                int is_logp, int *status)
{
  ccl_f3d_t *tsp=ccl_f3d_t_new(na,aarr,nk,lkarr,tkkarr,NULL,NULL,0,
			       order_lok,order_hik,ccl_f2d_constantgrowth,
			       is_logp,1,4,ccl_f2d_3,status);
  return tsp;
}

ccl_f3d_t *tk3d_new_factorizable(double* lkarr,int nk,
                                 double* aarr,int na,
                                 double* pk1arr, int npk1,
                                 double* pk2arr, int npk2,
                                 int order_lok,int order_hik,
                                 int is_logp, int *status)
{
  ccl_f3d_t *tsp=ccl_f3d_t_new(na,aarr,nk,lkarr,NULL,pk1arr,pk2arr,1,
			       order_lok,order_hik,ccl_f2d_constantgrowth,
			       is_logp,1,4,ccl_f2d_3,status);
  return tsp;
}

double tk3d_eval_single(ccl_f3d_t *tsp,double lk,double a,int *status)
{
  ccl_a_finder *finda = ccl_a_finder_new_from_f3d(tsp);
  double tkk = ccl_f3d_t_eval(tsp,lk,lk,a,finda,NULL,status);
  ccl_a_finder_free(finda);
  return tkk;
}

void tk3d_eval_multi(ccl_f3d_t *tsp,double* lkarr,int nk,
		     double a,int ndout,double *doutput,
                     int *status)
{
  ccl_a_finder *finda = ccl_a_finder_new_from_f3d(tsp);
  for(int ii=0;ii<nk;ii++) {
    for(int jj=0;jj<nk;jj++) {
      doutput[jj+nk*ii]=ccl_f3d_t_eval(tsp,lkarr[jj],lkarr[ii],
                                       a,finda,NULL,status);
    }
  }
  ccl_a_finder_free(finda);
}
%}
