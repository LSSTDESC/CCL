#include "ccl.h"
#include "ctest.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


double a_func(double a)
{
  return a;
}

double k_func(double k)
{
  return pow(k/0.1,-1.);
}

double fka1_model(double k, double a)
{
  return k_func(k)*pow(a_func(a), 2);
}

double fka2_model(double k, double a)
{
  return pow(k_func(k),1.1)*pow(a_func(a), 2);
}

double tkka_model(double k1, double k2,double a)
{
  return fka1_model(k1, a)*fka2_model(k2, a);
}

CTEST_DATA(f3d) {
  int n_a;
  double *a_arr;
  int n_k;
  double *lk_arr;
  double *fka1_arr;
  double *fka2_arr;
  double *tkka_arr;
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double sigma8;
};

CTEST_SETUP(f3d) {
  data->Omega_c = 0.30;
  data->Omega_b = 0.00;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->sigma8=0.8;
  data->n_s = 0.96;

  data->n_a=100;
  data->n_k=10;
  data->a_arr=malloc(data->n_a*sizeof(double));
  data->lk_arr=malloc(data->n_k*sizeof(double));
  data->fka1_arr=malloc(data->n_a*data->n_k*sizeof(double));
  data->fka2_arr=malloc(data->n_a*data->n_k*sizeof(double));

  for(int ii=0;ii<data->n_a;ii++)
    data->a_arr[ii]=0.05+0.95*ii/(data->n_a-1.);

  for(int ii=0;ii<data->n_k;ii++)
    data->lk_arr[ii]=log(1E-4)+log(1E6)*(ii+0.5)/data->n_k;

  for(int ii=0;ii<data->n_a;ii++) {
    for(int jj=0;jj<data->n_k;jj++) {
      data->fka1_arr[ii*data->n_k+jj]=log(fka1_model(exp(data->lk_arr[jj]),data->a_arr[ii]));
      data->fka2_arr[ii*data->n_k+jj]=log(fka2_model(exp(data->lk_arr[jj]),data->a_arr[ii]));
    }
  }

  data->tkka_arr=malloc(data->n_a*data->n_k*data->n_k*sizeof(double));
  for(int ii=0;ii<data->n_a;ii++) {
    for(int jj=0;jj<data->n_k;jj++) {
      for(int kk=0;kk<data->n_k;kk++)
        data->tkka_arr[kk+data->n_k*(jj+data->n_k*ii)]=log(tkka_model(exp(data->lk_arr[kk]),
                                                                     exp(data->lk_arr[jj]),
                                                                     data->a_arr[ii]));
    }
  }
}

CTEST_TEARDOWN(f3d) {
  free(data->a_arr);
  free(data->lk_arr);
  free(data->fka1_arr);
  free(data->fka2_arr);
  free(data->tkka_arr);
}

CTEST2(f3d,baseline) {
  int status=0;
  ccl_f3d_t *tsp;
  ccl_a_finder *finda;
  double tkka;
  double lktest=-2.,atest=0.5;

  status=0;
  tsp=ccl_f3d_t_new(data->n_a,data->a_arr,
                    data->n_k,data->lk_arr,
                    data->tkka_arr, NULL, NULL,
                    0, 1, 1,
                    ccl_f2d_constantgrowth,
                    1, 1, 4, ccl_f2d_3,
                    &status);
  ASSERT_TRUE(status==0);
  finda=ccl_a_finder_new_from_f3d(tsp);
  ASSERT_TRUE(status==0);

  tkka=ccl_f3d_t_eval(tsp,lktest,lktest,atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(exp(lktest),exp(lktest),atest));

  //Extrapolation in a
  tkka=ccl_f3d_t_eval(tsp,lktest,lktest,0.01,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(exp(lktest),exp(lktest),0.05));

  //Extrapolation in k (low, #1)
  double klow=1E-5;
  tkka=ccl_f3d_t_eval(tsp,log(klow),lktest,atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(klow,exp(lktest),atest));

  //Extrapolation in k (low, #2)
  tkka=ccl_f3d_t_eval(tsp,lktest, log(klow),atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(exp(lktest),klow,atest));

  //Extrapolation in k (low, #1 and #2)
  tkka=ccl_f3d_t_eval(tsp,log(klow), log(klow),atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(klow,klow,atest));

  //Extrapolation in k (high, #1)
  double khigh=1E3;
  tkka=ccl_f3d_t_eval(tsp,log(khigh),lktest,atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(khigh,exp(lktest),atest));

  //Extrapolation in k (high, #2)
  tkka=ccl_f3d_t_eval(tsp,lktest, log(khigh),atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(exp(lktest),khigh,atest));

  //Extrapolation in k (high, #1 and #2)
  tkka=ccl_f3d_t_eval(tsp,log(khigh), log(khigh),atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(khigh,khigh,atest));

  ccl_a_finder_free(finda);
  ccl_f3d_t_free(tsp);
}

CTEST2(f3d,factorized) {
  int status=0;
  ccl_f3d_t *tsp;
  ccl_a_finder *finda;
  double tkka;
  double lktest=-2.,atest=0.5;

  status=0;
  tsp=ccl_f3d_t_new(data->n_a,data->a_arr,
                    data->n_k,data->lk_arr,
                    NULL, data->fka1_arr, data->fka2_arr,
                    1, 1, 1,
                    ccl_f2d_constantgrowth,
                    1, 1, 4, ccl_f2d_3,
                    &status);
  ASSERT_TRUE(status==0);
  finda=ccl_a_finder_new_from_f3d(tsp);
  ASSERT_TRUE(status==0);

  tkka=ccl_f3d_t_eval(tsp,lktest,lktest,atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(exp(lktest),exp(lktest),atest));

  //Extrapolation in a
  tkka=ccl_f3d_t_eval(tsp,lktest,lktest,0.01,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(exp(lktest),exp(lktest),0.05));

  //Extrapolation in k (low, #1)
  double klow=1E-5;
  tkka=ccl_f3d_t_eval(tsp,log(klow),lktest,atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(klow,exp(lktest),atest));

  //Extrapolation in k (low, #2)
  tkka=ccl_f3d_t_eval(tsp,lktest, log(klow),atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(exp(lktest),klow,atest));

  //Extrapolation in k (low, #1 and #2)
  tkka=ccl_f3d_t_eval(tsp,log(klow), log(klow),atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(klow,klow,atest));

  //Extrapolation in k (high, #1)
  double khigh=1E3;
  tkka=ccl_f3d_t_eval(tsp,log(khigh),lktest,atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(khigh,exp(lktest),atest));

  //Extrapolation in k (high, #2)
  tkka=ccl_f3d_t_eval(tsp,lktest, log(khigh),atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(exp(lktest),khigh,atest));

  //Extrapolation in k (high, #1 and #2)
  tkka=ccl_f3d_t_eval(tsp,log(khigh), log(khigh),atest,finda,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,tkka/tkka_model(khigh,khigh,atest));

  ccl_a_finder_free(finda);
  ccl_f3d_t_free(tsp);
}
