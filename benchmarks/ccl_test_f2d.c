#include "ccl.h"
#include "ctest.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

double growth_function(double a)
{
  return 1;
}

double k_function(double k)
{
  return pow(k/0.1,-1.);
}

double fka_model_analytical(double k,double a)
{
  return k_function(k)*growth_function(a)*growth_function(a);
}

CTEST_DATA(f2d) {
  int n_a;
  double *a_arr;
  int n_k;
  double *lk_arr;
  double *fk_arr;
  double *fa_arr;
  double *fka_arr;
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double sigma8;
};

CTEST_SETUP(f2d) {
  data->Omega_c = 0.30;
  data->Omega_b = 0.00;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->sigma8=0.8;
  data->n_s = 0.96;

  data->n_a=100;
  data->n_k=10;
  data->a_arr=malloc(data->n_a*sizeof(double));
  data->fa_arr=malloc(data->n_a*sizeof(double));
  data->lk_arr=malloc(data->n_k*sizeof(double));
  data->fk_arr=malloc(data->n_k*sizeof(double));
  for(int ii=0;ii<data->n_a;ii++) {
    data->a_arr[ii]=0.05+0.95*ii/(data->n_a-1.);
    data->fa_arr[ii]=2*log(growth_function(data->a_arr[ii]));
  }
  for(int ii=0;ii<data->n_k;ii++) {
    data->lk_arr[ii]=log(1E-4)+log(1E6)*(ii+0.5)/data->n_k;
    data->fk_arr[ii]=log(k_function(exp(data->lk_arr[ii])));
  }
  data->fka_arr=malloc(data->n_a*data->n_k*sizeof(double));
  for(int ii=0;ii<data->n_a;ii++) {
    for(int jj=0;jj<data->n_k;jj++)
      data->fka_arr[ii*data->n_k+jj]=log(fka_model_analytical(exp(data->lk_arr[jj]),data->a_arr[ii]));
  }
  //f(k,a)=(k/0.1)**-1*a**0.75
}

CTEST_TEARDOWN(f2d) {
  free(data->a_arr);
  free(data->lk_arr);
  free(data->fk_arr);
  free(data->fa_arr);
  free(data->fka_arr);
}

CTEST2(f2d,constant) {
  int status=0;
  ccl_f2d_t *psp;
  double fka;
  double lktest=-2.,atest=0.5;

  //Constant in k
  status=0;
  psp=ccl_f2d_t_new(data->n_a,data->a_arr,
		    -1,NULL,
		    NULL,NULL,data->fa_arr,
		    1,
		    2,
		    2,
		    ccl_f2d_constantgrowth,
		    1,0,2,
		    ccl_f2d_3,
		    &status);
  ASSERT_TRUE(status==0);

  fka=ccl_f2d_t_eval(psp,lktest,atest,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,fka/(growth_function(atest)*growth_function(atest)));
  ccl_f2d_t_free(psp);

  //Constant in a
  status=0;
  psp=ccl_f2d_t_new(-1,NULL,
		    data->n_k,data->lk_arr,
		    NULL,data->fk_arr,NULL,
		    1,
		    2,
		    2,
		    ccl_f2d_constantgrowth,
		    1,0,2,
		    ccl_f2d_3,
		    &status);
  ASSERT_TRUE(status==0);

  fka=ccl_f2d_t_eval(psp,lktest,atest,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,fka/k_function(exp(lktest)));
  ccl_f2d_t_free(psp);

  //Constant in k and a
  status=0;
  psp=ccl_f2d_t_new(-1,NULL,
		    -1,NULL,
		    NULL,NULL,NULL,
		    0,
		    2,
		    2,
		    ccl_f2d_constantgrowth,
		    1,0,2,
		    ccl_f2d_3,
		    &status);
  ASSERT_TRUE(status==0);

  fka=ccl_f2d_t_eval(psp,lktest,atest,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,fka);

  ccl_f2d_t_free(psp);
}

CTEST2(f2d,a_overflow) {
  int status=0;
  ccl_f2d_t *psp;
  double fka;
  double lktest=-2.,atest=0.5;

  //Populate properly
  status=0;
  data->a_arr[data->n_a-1]=1.;
  psp=ccl_f2d_t_new(data->n_a,data->a_arr,
  		    data->n_k,data->lk_arr,
  		    data->fka_arr,
		    NULL, NULL, 0,
  		    2, //extrap_lok
		    2, //extrap_hik
  		    ccl_f2d_constantgrowth, //extrap_growth
  		    1, //is_fka_log
  		    0,2,
  		    ccl_f2d_3,
  		    &status);
  ASSERT_TRUE(status==0);

  //Check that the function stays constant above a=1
  fka=ccl_f2d_t_eval(psp,lktest,1.1,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(ccl_f2d_t_eval(psp,lktest,1.,NULL,&status),fka);
  ccl_f2d_t_free(psp);
}

CTEST2(f2d,sanity) {
  int status=0;
  ccl_f2d_t *psp;
  double fka;
  double lktest=-2.,atest=0.5;
  double alo=0.02;

  //Now populate properly
  status=0;
  data->a_arr[data->n_a-1]=1.;
  psp=ccl_f2d_t_new(data->n_a,data->a_arr,
  		    data->n_k,data->lk_arr,
  		    data->fka_arr,
		    NULL, NULL, 0,
  		    2, //extrap_lok
		    2, //extrap_hik
  		    ccl_f2d_constantgrowth, //extrap_growth
  		    1, //is_fka_log
  		    1,2,
  		    ccl_f2d_3,
  		    &status);
  ASSERT_TRUE(status==0);

  //Now put some sensible numbers within the redshift and k range
  fka=ccl_f2d_t_eval(psp,lktest,atest,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,fka/fka_model_analytical(exp(lktest),atest));

  //Evaluate at very high z and see if it checks out
  fka=ccl_f2d_t_eval(psp,lktest,alo,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,fka/fka_model_analytical(exp(lktest),alo));

  //Evaluate at very high k and see if it checks out
  double lkhi=data->lk_arr[data->n_k-1]*1.1;
  fka=ccl_f2d_t_eval(psp,lkhi,atest,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,fka/fka_model_analytical(exp(lkhi),atest));

  //Evaluate at very low k and see if it checks out
  double lklo=data->lk_arr[0]/1.1;
  fka=ccl_f2d_t_eval(psp,lklo,atest,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,fka/fka_model_analytical(exp(lklo),atest));

  ccl_f2d_t_free(psp);
}

CTEST2(f2d,factorize) {
  int status=0;
  ccl_f2d_t *psp;
  double fka;
  double lktest=-2.,atest=0.5;
  double alo=0.02;

  //Now populate properly
  status=0;
  data->a_arr[data->n_a-1]=1.;
  psp=ccl_f2d_t_new(data->n_a,data->a_arr,
  		    data->n_k,data->lk_arr,
  		    NULL,
		    data->fk_arr,data->fa_arr,1,
  		    2, //extrap_lok
		    2, //extrap_hik
  		    ccl_f2d_constantgrowth, //extrap_growth
  		    1, //is_fka_log
  		    1,2,
  		    ccl_f2d_3,
  		    &status);
  ASSERT_TRUE(status==0);

  //Now put some sensible numbers within the redshift and k range
  fka=ccl_f2d_t_eval(psp,lktest,atest,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,fka/fka_model_analytical(exp(lktest),atest));

  //Evaluate at very high z and see if it checks out
  fka=ccl_f2d_t_eval(psp,lktest,alo,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,fka/fka_model_analytical(exp(lktest),alo));

  //Evaluate at very high k and see if it checks out
  double lkhi=data->lk_arr[data->n_k-1]*1.1;
  fka=ccl_f2d_t_eval(psp,lkhi,atest,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,fka/fka_model_analytical(exp(lkhi),atest));

  //Evaluate at very low k and see if it checks out
  double lklo=data->lk_arr[0]/1.1;
  fka=ccl_f2d_t_eval(psp,lklo,atest,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,fka/fka_model_analytical(exp(lklo),atest));

  ccl_f2d_t_free(psp);
}

CTEST2(f2d,pk) {
  int status=0;
  ccl_f2d_t *psp;
  double fka;
  double lktest=-2.,atest=0.5;
  double alo=0.02;

  //Now verify that things scale with the CCL growth factor as intended
  //First initialize the cosmology object
  double gz;
  double mnu=0.;
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  config.matter_power_spectrum_method = ccl_linear;
  ccl_parameters params = ccl_parameters_create(data->Omega_c,data->Omega_b, 0.0, 3.046, &mnu,
                                                0, -1.0, 0.0, data->h, data->A_s,data->n_s,
                                                -1, -1, -1, 0., 0., 1.0, 1.0, 0.0, 
						-1, NULL, NULL, &status);
  params.T_CMB=2.7;
  params.Omega_k=0;
  params.Omega_g=0;
  params.Omega_nu_rel=0;
  params.Omega_l = 1.0 - params.Omega_m;
  params.sigma8=data->sigma8;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);
  //Compute growth factor to earliest redshift
  ccl_cosmology_compute_growth(cosmo, &status);
  ASSERT_TRUE(status==0);
  gz=ccl_growth_factor(cosmo,alo,&status)/ccl_growth_factor(cosmo,data->a_arr[0],&status);

  //Initialize f2d struct
  status=0;
  data->a_arr[data->n_a-1]=1.;
  psp=ccl_f2d_t_new(data->n_a,data->a_arr,
  		    data->n_k,data->lk_arr,
  		    data->fka_arr,
		    NULL, NULL, 0,
  		    2, //extrap_lok
		    2, //extrap_hik
  		    ccl_f2d_cclgrowth, //extrap_growth
  		    1, //is_fka_log
  		    0,2,
  		    ccl_f2d_3,
  		    &status);
  ASSERT_TRUE(status==0);

  //Evaluate at very low z and see if it checks out
  double fka0=ccl_f2d_t_eval(psp,lktest,data->a_arr[0],NULL,&status);
  ASSERT_TRUE(status==0);
  fka=ccl_f2d_t_eval(psp,lktest,alo,cosmo,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,fka/(fka0*gz*gz));

  ccl_f2d_t_free(psp);

  ccl_parameters_free(&(cosmo->params));
  ccl_cosmology_free(cosmo);
}
