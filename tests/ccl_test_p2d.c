#include "ccl.h"
#include "ctest.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

double growth_function(double a)
{
  return pow(a,0.375);
}

double pk_model_analytical(double k,double a)
{
  return pow(k/0.1,-1.)*growth_function(a)*growth_function(a);
}

CTEST_DATA(p2d) {
  int n_a;
  double *a_arr;
  int n_k;
  double *lk_arr;
  double *pk_arr;
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double sigma8;
};

CTEST_SETUP(p2d) {
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
  for(int ii=0;ii<data->n_a;ii++)
    data->a_arr[ii]=0.05+0.95*ii/(data->n_a-1.);
  for(int ii=0;ii<data->n_k;ii++)
    data->lk_arr[ii]=log(1E-4)+log(1E6)*(ii+0.5)/data->n_k;
  data->pk_arr=malloc(data->n_a*data->n_k*sizeof(double));
  for(int ii=0;ii<data->n_a;ii++) {
    for(int jj=0;jj<data->n_k;jj++)
      data->pk_arr[ii*data->n_k+jj]=log(pk_model_analytical(exp(data->lk_arr[jj]),data->a_arr[ii]));
  }
  //P(k,a)=(k/0.1)**-1*a**0.75
}

CTEST_TEARDOWN(p2d) {
  free(data->a_arr);
  free(data->lk_arr);
  free(data->pk_arr);
}

CTEST2(p2d,sanity) {
  int status=0;
  ccl_p2d_t *psp;
  double pk;

  //First check that if we do not populate the P(k) all the way to z=0 we get an error
  data->a_arr[data->n_a-1]=1.1;
  psp=ccl_p2d_t_new(data->n_a,data->a_arr,
		    data->n_k,data->lk_arr,
		    data->pk_arr,
		    0, //extrap_lok
		    2, //extrap_hik
		    ccl_p2d_constantgrowth, //extrap_growth
		    1, //is_pk_log
		    NULL,0,
		    ccl_p2d_3,
		    &status);
  ASSERT_TRUE(status);
  ccl_p2d_t_free(psp);

  //Now populate properly
  status=0;
  data->a_arr[data->n_a-1]=1.;
  psp=ccl_p2d_t_new(data->n_a,data->a_arr,
  		    data->n_k,data->lk_arr,
  		    data->pk_arr,
  		    2, //extrap_lok
		    2, //extrap_hik
  		    ccl_p2d_customgrowth, //extrap_growth
  		    1, //is_pk_log
  		    growth_function,0,
  		    ccl_p2d_3,
  		    &status);
  ASSERT_TRUE(status==0);

  //Get an error if we evaluate above a=1
  double lktest=-2.,atest=0.5;
  pk=ccl_p2d_t_eval(psp,lktest,1.1,NULL,&status);
  ASSERT_TRUE(status);
  ASSERT_DBL_NEAR(-1.,pk);
  status=0;

  //Now put some sensible numbers within the redshift and k range
  pk=ccl_p2d_t_eval(psp,lktest,atest,NULL,&status); 
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,pk/pk_model_analytical(exp(lktest),atest));

  //Evaluate at very low z and see if it checks out
  double alo=0.02;
  pk=ccl_p2d_t_eval(psp,lktest,alo,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,pk/pk_model_analytical(exp(lktest),alo));

  //Evaluate at very high k and see if it checks out
  double lkhi=data->lk_arr[data->n_k-1]*1.1;
  pk=ccl_p2d_t_eval(psp,lkhi,atest,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,pk/pk_model_analytical(exp(lkhi),atest));

  //Evaluate at very high k and see if it checks out
  double lklo=data->lk_arr[0]*1.1;
  pk=ccl_p2d_t_eval(psp,lklo,atest,NULL,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,pk/pk_model_analytical(exp(lklo),atest));

  ccl_p2d_t_free(psp);

  //Now verify that things scale with the CCL growth factor as intended
  //First initialize the cosmology object
  double gz;
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  config.matter_power_spectrum_method = ccl_linear;
  ccl_parameters params = ccl_parameters_create_flat_lcdm(data->Omega_c,data->Omega_b,data->h,
							  data->A_s,data->n_s, &status);
  params.Omega_k=0;
  params.Omega_g=0;
  params.Omega_n_rel=0;
  params.Omega_l = 1.0 - params.Omega_m;
  params.sigma8=data->sigma8;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);
  //Compute growth factor to earliest redshift
  gz=ccl_growth_factor(cosmo,alo,&status)/ccl_growth_factor(cosmo,data->a_arr[0],&status);

  //Initialize p2d struct
  status=0;
  data->a_arr[data->n_a-1]=1.;
  psp=ccl_p2d_t_new(data->n_a,data->a_arr,
  		    data->n_k,data->lk_arr,
  		    data->pk_arr,
  		    2, //extrap_lok
		    2, //extrap_hik
  		    ccl_p2d_cclgrowth, //extrap_growth
  		    1, //is_pk_log
  		    NULL,0,
  		    ccl_p2d_3,
  		    &status);
  ASSERT_TRUE(status==0);

  //Evaluate at very low z and see if it checks out
  double pk0=ccl_p2d_t_eval(psp,lktest,data->a_arr[0],NULL,&status);
  ASSERT_TRUE(status==0);
  pk=ccl_p2d_t_eval(psp,lktest,alo,cosmo,&status);
  ASSERT_TRUE(status==0);
  ASSERT_DBL_NEAR(1,pk/(pk0*gz*gz));

  ccl_p2d_t_free(psp);
  
  ccl_cosmology_free(cosmo);
}
