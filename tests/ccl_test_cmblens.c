#include "ccl.h"
#include "../include/ccl_params.h"
#include "ctest.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define CLS_TOLERANCE 1E-3
#define CLS_FRACTION 1E-3

CTEST_DATA(cls) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double sigma_8;
};

CTEST_SETUP(cls) {
  data->Omega_c = 0.30;
  data->Omega_b = 0.00;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->sigma_8=0.8;
  data->n_s = 0.96;
}

static int linecount(FILE *f)
{
  //////
  // Counts #lines from file
  int i0=0;
  char ch[1000];
  while((fgets(ch,sizeof(ch),f))!=NULL) {
    i0++;
  }
  return i0;
}

static void compare_cls(struct cls_data * data)
{
  int status=0;
  char fname[256];
  double factor_tol=3.;
  double zlss=1100.;
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_log_cl_cc.txt");

  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  config.matter_power_spectrum_method = ccl_linear;
  ccl_parameters params = ccl_parameters_create_flat_lcdm(data->Omega_c,data->Omega_b,data->h,
							  data->A_s,data->n_s, &status);
  params.sigma_8=data->sigma_8;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);

  //Fix spline parameters for the high-redshift needs of the CMB lensing power spectrum
  ccl_splines->A_SPLINE_NA=10000;
  ccl_splines->A_SPLINE_NA_PK=500;

  FILE *fi_cc;
  CCL_ClTracer *tr_cl=ccl_cl_tracer_cmblens_new(cosmo,zlss,&status);
  ASSERT_NOT_NULL(tr_cl);
  fi_cc=fopen(fname,"r"); ASSERT_NOT_NULL(fi_cc);
  double fraction_failed=0;
  for(int ii=0;ii<3001;ii++) {
    int l, rtn;
    double cl_cc,cl_cc_h;
    rtn = fscanf(fi_cc,"%d %lf",&l,&cl_cc);
    cl_cc_h=ccl_angular_cl(cosmo,l,tr_cl,tr_cl,&status);
    if (status) printf("%s\n",cosmo->status_message);
    if(fabs(cl_cc_h/cl_cc-1)>factor_tol*CLS_TOLERANCE) {
      fraction_failed++;
    }
  }
  fclose(fi_cc);

  fraction_failed/=3001;
  printf("%lf %% ",fraction_failed*100);
  ASSERT_TRUE((fraction_failed<CLS_FRACTION));

  ccl_cl_tracer_free(tr_cl);
  ccl_cosmology_free(cosmo);
}

CTEST2(cls,cmblens) {
  compare_cls(data);
}
