#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

#define SIGMAM_TOLERANCE 3.0E-5

CTEST_DATA(sigmam) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double sigma8;
  double Neff;
  double* mnu;
  ccl_mnu_convention mnu_type;
  double Omega_v[5];
  double Omega_k[5];
  double w_0[5];
  double w_a[5];
};

CTEST_SETUP(sigmam) {
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->sigma8=0.8;
  data->n_s = 0.96;
  data->Neff=0;
  double mnuval = 0.;
  data->mnu=&mnuval;
  data-> mnu_type = ccl_mnu_sum;

  double Omega_v[5]={0.7, 0.7, 0.7, 0.65, 0.75};
  double w_0[5] = {-1.0, -0.9, -0.9, -0.9, -0.9};
  double w_a[5] = {0.0, 0.0, 0.1, 0.1, 0.1};

  for(int i=0;i<5;i++) {
    data->Omega_v[i] = Omega_v[i];
    data->w_0[i] = w_0[i];
    data->w_a[i] = w_a[i];
    data->Omega_k[i] = 1.0 - data->Omega_c - data->Omega_b - data->Omega_v[i];
  }
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

static void compare_sigmam(int i_model,struct sigmam_data * data)
{
  int nm,i;
  char fname[256],str[1024];
  char* rtn;
  FILE *f;
  int status=0;
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  ccl_parameters params = ccl_parameters_create(data->Omega_c,data->Omega_b,data->Omega_k[i_model-1],
						data->Neff, data->mnu, data->mnu_type,
						data->w_0[i_model-1],data->w_a[i_model-1],data->h,
						data->A_s,data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  params.T_CMB=2.7;
  params.sigma8=data->sigma8;
  params.Omega_g=0.;
  params.Omega_l=data->Omega_v[i_model-1];
  
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);

  sprintf(fname,"./tests/benchmark/model%d_sm.txt",i_model);
  f=fopen(fname,"r");
  if(f==NULL) {
    fprintf(stderr,"Error opening file %s\n",fname);
    exit(1);
  }
  nm=linecount(f)-1; rewind(f);
  
  rtn = fgets(str, 1024, f);
  for(i=0;i<nm;i++) {
    double m,m_h,sm_bench,sm_h,err;
    int stat;
    stat=fscanf(f,"%lf %lf",&m_h,&sm_bench);
    if(stat!=2) {
      fprintf(stderr,"Error reading file %s, line %d\n",fname,i+2);
      exit(1);
    }
    m=m_h/data->h;
    sm_h=ccl_sigmaM(cosmo,m,1.,&status);
    if (status) printf("%s\n",cosmo->status_message);
    err=sm_h/sm_bench-1;
    ASSERT_DBL_NEAR_TOL(err,0.,SIGMAM_TOLERANCE);
  }
  fclose(f);

  free(cosmo);
}

CTEST2(sigmam,model_1) {
  int model=1;
  compare_sigmam(model,data);
}

CTEST2(sigmam,model_2) {
  int model=2;
  compare_sigmam(model,data);
}

CTEST2(sigmam,model_3) {
  int model=3;
  compare_sigmam(model,data);
}
