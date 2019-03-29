#include "ccl.h"
#include "ctest.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

CTEST_DATA(corrs_3dRSD) {
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

CTEST_SETUP(corrs_3dRSD) {
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 0.8;
  data->n_s = 0.96;
  data->sigma8=0.8;
  data->Neff=3.046;
  double mnuval = 0.;
  data->mnu= &mnuval;
  data->mnu_type = ccl_mnu_sum;

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

static void compare_correlation_3dRSD(int i_model,struct corrs_3dRSD_data * data)
{
  int nk,nr,i,j;
  int status=0;
  double beta[3]={0.512796247727676,0.5107254761543404,0.5102500557881973};
  char fname[256],str[1024];
  char* rtn;
  FILE *f;
  ccl_configuration config = default_config;
  config.matter_power_spectrum_method= ccl_halofit;
  config.transfer_function_method = ccl_boltzmann_class;
  ccl_parameters params = ccl_parameters_create(data->Omega_c,data->Omega_b,data->Omega_k[i_model-1],
		data->Neff, data->mnu, data->mnu_type, data->w_0[i_model-1],data->w_a[i_model-1],
		data->h,data->A_s,data->n_s,-1, -1, -1, -1,NULL,NULL, &status);
  params.Omega_g=0.0;
  params.Omega_l=data->Omega_v[i_model-1];
  params.sigma8=data->sigma8;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);

  sprintf(fname,"./tests/benchmark/model%d_xiRSD.txt",i_model);
  f=fopen(fname,"r");
  if(f==NULL) {
    fprintf(stderr,"Error opening file %s\n",fname);
    exit(1);
  }
  nr=linecount(f)-1; rewind(f);

  // FIXME: these are not real standards
  // tolerence on abs difference in r^2 xi(r) for the range r = 0.1 - 100 Mpc (40 points in r) for z=0
  double CORR_TOLERANCE1 = 0.10;
  // tolerence on abs difference in r^2 xi(r) for the range r = 50 - 250 Mpc (100 points in r) for z=0
  double CORR_TOLERANCE2 = 0.10;

  int N1=40;
  double *r_arr1=malloc(N1*sizeof(double));
  double *r_arr2=malloc((nr-N1)*sizeof(double));
  double *ximm_bench_arr=malloc(nr*sizeof(double));

  rtn = fgets(str, 1024, f);
  for(i=0;i<nr;i++) {
    double r_h;
    int stat;
    stat=fscanf(f,"%lf",&r_h);
    if(stat!=1) {
      fprintf(stderr,"Error reading file %s, line %d\n",fname,i+2);
      exit(1);
    }

    if(i<N1)
      r_arr1[i]=r_h;
    else
      r_arr2[i-N1]=r_h;

    double ximm_bench;
    stat=fscanf(f,"%lf",&ximm_bench);
    if(stat!=1) {
	fprintf(stderr,"Error reading file %s, line %d\n",fname,i+2);
	exit(1);
      }

    ximm_bench_arr[i]=ximm_bench;
  }

  double *ximm_ccl_out1=malloc(N1*sizeof(double));
  double *ximm_ccl_out2=malloc((nr-N1)*sizeof(double));

  double z = j+0.;
  ccl_correlation_3dRsd_avgmu(cosmo,1.0,N1,r_arr1,beta[i_model-1],ximm_ccl_out1,&status);
  ccl_correlation_3dRsd_avgmu(cosmo,1.0,nr-N1,r_arr2,beta[i_model-1],ximm_ccl_out2,&status);

  if (status) printf("%s\n",cosmo->status_message);
  for(i=0;i<nr;i++){
  double err;

  if(i<N1){
  err=fabs(r_arr1[i]*r_arr1[i]*(ximm_ccl_out1[i]-ximm_bench_arr[i]));
  ASSERT_DBL_NEAR_TOL(0.,err,CORR_TOLERANCE1);
  }
  else{
  err=fabs(r_arr2[i-N1]*r_arr2[i-N1]*(ximm_ccl_out2[i-N1]-ximm_bench_arr[i]));
  ASSERT_DBL_NEAR_TOL(0.,err,CORR_TOLERANCE2);
  }
  }
  fclose(f);

  free(r_arr1);
  free(r_arr2);
  free(ximm_bench_arr);
  free(ximm_ccl_out1);
  free(ximm_ccl_out2);

  ccl_cosmology_free(cosmo);
}

CTEST2(corrs_3dRSD,model_1) {
  int model=1;
  compare_correlation_3dRSD(model,data);
}

CTEST2(corrs_3dRSD,model_2) {
  int model=2;
  compare_correlation_3dRSD(model,data);
}

CTEST2(corrs_3dRSD,model_3) {
  int model=3;
  compare_correlation_3dRSD(model,data);
}
