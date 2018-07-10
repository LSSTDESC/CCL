#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

#define BCM_TOLERANCE 1e-4

CTEST_DATA(bcm) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double sigma8;
  double Omega_v[1];
  double Omega_k[1];
  double w_0[1];
  double w_a[1];
  double Neff;
  double* m_nu;
  ccl_mnu_convention mnu_type;
};

CTEST_SETUP(bcm) {
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.2e-9;
  data->n_s = 0.96;
  data->Neff = 3.046;
  double mnuval = 0.;
  data->m_nu= &mnuval;
  data-> mnu_type = ccl_mnu_sum;

  double Omega_v[1]={0.7};
  double w_0[1] = {-1.0};
  double w_a[1] = {0.0};

  for(int i=0;i<1;i++) {
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

static void compare_bcm(int i_model,struct bcm_data * data)
{
  int nk,i,j;
  int status =0;
  char fname[256],str[1024];
  char fname2[256];
  char* rtn;
  FILE *f,*f2;
  ccl_configuration config = default_config;
  config.baryons_power_spectrum_method=ccl_bcm;
  ccl_parameters params = ccl_parameters_create(data->Omega_c,data->Omega_b,data->Omega_k[i_model-1],
						data->Neff, data->m_nu, data-> mnu_type,
						data->w_0[i_model-1],data->w_a[i_model-1],
						data->h,data->A_s,data->n_s,14,-1,-1,-1,NULL,NULL, &status);
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);
  ccl_configuration config_nobar = default_config;
  ccl_parameters params_nobar = ccl_parameters_create(data->Omega_c,data->Omega_b,data->Omega_k[i_model-1],
						data->Neff, data->m_nu, data->mnu_type,
						data->w_0[i_model-1],data->w_a[i_model-1],
						data->h,data->A_s,data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  params.sigma8=data->sigma8;
  params.Omega_g=0;
  ccl_cosmology * cosmo_nobar = ccl_cosmology_create(params_nobar, config_nobar);
  ASSERT_NOT_NULL(cosmo_nobar);
  
  sprintf(fname,"./tests/benchmark/bcm/w_baryonspk_nl.dat");
  f=fopen(fname,"r");
  if(f==NULL) {
    fprintf(stderr,"Error opening file %s\n",fname);
    exit(1);
  }
  sprintf(fname2,"./tests/benchmark/bcm/wo_baryonspk_nl.dat");
  f2=fopen(fname2,"r");
  if(f2==NULL) {
    fprintf(stderr,"Error opening file %s\n",fname2);
    exit(1);
  }
  nk=linecount(f)-1; rewind(f);

  //Skip first 4 lines
  rtn = fgets(str, 1024, f);
  rtn = fgets(str, 1024, f);
  rtn = fgets(str, 1024, f);
  rtn = fgets(str, 1024, f);
  
  rtn = fgets(str, 1024, f2);
  rtn = fgets(str, 1024, f2);
  rtn = fgets(str, 1024, f2);
  rtn = fgets(str, 1024, f2);
  
  for(i=0;i<nk-4;i++) {
    double k_h,k;
    int stat;
    double psbar,psnobar,fbcm_bench,err;
    double psbar_bench,psnobar_bench;
    stat=fscanf(f,"%le %le",&k_h,&psbar);
    if(stat!=2) {
      fprintf(stderr,"Error reading file %s, line %d\n",fname,i+5);
      exit(1);
    }
    stat=fscanf(f2,"%*le %le",&psnobar);
    if(stat!=1) {
      fprintf(stderr,"Error reading file %s, line %d\n",fname2,i+5);
      exit(1);
    }
    k=k_h*data->h;
    //Check baryonic correction directly
    fbcm_bench=ccl_bcm_model_fkz(cosmo,k,1.,&status);
    if (status) printf("%s\n",cosmo->status_message);
    err=fabs(psbar/psnobar/fbcm_bench-1);
    ASSERT_DBL_NEAR_TOL(err,0.,BCM_TOLERANCE);
    //And check the ratio between power spectra
    psbar_bench=ccl_nonlin_matter_power(cosmo,k,1.,&status);
    if (status) printf("%s\n",cosmo->status_message);
    psnobar_bench=ccl_nonlin_matter_power(cosmo_nobar,k,1.,&status);
    if (status) printf("%s\n",cosmo_nobar->status_message);
    err=fabs(psbar/psnobar/(psbar_bench/psnobar_bench)-1);
    ASSERT_DBL_NEAR_TOL(err,0.,BCM_TOLERANCE);
    
  }
  fclose(f);
  fclose(f2);
  
  ccl_cosmology_free(cosmo);
}

CTEST2(bcm,model_1) {
  int model=1;
  compare_bcm(model,data);
}
