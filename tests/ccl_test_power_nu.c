#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

#define POWER_NU_TOL 1.0E-4

CTEST_DATA(power_nu) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double sigma8;
  double Neff;
  double mnu0[3], mnu1[3], mnu2[3];
  ccl_mnu_convention mnu_type;
  double Omega_v[3];
  double Omega_k;
  double w_0[3];
  double w_a[3];
};

CTEST_SETUP(power_nu) {
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->sigma8=0.8;
  data->Neff=3.046;
  data->mnu_type =ccl_mnu_list;
  
  double mnu0[3]	= 	{0.04, 0., 0.};
  double mnu1[3]	= 	{0.05, 0.01, 0.};
  double mnu2[3]	= 	{0.03, 0.02, 0.04};
  
  data->mnu0[0] = mnu0[0];
  data->mnu1[0] = mnu1[0];
  data->mnu2[0] = mnu2[0];
  
  data->mnu0[1] = mnu0[1];
  data->mnu1[1] = mnu1[1];
  data->mnu2[1] = mnu2[1];
  
  data->mnu0[2] = mnu0[2];
  data->mnu1[2] = mnu1[2];
  data->mnu2[2] = mnu2[2];

  double Omega_v[3]={0.7, 0.7, 0.7};
  double w_0[3] = {-1.0, -0.9, -0.9};
  double w_a[3] = {0.0, 0.0, 0.1};

  for(int i=0;i<3;i++) {
    data->Omega_v[i] = Omega_v[i];
    data->w_0[i] = w_0[i];
    data->w_a[i] = w_a[i];
  }
  data-> Omega_k = 0.;
}

CTEST_DATA(power_nu_nl) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double sigma8;
  double Neff;
  double mnu0[3], mnu1[3], mnu2[3];
  ccl_mnu_convention mnu_type;
  double Omega_v[3];
  double Omega_k;
  double w_0[3];
  double w_a[3];
};

CTEST_SETUP(power_nu_nl) {
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->sigma8=0.8;
  data->Neff=3.046;
  data->mnu_type =ccl_mnu_list;
  
  double mnu0[3]	= 	{0.04, 0., 0.};
  double mnu1[3]	= 	{0.05, 0.01, 0.};
  double mnu2[3]	= 	{0.03, 0.02, 0.04};
  
  data->mnu0[0] = mnu0[0];
  data->mnu1[0] = mnu1[0];
  data->mnu2[0] = mnu2[0];
  
  data->mnu0[1] = mnu0[1];
  data->mnu1[1] = mnu1[1];
  data->mnu2[1] = mnu2[1];
  
  data->mnu0[2] = mnu0[2];
  data->mnu1[2] = mnu1[2];
  data->mnu2[2] = mnu2[2];

  double Omega_v[3]={0.7, 0.7, 0.7};
  double w_0[3] = {-1.0, -0.9, -0.9};
  double w_a[3] = {0.0, 0.0, 0.1};

  for(int i=0;i<3;i++) {
    data->Omega_v[i] = Omega_v[i];
    data->w_0[i] = w_0[i];
    data->w_a[i] = w_a[i];;
  }
  
  data->Omega_k = 0.;
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

static void compare_power_nu(int i_model,struct power_nu_data * data)
{
  int nk,i,j;
  int status=0;
  char fname[256], fname_nl[256], str[1024];
  FILE *f;
  ccl_configuration config_linear = default_config;
  config_linear.matter_power_spectrum_method = ccl_linear;
  
  ccl_parameters params;
  
  if (i_model==1){
  
      params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k,
						data->Neff, data->mnu0, data-> mnu_type, 
						data->w_0[i_model-1], data->w_a[i_model-1],
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  } else if (i_model==2){
	  params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k,
						data->Neff, data->mnu1, data->mnu_type,
						data->w_0[i_model-1], data->w_a[i_model-1],
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  } else if (i_model==3){
	 params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k,
						data->Neff, data->mnu2, data->mnu_type,
						data->w_0[i_model-1], data->w_a[i_model-1],
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  }

  ccl_cosmology * cosmo_linear = ccl_cosmology_create(params, config_linear);
  ASSERT_NOT_NULL(cosmo_linear);
  
  sprintf(fname,"./tests/benchmark/model%d_pk_nu.txt",i_model);
  f=fopen(fname,"r");
  if(f==NULL) {
    fprintf(stderr,"Error opening file %s\n",fname);
    exit(1);
  }
  nk=linecount(f)-1; rewind(f);
  
  double k=0.,pk_bench=0.,pk_ccl,err, k_h, pk_h;
  double z=0.; //Other redshift checks are possible but not currently implemented
  int stat=0;

	for(i=0;i<nk;i++) {      
    stat=fscanf(f,"%le %le\n",&k_h, &pk_h);
    if(stat!=2) {
      fprintf(stderr,"Error reading file %s, line %d\n",fname,i);
      exit(1);
    }
    k=k_h*data->h;
    pk_bench=pk_h/pow(data->h,3);
    
    pk_ccl=ccl_linear_matter_power(cosmo_linear,k,1./(1+z),&status);
    if (status) printf("%s\n",cosmo_linear->status_message);
    err=fabs(pk_ccl/pk_bench-1);
    ASSERT_DBL_NEAR_TOL(err,0.,POWER_NU_TOL);
    }
    
  fclose(f);

  ccl_cosmology_free(cosmo_linear);

}

static void compare_power_nu_nl(int i_model,struct power_nu_nl_data * data)
{
  int i,j, nk_nl;
  int status=0;
  char fname_nl[256], str[1024];
  FILE *f_nl;
  ccl_configuration config_nonlinear = default_config;
  config_nonlinear.matter_power_spectrum_method = ccl_halofit;
  
  ccl_parameters params;
  
  if (i_model==1){
  
      params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k,
						data->Neff, data->mnu0, data-> mnu_type, 
						data->w_0[i_model-1], data->w_a[i_model-1],
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  } else if (i_model==2){
	  params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k,
						data->Neff, data->mnu1, data->mnu_type,
						data->w_0[i_model-1], data->w_a[i_model-1],
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  } else if (i_model==3){
	 params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k,
						data->Neff, data->mnu2, data->mnu_type,
						data->w_0[i_model-1], data->w_a[i_model-1],
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  }
  
  ccl_cosmology * cosmo_nonlin = ccl_cosmology_create(params, config_nonlinear);
  ASSERT_NOT_NULL(cosmo_nonlin);
  
  sprintf(fname_nl,"./tests/benchmark/model%d_pk_nl_nu.txt",i_model);
  f_nl=fopen(fname_nl,"r");
  if(f_nl==NULL) {
    fprintf(stderr,"Error opening file %s\n",fname_nl);
    exit(1);
  }
  nk_nl=linecount(f_nl)-1; rewind(f_nl);
  
  double k=0.,pk_bench=0.,pk_ccl,err, k_h, pk_h;
  double z=0.; //Other redshift checks are possible but not currently implemented
  int stat=0;

  for(i=0;i<nk_nl;i++) {      
    stat=fscanf(f_nl,"%le %le\n",&k_h, &pk_h);
    if(stat!=2) {
      fprintf(stderr,"Error reading file %s, line %d\n",fname_nl,i);
      exit(1);
    }
    k=k_h*data->h;
    pk_bench=pk_h/pow(data->h,3);
    
    pk_ccl=ccl_nonlin_matter_power(cosmo_nonlin,k,1./(1+z),&status);
    if (status) printf("%s\n",cosmo_nonlin->status_message);
    err=fabs(pk_ccl/pk_bench-1);
    ASSERT_DBL_NEAR_TOL(err,0.,POWER_NU_TOL);
    }
 
  fclose(f_nl);

  ccl_cosmology_free(cosmo_nonlin);
}

CTEST2(power_nu,model_1) {
  int model=1;
  compare_power_nu(model,data);
}

CTEST2(power_nu,model_2) {
  int model=2;
  compare_power_nu(model,data);
}

CTEST2(power_nu,model_3) {
  int model=3;
  compare_power_nu(model,data);
}

CTEST2(power_nu_nl,model_1) {
  int model=1;
  compare_power_nu_nl(model,data);
}

CTEST2(power_nu_nl,model_2) {
  int model=2;
  compare_power_nu_nl(model,data);
}

CTEST2(power_nu_nl,model_3) {
  int model=3;
  compare_power_nu_nl(model,data);
}
