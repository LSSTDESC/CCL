#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

// We allow a less stringent tolerance than usual here because we are 
// comparing to benchmarks produced with CAMB. At our current default 
// precision settings for CLASS, we only agree even in GR at 
// around 3e-3 
#define POWER_MG_TOL 5e-3

CTEST_DATA(power_MG) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double sigma8;
  double Neff;
  double mnuval;
  ccl_mnu_convention mnu_type;
  double Omega_k;
  double w0;
  double wa;
  double mu_0[5];
  double sigma_0[5];
};

CTEST_SETUP(power_MG) {
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->sigma8 = 0.8;
  data->n_s = 0.96;
  data->Neff=3.046;
  data->mnu_type =ccl_mnu_sum;
  data->mnuval = 0.;
  data->w0= -1.0;
  data->wa = 0.0;
  data-> Omega_k = 0.;
  
  double mu_0[5]={0., 0.1, -0.1, 0.1, -0.1};
  double sigma_0[5] = {0., 0.1, -0.1, -0.1, 0.1};

  for(int i=0;i<5;i++) {
    data->mu_0[i] = mu_0[i];
    data->sigma_0[i] = sigma_0[i];
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

static void compare_power_MG(int i_model,struct power_MG_data * data)
{
  int nk,i,j;
  int status=0;
  char fname[256], fname_nl[256], str[1024];
  FILE *f;
  ccl_configuration config_linear = default_config;
  config_linear.matter_power_spectrum_method = ccl_linear;
  
  ccl_parameters params;
  
  params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k,
		data->Neff, &(data->mnuval), data-> mnu_type, 
		data->w0, data->wa,  data->h, data->A_s, 
		data->n_s,-1,-1,-1,data->mu_0[i_model], data->sigma_0[i_model],-1,NULL,NULL, &status);

  ccl_cosmology * cosmo= ccl_cosmology_create(params, config_linear);
  ASSERT_NOT_NULL(cosmo);
  
  sprintf(fname,"./tests/benchmark/model%d_pk_MG.dat",i_model);
   f=fopen(fname,"r");
  if(f==NULL) {
    fprintf(stderr,"Error opening file %s\n",fname);
    exit(1);
  }
  nk=linecount(f)-1; rewind(f);
  
  double k=0.,pk_bench=0.,pk_ccl,err, k_h, pk_h;
  double z=0.; //Other redshift checks are possible 
  int stat=0;

	for(i=0;i<nk;i++) {      
    stat=fscanf(f,"%le %le\n",&k_h, &pk_h);
    if(stat!=2) {
      fprintf(stderr,"Error reading file %s, line %d\n",fname,i);
      exit(1);
    }
     
    // This is required if benchmark is in little h units. (Mpc/h etc)
    k=k_h*data->h;
    pk_bench=pk_h/pow(data->h,3);
    
    pk_ccl=ccl_linear_matter_power(cosmo,k,1./(1+z),&status);
    
    err=fabs(pk_ccl/pk_bench-1);
    ASSERT_DBL_NEAR_TOL(err,0.,POWER_MG_TOL);
    }
    
  fclose(f);

  ccl_cosmology_free(cosmo);

}

static void check_transfer_error(ccl_configuration config, struct power_MG_data * data)
{
  int status=0;
  
  ccl_parameters params;
  
  // Initialize ccl_cosmology struct
  params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k,
		data->Neff, &(data->mnuval), data-> mnu_type, 
		data->w0, data->wa,  data->h, data->A_s, 
		data->n_s,-1,-1,-1,data->mu_0[0], data->sigma_0[0],-1,NULL,NULL, &status);

  ccl_cosmology * cosmo= ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);
  
  // Call P(k) with unacceptable transfer function methods, check we get expected error.
  ccl_cosmology_compute_power(cosmo, &status);
  ASSERT_STR(cosmo->status_message, "ccl_power.c: ccl_cosmology_compute_power(): The power spectrum in the mu / Sigma modified gravity parameterisation is only implemented with the ccl_boltzmann_class power spectrum method.\n");
  
  ccl_cosmology_free(cosmo);
}

static void check_nonlin_error(struct power_MG_data * data)
{
  int status=0;
  
  ccl_configuration config = default_config;
  
  ccl_parameters params;
  
  // Initialize ccl_cosmology struct
  params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k,
		data->Neff, &(data->mnuval), data-> mnu_type, 
		data->w0, data->wa,  data->h, data->A_s, 
		data->n_s,-1,-1,-1,data->mu_0[0], data->sigma_0[0],-1,NULL,NULL, &status);

  ccl_cosmology * cosmo= ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);
  
  // Call P(k) trying to get the nonlinear power spectrum, check we get expected error.
  double k = 0.1;
  double a = 1.;
  double pk = ccl_nonlin_matter_power(cosmo, k, a, &status);
  ASSERT_STR(cosmo->status_message, "ccl_power.c: ccl_nonlin_matter_power(): Nonlinear behaviour for the mu / Sigma parameterization of modified gravity is not implemented. \n");
  
  ccl_cosmology_free(cosmo);
}

/*
CTEST2(power_MG, MG_emu_error) {

  ccl_configuration config_emu = {ccl_emulator, ccl_emu, ccl_nobaryons, ccl_tinker10, ccl_emu_strict};	
	
  check_transfer_error(config_emu, data);
}

CTEST2(power_MG, MG_eh_error) {

  ccl_configuration config_eh = {ccl_eisenstein_hu, ccl_linear, ccl_nobaryons, ccl_tinker10, ccl_emu_strict};	
	
  check_transfer_error(config_eh, data);
}

CTEST2(power_MG, MG_bbks_error) {

  ccl_configuration config_bbks = {ccl_bbks, ccl_linear, ccl_nobaryons, ccl_tinker10, ccl_emu_strict};	
	
  check_transfer_error(config_bbks, data);
}

CTEST2(power_MG, MG_nonlin_error) {
	
  check_nonlin_error(data);
}*/

CTEST2(power_MG, MG_pk_model0) {
  int model=0;	
  compare_power_MG(model,data);
}

/*CTEST2(power_MG, MG_pk_model1) {
  int model=1;	
  compare_power_MG(model,data);
}

/*
CTEST2(power_MG, MG_pk_model2) {
  int model=2;	
  compare_power_MG(model,data);
}

CTEST2(power_MG, MG_pk_model3) {
  int model=3;	
  compare_power_MG(model,data);
}

CTEST2(power_MG, MG_pk_model4) {
  int model=4;	
  compare_power_MG(model,data);
}*/

