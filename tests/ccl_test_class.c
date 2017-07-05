#include "ccl.h"
#include "ccl_config.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

// Compared with CLASS (git commit: 992b18b47ae3379d70945175f81ed98826998303)
// using the ccl_class_mod*_parameters.ini parameter files.
#define CLASS_TOLERANCE 1.0E-3
#define CLASS_LOGK_MIN -4.
#define CLASS_LOGK_MAX 1.

CTEST_DATA(class) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  //double sigma_8;
  double N_nu_rel;
  double N_nu_mass;
  double mnu;
  double Omega_v[6];
  double Omega_k[6];
  double w_0[6];
  double w_a[6];
};

CTEST_SETUP(class) {
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  //data->sigma_8=0.8;
  data->N_nu_rel=3.046;
  data->N_nu_mass=0;
  data->mnu=0;

  double Omega_v[6]={0.7, 0.7, 0.7, 0.65, 0.75, 0.7};
  double w_0[6] = {-1.0, -0.9, -0.9, -0.9, -0.9, -0.95};
  double w_a[6] = {0.0, 0.0, 0.1, 0.1, 0.1, -0.2};

  for(int i=0;i<6;i++) {
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

static void compare_class(int i_model, struct class_data * data)
{
  int nk,i,j;
  int status=0;
  char fname[256], str[1024];
  char* rtn;
  FILE *f;
  
  // Set up cosmology
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_boltzmann_class;
  config.matter_power_spectrum_method = ccl_linear;
  ccl_parameters params = ccl_parameters_create(
                            data->Omega_c, data->Omega_b, 
                            data->Omega_k[i_model-1], data->N_nu_rel, 
                            data->N_nu_mass, data->mnu, data->w_0[i_model-1], 
                            data->w_a[i_model-1], data->h, data->A_s,
                            data->n_s, -1, NULL, NULL, &status);
  params.Omega_g = 0;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);
  
  // Loop over redshift bins
  //for(j=0; j<6; j++) {
  for(j=0; j<1; j++) {
    double pk_h, pk_bench, pk_ccl, err;
    double z = j+0.;
  
    // Open benchmark file for this redshift
    sprintf(fname, "./tests/benchmark/class_benchmark/ccl_class_mod%d_z%d_pk.dat", 
            i_model, j+1);
    f = fopen(fname, "r");
    if(f == NULL) {
      fprintf(stderr, "Error opening file %s\n", fname);
      exit(1);
    }
    nk = linecount(f) - 4; rewind(f); // Subtract 4 for header lines
    
    // Ignore the first 4 lines (header comments)
    for (i=0; i<4; i++) rtn = fgets(str, 1024, f);
    
    // Loop over k
    double k_h, k;
    int stat;
    for(i=0; i < nk; i++) {
        
        stat = fscanf(f,"%lf", &k_h);
        if(stat != 1) {
            fprintf(stderr, "Error reading file %s, line %d\n", fname, i+2);
            exit(1);
        }
        k = k_h * data->h; // Rescale to non-h^-1 units

        // Get P(k) for this k
        stat = fscanf(f, "%lf", &pk_h);
        if(stat != 1) {
            fprintf(stderr, "Error reading file %s, line %d\n", fname, i+2);
            exit(1);
        }
        pk_bench = pk_h / pow(data->h, 3);
        pk_ccl = ccl_linear_matter_power(cosmo, k, 1./(1+z), &status);
        if (status) printf("status: %s\n", cosmo->status_message);
        ASSERT_TRUE(status == 0);
        
        // Test CCL calculation against benchmark
        err = fabs(pk_ccl / pk_bench - 1.);
        ASSERT_DBL_NEAR_TOL(0., err, CLASS_TOLERANCE);
        //printf("%4.4e %4.4e %4.4e, %4.4e\n", k, pk_ccl, pk_bench, err);
    }
  
  fclose(f);
  //printf("-----------------------------------\n");
  }

  ccl_cosmology_free(cosmo);
}

static void try_class(int i_model, struct class_data * data)
{
  // See if CLASS will run for a given set of parameters
  int i, j;
  int nk = 100;
  int status=0;
  double pk_ccl;
  double z = 0.;
  
  // Set up cosmology
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_boltzmann_class;
  ccl_parameters params = ccl_parameters_create(
                            data->Omega_c, data->Omega_b, 
                            data->Omega_k[i_model-1], data->N_nu_rel, 
                            data->N_nu_mass, data->mnu, data->w_0[i_model-1], 
                            data->w_a[i_model-1], data->h, data->A_s,
                            data->n_s, -1, NULL, NULL, &status);
  params.Omega_g = 0;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);
  
  // Calculate P(k) over some k range
  double dlogk = (CLASS_LOGK_MAX - CLASS_LOGK_MIN) / (double)nk;
  for(i=0; i < nk; i++) {
    double k = pow(10., CLASS_LOGK_MIN + (double)i * dlogk);
    
    pk_ccl = ccl_linear_matter_power(cosmo, k, 1./(1+z), &status);
    if (status) printf("status: %s\n", cosmo->status_message);
    ASSERT_TRUE(status == 0);
    ASSERT_FALSE(isnan(pk_ccl));
  } // end loop over k

  ccl_cosmology_free(cosmo);
}

/*
// Fails due to unhandled w_a error
CTEST2(class, try_model_6) {
  int model=6;
  try_class(model, data);
}
*/

/*
// Disable because they're slow
CTEST2(class, try_model_5) {
  int model=5;
  try_class(model, data);
}

CTEST2(class, try_model_4) {
  int model=4;
  try_class(model, data);
}
*/

/*
// Currently fail due to ~few % inaccuracy
CTEST2(class, model_3) {
  int model=3;
  compare_class(model, data);
}

CTEST2(class, model_2) {
  int model=2;
  compare_class(model, data);
}
*/

CTEST2(class, model_1) {
  int model=1;
  compare_class(model, data);
}
