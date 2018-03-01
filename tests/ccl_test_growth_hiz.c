#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

// The tolerance in D(z) for all the
#define GROWTH_HIZ_TOLERANCE 1.0e-4

CTEST_DATA(growth_hiz) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double Neff;
  double* mnu;
  double Omega_v[3];
  double Omega_k[3];
  double w_0[3];
  double w_a[3];
  ccl_mnu_convention mnu_type;
  
  double z[7];
  double gf[3][7];
};

// Read the fixed format file containing all the growth factor
// benchmarks
static void read_growth_hiz_test_file(double z[7], double gf[3][7])
{
  //Growth is normalized to ~a at early times
  FILE * f = fopen("./tests/benchmark/growth_hiz_model1-3.txt", "r");
  ASSERT_NOT_NULL(f);
  
  // Ignore header line
  char str[1024];
  char* rtn;
  rtn = fgets(str, 1024, f);
  
    // File is fixed format - five rows and six columns
  for (int i=0; i<7; i++) {
    int count = fscanf(f, "%le %le %le %le\n", &z[i],
		       &gf[0][i], &gf[1][i], &gf[2][i]);
    	// Check that all the stuff in the benchmark is there
    ASSERT_EQUAL(4, count);
  }
  fclose(f);
}

// Set up the cosmological parameters to be used in each of the
// models
CTEST_SETUP(growth_hiz) {
  // Values that are the same for all 3 models
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->Neff=0;
  double mnuval = 0.;
  data->mnu= &mnuval;
  data->mnu_type = ccl_mnu_sum;
  
  
  // Values that are different for the different models
  double Omega_v[3] = {  0.7,  0.7,  0.7};
  double w_0[3]     = { -1.0, -0.9, -0.9};
  double w_a[3]     = {  0.0,  0.0,  0.1};
  
  // Fill in the values from these constant arrays.
  for (int i=0; i<3; i++) {
    data->Omega_v[i] = Omega_v[i];
    data->w_0[i]     = w_0[i];
    data->w_a[i]     = w_a[i];
    data->Omega_k[i] = 1.0 - data->Omega_c - data->Omega_b - data->Omega_v[i];
  }

  // The file of benchmark data.
  read_growth_hiz_test_file(data->z, data->gf);
}

static void compare_growth_hiz(int model, struct growth_hiz_data * data)
{
  int status=0; 	
  // Make the parameter set from the input data
  // Values of some parameters depend on the model index
  ccl_parameters params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k[model], data->Neff, data->mnu, data->mnu_type, data->w_0[model], data->w_a[model], data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  params.Omega_g=0;
  // Make a cosmology object from the parameters with the default configuration
  ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);
  ASSERT_NOT_NULL(cosmo);
  
  // Compare to benchmark data
  for (int j=0; j<7; j++) {
    double a = 1/(1.+data->z[j]);
    double gf_ij=ccl_growth_factor_unnorm(cosmo,a, &status);
    if (status) printf("%s\n",cosmo->status_message);
    double absolute_tolerance = GROWTH_HIZ_TOLERANCE*data->gf[model][j];
    if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
    ASSERT_DBL_NEAR_TOL(data->gf[model][j], gf_ij, absolute_tolerance);
  }

  ccl_cosmology_free(cosmo);
}

CTEST2(growth_hiz, model_1) {
  int model = 0;
  compare_growth_hiz(model, data);
}

CTEST2(growth_hiz, model_2) {
  int model = 1;
  compare_growth_hiz(model, data);
}

CTEST2(growth_hiz, model_3) {
  int model = 2;
  compare_growth_hiz(model, data);
}
