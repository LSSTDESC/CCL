#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

// The tolerance in D(z) 
#define GROWTH_HIZ_TOLERANCE 6.0e-6

CTEST_DATA(growth_hiz) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double Neff;
  double* mnu;
  double Omega_v[7];
  double Omega_k[7];
  double w_0[7];
  double w_a[7];
  ccl_mnu_convention mnu_type;
  double mu_0[7];
  double sigma_0[7];
  
  double z[7];
  double gf[7][7];
};

// Read the fixed format file containing all the growth factor
// benchmarks
static void read_growth_hiz_test_file(double z[7], double gf[3][7])
{
	
  // First get orginal benchmark for growth in non-modified-gravity models:
   	
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
  
  /*// Now get the growth in the models with non-zero mu_0 and sigma_0
  
  FILE * fMG = fopen("./tests/benchmark/growth_hiz_model4-7_MG.txt", "r");
  ASSERT_NOT_NULL(fMG);
  
  // Ignore header line
  char strMG[1024];
  char* rtnMG;
  rtnMG = fgets(strMG, 1024, fMG);
  
    // File is fixed format - five rows and six columns
  for (int i=0; i<7; i++) {
    int count = fscanf(f, "%le %le %le %le %le\n", &z[i],
		       &gf[3][i], &gf[4][i], &gf[5][i], &gf[7][i]);
    	// Check that all the stuff in the benchmark is there
    ASSERT_EQUAL(5, count);
  }
  fclose(fMG);*/
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
  double Omega_v[7] = {  0.7,  0.7,  0.7, 0.7, 0.7, 0.7, 0.7};
  double w_0[7]     = { -1.0, -0.9, -0.9, -0.9, -0.9 -0.9, -0.9};
  double w_a[7]     = {  0.0,  0.0,  0.1, 0., 0., 0., 0. };
  double mu_0[7]    = { 0., 0., 0., 0.1, -0.1, 0.1, -0.1};
  double sigma_0[7] = {0., 0., 0., 0.1, -0.1, -0.1, 0.1};
  
  // Fill in the values from these constant arrays.
  for (int i=0; i<7; i++) {
    data->Omega_v[i] = Omega_v[i];
    data->w_0[i]     = w_0[i];
    data->w_a[i]     = w_a[i];
    data->Omega_k[i] = 1.0 - data->Omega_c - data->Omega_b - data->Omega_v[i];
    data->mu_0[i] = mu_0[i];
    data->sigma_0[i] = sigma_0[i];
  }

  // The file of benchmark data.
  read_growth_hiz_test_file(data->z, data->gf);
}

static void compare_growth_hiz(int model, struct growth_hiz_data * data)
{
  int status=0; 	
  // Make the parameter set from the input data
  // Values of some parameters depend on the model index
  ccl_parameters params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k[model], data->Neff, data->mnu, data->mnu_type, data->w_0[model], data->w_a[model], data->h, data->A_s, data->n_s,-1,-1,-1,data->mu_0[model], data->sigma_0[model],-1,NULL,NULL, &status);
  params.Omega_g=0; //enforce no radiation
  params.Omega_l = 1.-params.Omega_m-params.Omega_k; //reomcpute Omega_l without radiation
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

/*CTEST2(growth_hiz, model_4) {
  int model = 3;
  compare_growth_hiz(model, data);
}
CTEST2(growth_hiz, model_5) {
  int model = 4;
  compare_growth_hiz(model, data);
}
CTEST2(growth_hiz, model_6) {
  int model = 5;
  compare_growth_hiz(model, data);
}
CTEST2(growth_hiz, model_7) {
  int model = 6;
  compare_growth_hiz(model, data);
}*/

