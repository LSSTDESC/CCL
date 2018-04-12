#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

// The tolerance in chi for all the
#define DISTANCES_HIZ_TOLERANCE 1.0e-3

CTEST_DATA(distances_hiz) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double Neff;
  double* mnu;
  ccl_mnu_convention mnu_type;
  double Omega_v[3];
  double Omega_k[3];
  double w_0[3];
  double w_a[3];
  
  double z[7];
  double chi[3][7];
};

// Read the fixed format file containing all the radial comoving
// distance benchmarks
static void read_chi_test_file(double z[7], double chi[3][7])
{
  //Distances are in Mpc/h
  FILE * f = fopen("./tests/benchmark/chi_hiz_model1-3.txt", "r");
  ASSERT_NOT_NULL(f);
  
  // Ignore header line
  char str[1024];
  fgets(str, 1024, f);
  
  // File is fixed format - five rows and six columns
  for (int i=0; i<7; i++) {
    int count = fscanf(f, "%le %le %le %le \n", &z[i],
		       &chi[0][i], &chi[1][i], &chi[2][i]);
    // Check that all the stuff in the benchmark is there
    ASSERT_EQUAL(4, count);
  }
  fclose(f);
}

// Set up the cosmological parameters to be used in each of the
// models
CTEST_SETUP(distances_hiz) {
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
  read_chi_test_file(data->z, data->chi);
}

static void compare_distances_hiz(int model, struct distances_hiz_data * data)
{
  int status=0;
  // Make the parameter set from the input data
  // Values of some parameters depend on the model index
  ccl_parameters params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k[model],
						data->Neff, data->mnu, data->mnu_type,
						data->w_0[model], data->w_a[model],
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  
  params.Omega_g=0;
  
  // Make a cosmology object from the parameters with the default configuration
  ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);
  ASSERT_NOT_NULL(cosmo);
  
  // Compare to benchmark data
  for (int j=0; j<7; j++) {
    double a = 1/(1.+data->z[j]);
    double chi_ij=ccl_comoving_radial_distance(cosmo,a, &status)*data->h;
    if (status) printf("%s\n",cosmo->status_message);
    double absolute_tolerance = DISTANCES_HIZ_TOLERANCE*data->chi[model][j];
    if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
    ASSERT_DBL_NEAR_TOL(data->chi[model][j], chi_ij, absolute_tolerance);
  }
  
  ccl_cosmology_free(cosmo);
}

CTEST2(distances_hiz, model_1) {
  int model = 0;
  compare_distances_hiz(model, data);
}

CTEST2(distances_hiz, model_2) {
  int model = 1;
  compare_distances_hiz(model, data);
}

CTEST2(distances_hiz, model_3) {
  int model = 2;
  compare_distances_hiz(model, data);
}
