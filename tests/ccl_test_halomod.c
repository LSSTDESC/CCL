#include "ccl.h"
#include "ccl_halomod.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

// the tolerance in dn/dm
#define HALOMOD_TOLERANCE 1e-4

CTEST_DATA(halomod){

  // Cosmological parameters
  double Omega_c;
  double Omega_b;
  double Omega_k[1];
  double Neff;
  double* mnu;
  ccl_mnu_convention mnu_type;
  double w_0[1];
  double w_a[1];
  double h;
  double A_s;
  double n_s;

  //Derived parameters;
  double Omega_v[1];
  double sigma_8;
  
  // Arrays for power-spectrum data
  double k[10];
  double Delta2[10];
  
};

static void read_halomod_test_file(double k[10], double Delta2[10]){
  
  // Masses are in Msun/h
  FILE * f = fopen("./tests/benchmark/model1_halomod_fake.txt", "r");
  ASSERT_NOT_NULL(f);

  // file is in fixed format - logmass, sigma, invsigma, and hmf, w/ 13 rows
  for (int i=0; i<10; i++) {
    int count = fscanf(f, "%le %le\n", &k[i], &Delta2[i]);
    // Check that all the stuff in the benchmark is there
    ASSERT_EQUAL(2, count);
  }
  fclose(f);
}

// set up the cosmological parameters to be used in the test case
CTEST_SETUP(halomod){

  // only single model at this point
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->sigma_8 = 0.8;
  data->Neff=0;
  double mnuval = 0.;
  data->mnu=&mnuval;
  data->mnu_type = ccl_mnu_sum;

  double Omega_v[1] = { 0.7 };
  double w_0[1]     = {-1.0 };
  double w_a[1]     = { 0.0 };
  for (int i=0; i<1; i++) {
    data->Omega_v[i] = Omega_v[i];
    data->w_0[i] = w_0[i];
    data->w_a[i]= w_a[i];
    data->Omega_k[i] = 1.0 - data->Omega_c - data->Omega_b - data->Omega_v[i];
  }

  // read the file of benchmark data
  read_halomod_test_file(data->k, data->Delta2);
}

static void compare_halomod(int model, struct halomod_data * data)
{

  // Status
  int stat = 0;
  int* status = &stat;

  // Set the cosmology
  ccl_parameters params = ccl_parameters_create(data->Omega_c, data->Omega_b,data->Omega_k[model],
						data->Neff, data->mnu, data->mnu_type, data->w_0[model],
						data->w_a[model], data->h,data->A_s, data->n_s,
						-1, -1, -1, -1, NULL, NULL, status);

  // sigma_8 is required for BBKS
  params.sigma_8 = data->sigma_8;

  // Set the default configuration, but with BBKS P(k) and Tinker mass function
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  config.mass_function_method = ccl_tinker;

  // Set the configuration
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

  // Check that the cosmology is assigned correctly
  ASSERT_NOT_NULL(cosmo);

  // Variables for the test
  double a = 1.0;
  double logmass = 10;

  printf("\n");
  
  // compare to benchmark data
  for (int j=0; j<10; j++) {

    // Set variables inside loop
    double mass = pow(10,logmass);
    double sigma_j = ccl_sigmaM(cosmo, mass, a, status);
    double k = 1.;
    //double Pk = ccl_p_halomod(cosmo, k, a, status);
    double absolute_tolerance = HALOMOD_TOLERANCE*data->Delta2[j];

    // Do the check
    if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
    ASSERT_DBL_NEAR_TOL(data->Delta2[j], sigma_j, absolute_tolerance);

    printf("%d\t %f\t %f\t %f\n", j, mass, sigma_j, data->Delta2[j]);

    //Increment the mass
    logmass += 0.5;
  }
  free(cosmo);
}

CTEST2(halomod, model_1) {
  int model = 0;
  compare_halomod(model, data);
}
