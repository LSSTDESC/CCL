#include "ccl.h"
#include "ccl_halomod.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

// the tolerance in dn/dm
#define HALOMOD_TOLERANCE 1e-2

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
  double k[160];
  double Delta2[160];
  
};

static void read_halomod_test_file(double k[160], double Delta2[160]){

  double spam;
  
  // Wavenumbers are k/h in benchmark data
  //FILE * f = fopen("./tests/benchmark/model1_halomod_fake.txt", "r");
  FILE * f = fopen("./tests/benchmark/model1_halomod_z0.txt", "r");
  ASSERT_NOT_NULL(f);

  // 
  for (int i=0; i<160; i++) {

    // Read in data from the benchmark file
    //int count = fscanf(f, "%le %le\n", &k[i], &Delta2[i]);
    int count = fscanf(f, "%le\t %le\t %le\t %le\t %le\n", &k[i], &spam, &spam, &spam, &Delta2[i]);
    //printf("%d\t %le\n", i, k[i]);
    
    // Check that we have enough columns
    ASSERT_EQUAL(5, count);
    
  }
  fclose(f);
}

// set up the cosmological parameters structure to be used in the test case
CTEST_SETUP(halomod){

  // only single model at this point
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9; // REMOVE THIS EVENTUALLY
  data->n_s = 0.96;
  data->sigma_8 = 0.8;
  data->Neff=0;
  double mnuval = 0.;
  data->mnu=&mnuval;
  data->mnu_type = ccl_mnu_sum;

  // REMOVE THIS EVENTUALLY
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
  //config.transfer_function_method = ccl_bbks;
  //config.mass_function_method = ccl_tinker;
  config.transfer_function_method = ccl_eisenstein_hu;
  config.mass_function_method = ccl_shethtormen;

  // Set the configuration
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

  // Check that the cosmology is assigned correctly
  ASSERT_NOT_NULL(cosmo);

  // Variables for the test
  double a = 1.0;
  //double logmass = 10;

  printf("\n");
  
  // compare to benchmark data
  for (int j=0; j<160; j++) {

    // Set variables inside loop
    //double mass = pow(10,logmass);
    //double sigma_j = ccl_sigmaM(cosmo, mass, a, status);
    double k = data->k[j]*params.h; //Convert the data k/h to pure k
    double Pk = 4.*M_PI*pow((k/(2.*M_PI)),3)*ccl_p_halomod(cosmo, k, a, status);
    double absolute_tolerance = HALOMOD_TOLERANCE*data->Delta2[j];

    // Do the check
    if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
    ASSERT_DBL_NEAR_TOL(data->Delta2[j], Pk, absolute_tolerance);

    printf("%d\t %le\t %le\t %le\t %lf\n", j, k, Pk, data->Delta2[j], Pk/data->Delta2[j]);

    //Increment the mass
    //logmass += 0.5;
  }
  free(cosmo);
}

CTEST2(halomod, model_1) {
  int model = 0;
  compare_halomod(model, data);
}
