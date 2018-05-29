#include "ccl.h"
#include "ccl_halomod.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

// Relative error tolerance in the halomodel matter power spectrum
#define HALOMOD_TOLERANCE 1e-3

CTEST_DATA(halomod){

  // Cosmological parameters
  double Omega_c;
  double Omega_b;
  double Omega_k;
  double Neff;
  double* mnu;
  ccl_mnu_convention mnu_type;
  double w_0;
  double w_a;
  double h;
  double sigma_8;
  double n_s;
    
  // Arrays for power-spectrum data
  double k[2][160];
  double Delta2[2][160];
  
};

// Function to read in the benchmark data
static void read_halomod_test_file(double k[2][160], double Delta2[2][160]){

  // Variables for reading in unwanted stuff and file name
  double spam;
  char infile[256];

  // Loop over redshifts
  for (int i=0; i<2; i++){

    // File names for different redshifts
    if(i==0){strncpy(infile, "./tests/benchmark/model1_halomod_z0.txt", 256);}
    if(i==1){strncpy(infile, "./tests/benchmark/model1_halomod_z1.txt", 256);}
    
    // Open the files
    FILE * f = fopen(infile, "r");
    ASSERT_NOT_NULL(f);

    // Loop over wavenumbers, which  are k/h in benchmark data, power is Delta^2(k)
    for (int j=0; j<160; j++) {

      // Read in data from the benchmark file
      int count = fscanf(f, "%le\t %le\t %le\t %le\t %le\n", &k[i][j], &spam, &spam, &spam, &Delta2[i][j]);
    
      // Check that we have read in enough columns from the benchmark file
      ASSERT_EQUAL(5, count);
    
    }

    // Close the file
    fclose(f);
    
  }
  
}

// set up the cosmological parameters structure to be used in the test case
CTEST_SETUP(halomod){

  // Move the cosmological parameters to the data structure
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->Omega_k = 0.00;
  data->Neff = 0.00;
  double mnuval = 0.00;
  data->mnu = &mnuval;
  data->mnu_type = ccl_mnu_sum;
  data->w_0 = -1.00;
  data->w_a = 0.00;
  data->h = 0.7;
  data->sigma_8 = 0.8;
  data->n_s = 0.96;  

  // read the file of benchmark data
  read_halomod_test_file(data->k, data->Delta2);
  
}

// Function to actually do the comparison
static void compare_halomod(int model, struct halomod_data * data)
{

  // Status variables
  int stat = 0;
  int* status = &stat;

  // Set the cosmology
  ccl_parameters params = ccl_parameters_create(data->Omega_c, data->Omega_b,data->Omega_k,
						data->Neff, data->mnu, data->mnu_type, data->w_0,
						data->w_a, data->h,data->sigma_8, data->n_s,
						-1, -1, -1, -1, NULL, NULL, status);

  // Set the default configuration, but with Eisenstein & Hu linear P(k) and Sheth & Tormen mass function
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_eisenstein_hu;
  config.mass_function_method = ccl_shethtormen;

  // Set the configuration
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

  // Check that the cosmology is assigned correctly
  ASSERT_NOT_NULL(cosmo);

  // Loop over redshifts
  for (int i=0; i<2; i++) {

    // Variables for the test
    double a;
    
    if(i==0){a = 1.0;}
    if(i==1){a = 0.5;}
  
    // Loop over wavenumbers
    for (int j=0; j<160; j++) {

      // Set variables inside loop, convert CCL outputs to the same units as benchmark
      double k = data->k[i][j]*params.h; // Convert the benchmark data k/h to pure k
      double Pk = 4.*M_PI*pow((k/(2.*M_PI)),3)*ccl_p_halomod(cosmo, k, a, status); // Convert CCL P(k) -> benchmark Delta^2(k)
      double absolute_tolerance = HALOMOD_TOLERANCE*data->Delta2[i][j]; // Convert relative -> absolute tolerance

      // Do the check
      ASSERT_DBL_NEAR_TOL(data->Delta2[i][j], Pk, absolute_tolerance);

      // Write to screen to check
      //printf("%d\t %le\t %le\t %le\t %lf\n", j, k, Pk, data->Delta2[i][j], Pk/data->Delta2[i][j]);

    }

  }

  // Free the cosmology object
  free(cosmo);
  
}

// No idea what this is
CTEST2(halomod, model_1) {
  int model = 0;
  compare_halomod(model, data);
}
