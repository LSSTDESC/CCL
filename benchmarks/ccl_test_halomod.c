#include "ccl.h"
#include "ccl_halomod.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

// Relative error tolerance in the halomodel matter power spectrum
#define HALOMOD_TOLERANCE 1E-3
#define NUMK 256

// Data structure for the CTEST
CTEST_DATA(halomod){

  // Cosmological parameters
  double Omega_c[3];
  double Omega_b[3];
  double Omega_k;
  double Neff;
  double* mnu;
  ccl_mnu_convention mnu_type;
  double w_0;
  double w_a;
  double h[3];
  double sigma_8[3];
  double n_s[3];

  // Arrays for power-spectrum data
  double k[3][2][NUMK];
  double Pk[3][2][NUMK];

};

// Function to read in the benchmark data
static void read_halomod_test_file(double k[3][2][NUMK], double Pk[3][2][NUMK]){

  // Variables for reading in unwanted stuff and file name
  double spam;
  char infile[256];

  // Loop over cosmological models
  for (int model=0; model<3; model++){

    // Loop over redshifts
    for (int i=0; i<2; i++){

      // File names for different redshifts and cosmological models
      if(model==0 && i==0){strncpy(infile, "./benchmarks/data/pk_hm_c1_z0.txt", 256);}
      if(model==0 && i==1){strncpy(infile, "./benchmarks/data/pk_hm_c1_z1.txt", 256);}
      if(model==1 && i==0){strncpy(infile, "./benchmarks/data/pk_hm_c2_z0.txt", 256);}
      if(model==1 && i==1){strncpy(infile, "./benchmarks/data/pk_hm_c2_z1.txt", 256);}
      if(model==2 && i==0){strncpy(infile, "./benchmarks/data/pk_hm_c3_z0.txt", 256);}
      if(model==2 && i==1){strncpy(infile, "./benchmarks/data/pk_hm_c3_z1.txt", 256);}

      // Open the file
      FILE * f = fopen(infile, "r");
      ASSERT_NOT_NULL(f);

      // Loop over wavenumbers, which  are k/h in benchmark data, power is Delta^2(k)
      for (int j=0; j<NUMK; j++) {

	// Read in data from the benchmark file
	int count = fscanf(f, "%le\t %le\t %le\t %le\t %le\n", &k[model][i][j], &spam, &spam, &spam, &Pk[model][i][j]);

	// Check that we have read in enough columns from the benchmark file
	ASSERT_EQUAL(5, count);

      }

      // Close the file
      fclose(f);

    }

  }

}

// set up the cosmological parameters structure to be used in the test case
CTEST_SETUP(halomod){

  // Move the cosmological parameters to the data structure
  data->Omega_k = 0.00;
  data->Neff = 0.00;
  double mnuval = 0.00;
  data->mnu = &mnuval;
  data->mnu_type = ccl_mnu_sum;
  data->w_0 = -1.00;
  data->w_a = 0.00;

  // Cosmological parameters that are different for different tests
  double Omega_c[3] = { 0.2500, 0.2265, 0.2685 };
  double Omega_b[3] = { 0.0500, 0.0455, 0.0490 };
  double h[3]       = { 0.7000, 0.7040, 0.6711 };
  double sigma_8[3] = { 0.8000, 0.8100, 0.8340 };
  double n_s[3]     = { 0.9600, 0.9670, 0.9624 };

  // Fill in the values from these constant arrays
  for (int model=0; model<3; model++){
    data->Omega_c[model] = Omega_c[model];
    data->Omega_b[model] = Omega_b[model];
    data->h[model]       = h[model];
    data->sigma_8[model] = sigma_8[model];
    data->n_s[model]     = n_s[model];
  }

  // read the file of benchmark data
  read_halomod_test_file(data->k, data->Pk);

}

// Function to actually do the comparison
static void compare_halomod(int model, struct halomod_data * data)
{

  // Status variables
  int stat = 0;
  int* status = &stat;

  // Set the cosmology
  ccl_parameters params = ccl_parameters_create(data->Omega_c[model], data->Omega_b[model],data->Omega_k,
						data->Neff, data->mnu, data->mnu_type, data->w_0,
						data->w_a, data->h[model],data->sigma_8[model], data->n_s[model],
						-1, -1, -1, -1, NULL, NULL, status);

  // Set the default configuration, but with Eisenstein & Hu linear P(k) and Sheth & Tormen mass function and Duffy (2008) halo concentrations
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_eisenstein_hu;
  config.matter_power_spectrum_method = ccl_halo_model;
  config.mass_function_method = ccl_shethtormen;
  config.halo_concentration_method = ccl_duffy2008;

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
    for (int j=0; j<NUMK; j++) {

      // Set variables inside loop, convert CCL outputs to the same units as benchmark
      double k = data->k[model][i][j]*params.h; // Convert the benchmark data k/h to pure k
      double Pk = data->Pk[model][i][j]/pow(params.h,3); // Convert the benchmark data Pk units to remove factors of h

      double Pk_ccl = ccl_halomodel_matter_power(cosmo, k, a, status); // Get CCL P(k)
      double absolute_tolerance = HALOMOD_TOLERANCE*Pk; // Convert relative -> absolute tolerance

      // Do the check
      ASSERT_DBL_NEAR_TOL(Pk, Pk_ccl, absolute_tolerance);

    }

  }

  // Free the cosmology object
  free(cosmo);

}

CTEST2(halomod, model_1) {
  int model = 0;
  compare_halomod(model, data);
}

CTEST2(halomod, model_2) {
  int model = 1;
  compare_halomod(model, data);
}

CTEST2(halomod, model_3) {
  int model = 2;
  compare_halomod(model, data);
}
