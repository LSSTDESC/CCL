#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

// Relative error tolerance in the halo profile density
#define HALOPROFILE_TOLERANCE 1E-3
#define NUMR 256

// Data structure for the CTEST
CTEST_DATA(haloprofile){

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

  // Arrays for halo profile data
  double R[4][NUMR];
  double RESULT[4][NUMR];

};

// Function to read in the benchmark data
static void read_haloprofile_test_file(double R[4][NUMR], double RESULT[4][NUMR]){

    char infile[256];

    // Masses are in Msun/h
    for (int model=0; model<4; model++){
        if (model==0) {strncpy(infile, "./benchmarks/data/haloprofile_nfw_colossus.txt", 256);}
        if (model==1) {strncpy(infile, "./benchmarks/data/haloprofile_projected_nfw_colossus.txt", 256);}
        if (model==2) {strncpy(infile, "./benchmarks/data/haloprofile_einasto_colossus.txt", 256);}
        if (model==3) {strncpy(infile, "./benchmarks/data/haloprofile_hernquist_colossus.txt", 256);}
        // Open the file
        FILE * f = fopen(infile, "r");
        ASSERT_NOT_NULL(f);

        // Ignore header line
        char str[1024];
        char* rtn;
        rtn = fgets(str, 1024, f);

        // file is in fixed format - R, RESULT, w/ NUMR rows
        for (int i=0; i<NUMR; i++) {
            int count = fscanf(f, "%le %le\n", &R[model][i], &RESULT[model][i]);
            // Check that all the stuff in the benchmark is there
            ASSERT_EQUAL(2, count);
        }
        fclose(f);
    }
}

// set up the cosmological parameters structure to be used in the test case
CTEST_SETUP(haloprofile){

  // Move the cosmological parameters to the data structure
  data->Omega_k = 0.00;
  data->Neff = 3.046;
  double mnuval = 0.00;
  data->mnu = &mnuval;
  data->mnu_type = ccl_mnu_sum;
  data->w_0 = -1.00;
  data->w_a = 0.00;
  data->Omega_c = 0.2603;
  data->Omega_b = 0.0486;
  data->h = 0.6774;
  data->n_s = 0.9667;
  data->sigma_8 = 0.8159;

  // read the file of benchmark data
  read_haloprofile_test_file(data->R, data->RESULT);

}

// Function to actually do the comparison
static void compare_haloprofile(int model, struct haloprofile_data * data)
{

  // Status variables
  int stat = 0;
  int* status = &stat;

  // Set the cosmology
  ccl_parameters params = ccl_parameters_create(data->Omega_c, data->Omega_b,data->Omega_k,
						data->Neff, data->mnu, data->mnu_type, data->w_0,
						data->w_a, data->h,data->sigma_8, data->n_s,
						-1, -1, -1, -1, NULL, NULL, status);

  // Set the default configuration, but with Eisenstein & Hu linear P(k) and Sheth & Tormen mass function and Duffy (2008) halo concentrations
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_eisenstein_hu;
  config.mass_function_method = ccl_shethtormen;

  // Set the configuration
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

  // Check that the cosmology is assigned correctly
  ASSERT_NOT_NULL(cosmo);

  double a = 1.0;
  double concentration = 5;
  double halomass = 6E13;
  double halomassdef = 200;
  double rmin = 0.01;
  double rmax = 100;
  double* result;
  double* r;
  r = malloc(NUMR*sizeof(double));
  result = malloc(NUMR*sizeof(double));
  // compare to benchmark data
  for (int j=0; j<NUMR; j++) {
      r[j] = exp(log(rmin)+log(rmax/rmin)*j/(NUMR-1.));
  }
  if (model==0) {
      ccl_halo_profile_nfw(cosmo, concentration, halomass, halomassdef, a, r, NUMR, result, status);
  }
  if (model==1) {
      ccl_projected_halo_profile_nfw(cosmo, concentration, halomass, halomassdef, a, r, NUMR, result, status);
  }
  else if (model==2) {
      ccl_halo_profile_einasto(cosmo, concentration, halomass, halomassdef, a, r, NUMR, result, status);
  }
  else if (model==3) {
      ccl_halo_profile_hernquist(cosmo, concentration, halomass, halomassdef, a, r, NUMR, result, status);
  }
  for (int j=0; j<NUMR; j++) {
      double absolute_tolerance = HALOPROFILE_TOLERANCE*data->RESULT[model][j];
      if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
      ASSERT_DBL_NEAR_TOL(data->RESULT[model][j], result[j], absolute_tolerance);
  }

  free(cosmo);
}

CTEST2(haloprofile, model_1) {
  int model = 0;
  compare_haloprofile(model, data);
}

CTEST2(haloprofile, model_2) {
  int model = 1;
  compare_haloprofile(model, data);
}

CTEST2(haloprofile, model_3) {
  int model = 2;
  compare_haloprofile(model, data);
}

CTEST2(haloprofile, model_4) {
  int model = 3;
  compare_haloprofile(model, data);
}
