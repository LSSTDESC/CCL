#include "ccl.h"
#include "ccl_massfunc.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

// the tolerance in dn/dm
#define MASSFUNC_TOLERANCE 1e-3

CTEST_DATA(massfunc) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double Omega_n;
  double Omega_v[1];
  double Omega_k[1];
  double w_0[1];
  double w_a[1];
  double sigma_8;

  double mass[9];
  double massfunc[7][9];
};

static void read_massfunc_test_file(double mass[9], double massfunc[7][9])
{
   // Masses are in Msun/h
   FILE * f = fopen("./tests/benchmark/mfunc.txt", "r");
   ASSERT_NOT_NULL(f);

   // Ignore header line
   char str[1024];
   fgets(str, 1024, f);

   // File is in fixed format - nine rows and eight columns
   for (int i=0; i<9; i++){
     int count = fscanf(f, "%le %le %le %le %le %le %le %le\n", &mass[i],
                        &massfunc[0][i], &massfunc[1][i], &massfunc[2][i],
                        &massfunc[3][i], &massfunc[4][i], &massfunc[5][i],
                        &massfunc[6][i]);
     // Check that all the stuff in the benchmark is there
     ASSERT_EQUAL(8, count);
   }
   fclose(f);
}

// set up the cosmological parameters to be used in the test case
CTEST_SETUP(massfunc){

  // only single model at tihs point
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->Omega_n =0.0;
  data->sigma_8 = 0.8;

  double Omega_v[1] = { 0.7 };
  double w_0[1]     = {-1.0 };
  double w_a[1]     = { 0.0 };

  for (int i=0; i<1; i++){
    data->Omega_v[i] = Omega_v[i];
    data->w_0[i] = w_0[i];
    data->w_a[i]= w_a[i];
    data->Omega_k[i] = 1.0 - data->Omega_c - data->Omega_b - data->Omega_n - data->Omega_v[i];
  }

  // read the file of benchmark data
  read_massfunc_test_file(data->mass, data->massfunc);
}

static void compare_massfunc(int model, struct massfunc_data * data)
{
  // make the parameter set from input data

  ccl_parameters params = ccl_parameters_create(data->Omega_c, data->Omega_b,
                                                data->Omega_k[model], data->Omega_n,
                                                data->w_0[model], data->w_a[model], data->h,
                                                data->A_s, data->n_s, -1,
                                                NULL, NULL);


  params.sigma_8 = data->sigma_8;
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  // test file generated using tinker 2008 currently
  config.mass_function_method = ccl_tinker;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  
  ASSERT_NOT_NULL(cosmo);

  double redshift = 0;
  double logmass = 11;

  // compare to benchmark data
  for (int j=0; j<9; j++){
    double mass = pow(10,logmass);
    redshift = 0;
    
    for (int i=0; i<7; i++){
      double massfunc_ij = ccl_massfunc(cosmo, mass/cosmo->params.h, redshift)/cosmo->params.h/cosmo->params.h/cosmo->params.h;
      //printf("%lf %lf %le %le\n", logmass, redshift, massfunc_ij, data->massfunc[i][j]);
      double absolute_tolerance = MASSFUNC_TOLERANCE*data->massfunc[i][j];
      if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
      ASSERT_DBL_NEAR_TOL(data->massfunc[i][j], massfunc_ij, absolute_tolerance);
      redshift += 0.2;
    }
    logmass += 0.5;
  }
  free(cosmo);
}

CTEST2(massfunc, model_1){
   int model = 0;
   compare_massfunc(model, data);
}
