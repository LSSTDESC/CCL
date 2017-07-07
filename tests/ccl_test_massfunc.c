#include "ccl.h"
#include "ccl_massfunc.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

// the tolerance in dn/dm
#define SIGMA_TOLERANCE 1e-4
#define INVSIGMA_TOLERANCE 5e-3
#define MASSFUNC_TOLERANCE 5e-3

CTEST_DATA(massfunc) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double N_nu_rel;
  double N_nu_mass;
  double mnu;
  double Omega_v[1];
  double Omega_k[1];
  double w_0[1];
  double w_a[1];
  double sigma_8;

  double mass[13];
  double massfunc[3][13];
};

static void read_massfunc_test_file(double mass[13], double massfunc[3][13])
{
   // Masses are in Msun/h
   FILE * f = fopen("./tests/benchmark/model1_hmf.txt", "r");
   ASSERT_NOT_NULL(f);

   // Ignore header line
   char str[1024];
   char* rtn;
   rtn = fgets(str, 1024, f);

   // file is in fixed format - logmass, sigma, invsigma, and hmf, w/ 13 rows
   for (int i=0; i<13; i++) {
     int count = fscanf(f, "%le %le %le %le\n", &mass[i],
                        &massfunc[0][i], &massfunc[1][i], &massfunc[2][i]);
     // Check that all the stuff in the benchmark is there
     ASSERT_EQUAL(4, count);
   }
   fclose(f);
}

// set up the cosmological parameters to be used in the test case
CTEST_SETUP(massfunc) {
  // only single model at tihs point
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->sigma_8 = 0.8;
  data->N_nu_rel=0;
  data->N_nu_mass=0;
  data->mnu=0;

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
  read_massfunc_test_file(data->mass, data->massfunc);
}

static void compare_massfunc(int model, struct massfunc_data * data)
{
  int stat = 0;
  int* status = &stat;

  ccl_parameters params = ccl_parameters_create(data->Omega_c, data->Omega_b,data->Omega_k[model],
						data->N_nu_rel, data->N_nu_mass, data->mnu,data->w_0[model],
						data->w_a[model], data->h,data->A_s, data->n_s,
						-1, NULL, NULL, status);

  params.sigma_8 = data->sigma_8;
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  // test file generated using tinker 2008 currently
  config.mass_function_method = ccl_tinker;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  
  ASSERT_NOT_NULL(cosmo);

  double a = 1.0;
  double logmass = 10;
  double odelta = 200;
  double rho_m = RHO_CRITICAL*cosmo->params.Omega_m*cosmo->params.h*cosmo->params.h;

  // compare to benchmark data
  for (int j=0; j<13; j++) {
    double mass = pow(10,logmass);
    double sigma_j = ccl_sigmaM(cosmo, mass, a, status);
    double loginvsigma_j = log10(1./sigma_j);
    double logmassfunc_j = log10(ccl_massfunc(cosmo, mass, a, odelta, status)*mass/(rho_m*log(10.)));

    double absolute_tolerance = SIGMA_TOLERANCE*data->massfunc[0][j];
    if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
    ASSERT_DBL_NEAR_TOL(data->massfunc[0][j], sigma_j, absolute_tolerance);

    absolute_tolerance = INVSIGMA_TOLERANCE*fabs(data->massfunc[1][j]);
    if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
    ASSERT_DBL_NEAR_TOL(fabs(data->massfunc[1][j]), fabs(loginvsigma_j), absolute_tolerance);

    absolute_tolerance = MASSFUNC_TOLERANCE*fabs(data->massfunc[2][j]);
    if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
    ASSERT_DBL_NEAR_TOL(fabs(data->massfunc[2][j]), fabs(logmassfunc_j), absolute_tolerance);
   
    logmass += 0.5;
  }
  free(cosmo);
}

CTEST2(massfunc, model_1) {
   int model = 0;
   compare_massfunc(model, data);
}
