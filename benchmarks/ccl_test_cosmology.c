#include "ccl.h"
#include "ctest.h"

// We can define any constants we want to use in a set of tests here.
// They are accessible as data->Omega_c, etc., in the tests themselves below.
// "params" is the name of the whole suite of tests.
CTEST_DATA(cosmology) {
  double Omega_c;
  double Omega_b;
  double Omega_k;
  double h;
  double A_s;
  double n_s;
  double wa;
  double w0;
  double Neff;
  double mnuval;
  ccl_mnu_convention mnu_type;
  int status;
  double mu_0;
  double sigma_0;
};

// This function is one before each test defined below with CTEST2_SKIP in the suite.
// It is used to set up any values needed by the tests.  The data
// that can be passed to the tests are always in a struct called "data"
// and defined above.
CTEST_SETUP(cosmology) {
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->Omega_k = 0.00;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->wa = 0.01;
  data->w0 = -1.0;
  data->Neff = 0.;
  data->mnuval =0.;
  data->status=0;
  data->mnu_type =ccl_mnu_sum;
  data->mu_0=0.;
  data->sigma_0=0.;
}

// Check to see if general ccl_cosmology struct is initialized correctly
CTEST2(cosmology, create_general_cosmo) {
  ccl_configuration config = default_config;
  ccl_parameters params = ccl_parameters_create_flat_lcdm(
    data->Omega_c, data->Omega_b, data->h, data->A_s, data->n_s,
    &(data->status));


  // Initialize ccl_cosmology struct
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

  // test a few critical things
  ASSERT_EQUAL(cosmo->status, 0);
  ASSERT_DBL_NEAR_TOL(cosmo->data.growth0, 1., 1e-10);
}
