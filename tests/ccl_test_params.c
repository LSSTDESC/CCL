#include "ccl.h"
#include "ctest.h"

// We can define any constants we want to use in a set of tests here.
// They are accessible as data->Omega_c, etc., in the tests themselves below. 
// "params" is the name of the whole suite of tests.
CTEST_DATA(params) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double wa;
  double w0;
  int status;
};

// This function is one before each test defined below with CTEST2 in the suite.
// It is used to set up any values needed by the tests.  The data
// that can be passed to the tests are always in a struct called "data"
// and defined above.
CTEST_SETUP(params) {
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->wa = 0.01;
  data->w0 = -1.0;
  data->status=0;
}

// The 2 on the end of CTEST2 means that for this test we use 
// the data defined above in CTEST_DATA and given values in CTEST_SETUP function.  
// We could also define CTEST_TEARDOWN(params) that would be run after the tests.

// This adds a new test called "create_lcdm" to the suite called "params".
// There are a variety of different assertions available.

// If you wanted to you could call other functions in here too and use these
// assertions there also.
CTEST2(params, create_lcdm) {
  ccl_parameters params = ccl_parameters_create_flat_lcdm(data->Omega_c, data->Omega_b, data->h, data->A_s, data->n_s, &(data->status));
  ASSERT_DBL_NEAR_TOL(params.Omega_c, data->Omega_c, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.w0, -1.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.wa, 0.0, 1e-10);
}

// This adds a second test in the same suite.  It uses the same setup function as the
// previous one (though the setup function is run afresh for each test).
CTEST2(params, create_wacdm) {
  ccl_parameters params = ccl_parameters_create_flat_lcdm(data->Omega_c, data->Omega_b, data->h, data->A_s, data->n_s, &(data->status));
  ASSERT_DBL_NEAR_TOL(params.Omega_c, data->Omega_c, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.Omega_k, 0.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.w0, -1.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.wa, 0.0, 1e-10);
}
