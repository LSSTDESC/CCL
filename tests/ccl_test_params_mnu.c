#include "ccl.h"
#include "ctest.h"

// We can define any constants we want to use in a set of tests here.
// They are accessible as data->Omega_c, etc., in the tests themselves below. 
// "params" is the name of the whole suite of tests.
CTEST_DATA(create_mnu) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double wa;
  double w0;
  double Omega_k;
  double mnuval;
  double Neff;
  int status;
  ccl_mnu_convention mnu_type_norm;
  ccl_mnu_convention mnu_type_inv;
};

// This function is one before each test defined below with CTEST2 in the suite.
// It is used to set up any values needed by the tests.  The data
// that can be passed to the tests are always in a struct called "data"
// and defined above.
CTEST_SETUP(create_mnu) {
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->wa = 0.01;
  data->w0 = -1.0;
  data->Neff=3.046;
  data->Omega_k = 0.;
  data->mnuval = 0.15;
  data->mnu_type_norm = ccl_mnu_sum;
  data->mnu_type_inv = ccl_mnu_sum_inverted;
  data->status = 0;
  
}

// This adds a new test called "create_mnu" to the suite called "params".
// There are a variety of different assertions available.

// If you wanted to you could call other functions in here too and use these
// assertions there also.
CTEST2(create_mnu, create_mnu_norm) {
  ccl_parameters params_norm = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k,
						data->Neff, &(data->mnuval), data->mnu_type_norm,
						data->w0, data->wa,
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &(data->status));
  						
  ASSERT_DBL_NEAR_TOL(params_norm.mnu[1]*params_norm.mnu[1] - params_norm.mnu[0]*params_norm.mnu[0], DELTAM12_sq, 1e-4);
  ASSERT_DBL_NEAR_TOL(params_norm.mnu[2]*params_norm.mnu[2] - params_norm.mnu[0]*params_norm.mnu[0], DELTAM13_sq_pos, 1e-4);
}

CTEST2(create_mnu, create_mnu_inv){
  
  ccl_parameters params_inv = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k,
						data->Neff, &(data->mnuval), data->mnu_type_inv,
						data->w0, data->wa,
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &(data->status));
						
  ASSERT_DBL_NEAR_TOL(params_inv.mnu[1]*params_inv.mnu[1] - params_inv.mnu[0]*params_inv.mnu[0], DELTAM12_sq, 1e-4);
  ASSERT_DBL_NEAR_TOL(params_inv.mnu[2]*params_inv.mnu[2] - params_inv.mnu[0]*params_inv.mnu[0], DELTAM13_sq_neg, 1e-4);
  
}

