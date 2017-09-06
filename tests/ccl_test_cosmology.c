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
  double N_nu_rel;
  double N_nu_mass;
  double m_nu;
  int status;
};

// This function is one before each test defined below with CTEST2 in the suite.
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
  data->N_nu_rel = 0.;
  data->N_nu_mass=0.;
  data->m_nu=0.;
  data->status=0;
}

// Check to see if general ccl_cosmology struct is initialized correctly
CTEST2(cosmology, create_general_cosmo) {
  ccl_configuration config = default_config;
  
  // Initialize ccl_cosmology struct
  ccl_cosmology * cosmo = ccl_cosmology_create_with_params(data->Omega_c, data->Omega_b, data->Omega_k, 
							   data->N_nu_rel, data->N_nu_mass, data->m_nu, 
							   data->w0, data->wa, data->h, data->A_s, data->n_s,
							   -1, NULL, NULL, config, &(data->status));
  
  // Pull ccl_parameters object out of ccl_cosmology
  ccl_parameters params = (*cosmo).params;
  
  ASSERT_DBL_NEAR_TOL(params.Omega_c, data->Omega_c, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.w0, -1.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.wa, data->wa, 1e-10);
}

// Check to see if LCDM ccl_cosmology struct is initialized correctly
CTEST2(cosmology, create_lcdm_cosmo) {
  ccl_configuration config = default_config;
  
  // Initialize ccl_cosmology struct
  ccl_cosmology * cosmo = ccl_cosmology_create_with_lcdm_params(data->Omega_c, data->Omega_b, data->Omega_k,
								data->h,data->A_s, data->n_s, config,
								&(data->status));
  
  // Pull ccl_parameters object out of ccl_cosmology
  ccl_parameters params = (*cosmo).params;
  
  ASSERT_DBL_NEAR_TOL(params.Omega_c, data->Omega_c, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.w0, -1.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.wa, 0.0, 1e-10);
}
