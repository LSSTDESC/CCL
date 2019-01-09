#include "ccl.h"
#include "ctest.h"

CTEST_DATA(MG) {
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
  double z;
};

// This function is one before each test defined below with CTEST2 in the suite.
// It is used to set up any values needed by the tests.  The data
// that can be passed to the tests are always in a struct called "data"
// and defined above.
CTEST_SETUP(MG) {
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->Omega_k = 0.00;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->wa = 0.0;
  data->w0 = -1.0;
  data->Neff = 0.;
  data->mnuval =0.;
  data->status=0;
  data->mnu_type =ccl_mnu_sum;
  data->mu_0=0.1;
  data->sigma_0=0.1;
  data->z = 0.;
}

static void call_mu_ofz(struct MG_data * data)
{
  int status=0;
  
  ccl_configuration config = default_config;
  
  // Initialize ccl_cosmology struct
  ccl_parameters params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k, 
							   data->Neff, &(data->mnuval), data->mnu_type,
							   data->w0, data->wa, data->h, data->A_s, data->n_s,
							   -1,-1,-1, data->mu_0, data->sigma_0,-1, NULL, NULL, &(data->status));
							   
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  
  ASSERT_NOT_NULL(cosmo);
  
  // Call mu(z)
  double a, mu_out; 
  a = 1/(1.+data->z);
  mu_out = ccl_mu_MG(cosmo, a, &status);
  ASSERT_DBL_NEAR_TOL(data->mu_0, mu_out, 1e-4);
  
  ccl_cosmology_free(cosmo);
}

static void call_sig_ofz(struct MG_data * data)
{
  int status=0;
  
  ccl_configuration config = default_config;
  
  // Initialize ccl_cosmology struct
  ccl_parameters params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k, 
							   data->Neff, &(data->mnuval), data->mnu_type,
							   data->w0, data->wa, data->h, data->A_s, data->n_s,
							   -1,-1,-1, data->mu_0, data->sigma_0,-1, NULL, NULL, &(data->status));
							   
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  
  ASSERT_NOT_NULL(cosmo);
  
  // Call mu(z)
  double a = 1/(1.+data->z);
  double sig_out = ccl_Sig_MG(cosmo, a, &status);
  ASSERT_DBL_NEAR_TOL(data->sigma_0, sig_out, 1e-4);
  
  ccl_cosmology_free(cosmo);
}

CTEST2(MG, create_mu_of_z) {
  call_mu_ofz(data);
}

CTEST2(MG, create_Sig_of_z) {
  call_sig_ofz(data);
}
