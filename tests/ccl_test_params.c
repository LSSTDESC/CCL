#include <math.h>
#include "ccl.h"
#include "ctest.h"

// We can define any constants we want to use in a set of tests here.
// They are accessible as data->Omega_c, etc., in the tests themselves below.
// "params" is the name of the whole suite of tests.
CTEST_DATA(parameters) {
  double Omega_c;
  double Omega_b;
  double Omega_k;
  double Neff;
  double mnu[3];
  double h;
  double A_s;
  double n_s;
  double wa;
  double w0;
  double bcm_log10Mc;
  double bcm_etab;
  double bcm_ks;
  double mu_0;
  double sigma_0;
  int status;
};

// This function is one before each test defined below with CTEST2 in the suite.
// It is used to set up any values needed by the tests.  The data
// that can be passed to the tests are always in a struct called "data"
// and defined above.
CTEST_SETUP(parameters) {
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->Omega_k = 0.1;
  data->Neff = 3.046;
  data->mnu[0] = 0.1;
  data->mnu[1] = 0.01;
  data->mnu[2] = 0.003;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->wa = 0.01;
  data->w0 = -0.8;
  data->bcm_log10Mc = 2.0;
  data->bcm_etab = 80.0;
  data->bcm_ks = 1.1,
  data->status=0;
  data->mu_0 = 0.;
  data->sigma_0 = 0.;
}

// The 2 on the end of CTEST2 means that for this test we use
// the data defined above in CTEST_DATA and given values in CTEST_SETUP function.
// We could also define CTEST_TEARDOWN(params) that would be run after the tests.

// This adds a new test called "create_lcdm" to the suite called "params".
// There are a variety of different assertions available.

// If you wanted to you could call other functions in here too and use these
// assertions there also.
CTEST2(parameters, create_lcdm) {
  ccl_parameters params = ccl_parameters_create_flat_lcdm(
    data->Omega_c, data->Omega_b, data->h, data->A_s, data->n_s, &(data->status));

  ASSERT_DBL_NEAR_TOL(params.Omega_c, data->Omega_c, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.Omega_b, data->Omega_b, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.Omega_m, data->Omega_b + data->Omega_c, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.Omega_k, 0.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.sqrtk, 0.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.k_sign, 0.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.w0, -1.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.wa, 0.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.H0, 70.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.h, 0.7, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.A_s, data->A_s, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.n_s, data->n_s, 1e-10);
  ASSERT_TRUE(isnan(params.sigma8));
  ASSERT_TRUE(isnan(params.z_star));
  ASSERT_DBL_NEAR_TOL(params.Neff, 3.046, 1e-10);
  ASSERT_EQUAL(params.N_nu_mass, 0);
  ASSERT_DBL_NEAR_TOL(params.N_nu_rel, 3.046, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.sum_nu_masses, 0.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.mnu[0], 0.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.Omega_n_mass, 0.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.T_CMB, TCMB, 1e-10);

  ASSERT_DBL_NEAR_TOL(params.bcm_ks, 55.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.bcm_log10Mc, log10(1.2e14), 1e-10);
  ASSERT_DBL_NEAR_TOL(params.bcm_etab, 0.5, 1e-10);

  ASSERT_FALSE(params.has_mgrowth);
  ASSERT_EQUAL(params.nz_mgrowth, 0);
  ASSERT_NULL(params.z_mgrowth);
  ASSERT_NULL(params.df_mgrowth);

  /* these are defined in the code via some constants - going to test the total
    Omega_n_rel
    Omega_g
    Omega_l
  */
  ASSERT_DBL_NEAR_TOL(
    params.Omega_l + params.Omega_m +
    params.Omega_g + params.Omega_n_rel +
    params.Omega_n_mass + params.Omega_k,
    1.,
    1e-10);
}

// This adds a second test in the same suite.  It uses the same setup function as the
// previous one (though the setup function is run afresh for each test).
void test_general(ccl_parameters params, struct parameters_data * data) {
  ASSERT_DBL_NEAR_TOL(params.Omega_c, data->Omega_c, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.Omega_b, data->Omega_b, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.Omega_m, data->Omega_b + data->Omega_c, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.Omega_k, data->Omega_k, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.sqrtk, sqrt(fabs(data->Omega_k))*data->h/CLIGHT_HMPC, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.k_sign, -1.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.w0, data->w0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.wa, data->wa, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.H0, data->h * 100.0, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.h, data->h, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.A_s, data->A_s, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.n_s, data->n_s, 1e-10);
  ASSERT_TRUE(isnan(params.sigma8));
  ASSERT_TRUE(isnan(params.z_star));
  ASSERT_DBL_NEAR_TOL(params.Neff, data->Neff, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.T_CMB, TCMB, 1e-10);

  ASSERT_DBL_NEAR_TOL(params.bcm_ks, data->bcm_ks, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.bcm_log10Mc, data->bcm_log10Mc, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.bcm_etab, data->bcm_etab, 1e-10);

  ASSERT_FALSE(params.has_mgrowth);
  ASSERT_EQUAL(params.nz_mgrowth, 0);
  ASSERT_NULL(params.z_mgrowth);
  ASSERT_NULL(params.df_mgrowth);

  /* these are defined in the code via some constants - going to test the total
    Omega_n_mass
    Omega_n_rel
    Omega_g
    Omega_l
  */
  ASSERT_DBL_NEAR_TOL(
    params.Omega_l + params.Omega_m + params.Omega_g + params.Omega_n_rel + params.Omega_n_mass + params.Omega_k,
    1.,
    1e-10);

}

CTEST2(parameters, create_general_nu_list) {
  int status = 0;

  ccl_parameters params =
    ccl_parameters_create(
      data->Omega_c,
      data->Omega_b,
      data->Omega_k,
      data->Neff,
      data->mnu,
      ccl_mnu_list,
      data->w0,
      data->wa,
      data->h,
      data->A_s,
      data->n_s,
      data->bcm_log10Mc,
      data->bcm_etab,
      data->bcm_ks,
      data->mu_0,
      data->sigma_0,
      -1,
      NULL,
      NULL,
      &status);

  ASSERT_EQUAL(status, 0);

  test_general(params, data);

  ASSERT_EQUAL(params.N_nu_mass, 3);
  ASSERT_DBL_NEAR_TOL(params.sum_nu_masses, data->mnu[0] + data->mnu[1] + data->mnu[2], 1e-10);
  ASSERT_DBL_NEAR_TOL(params.mnu[0], 0.1, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.mnu[1], 0.01, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.mnu[2], 0.003, 1e-10);
}

CTEST2(parameters, create_general_nu_sum) {
  int status = 0;

  ccl_parameters params =
    ccl_parameters_create(
      data->Omega_c,
      data->Omega_b,
      data->Omega_k,
      data->Neff,
      data->mnu,
      ccl_mnu_sum,
      data->w0,
      data->wa,
      data->h,
      data->A_s,
      data->n_s,
      data->bcm_log10Mc,
      data->bcm_etab,
      data->bcm_ks,
      data->mu_0,
      data->sigma_0,
      -1,
      NULL,
      NULL,
      &status);

  ASSERT_EQUAL(status, 0);

  test_general(params, data);

  ASSERT_EQUAL(params.N_nu_mass, 3);
  ASSERT_DBL_NEAR_TOL(params.sum_nu_masses, data->mnu[0], 1e-10);
}

CTEST2(parameters, create_general_nu_sum_inverted) {
  int status = 0;

  ccl_parameters params =
    ccl_parameters_create(
      data->Omega_c,
      data->Omega_b,
      data->Omega_k,
      data->Neff,
      data->mnu,
      ccl_mnu_sum,
      data->w0,
      data->wa,
      data->h,
      data->A_s,
      data->n_s,
      data->bcm_log10Mc,
      data->bcm_etab,
      data->bcm_ks,
      data->mu_0,
      data->sigma_0,
      -1,
      NULL,
      NULL,
      &status);

  ASSERT_EQUAL(status, 0);

  test_general(params, data);

  ASSERT_EQUAL(params.N_nu_mass, 3);
  ASSERT_DBL_NEAR_TOL(params.sum_nu_masses, data->mnu[0], 1e-10);
}

CTEST2(parameters, create_general_nu_sum_equal) {
  int status = 0;

  ccl_parameters params =
    ccl_parameters_create(
      data->Omega_c,
      data->Omega_b,
      data->Omega_k,
      data->Neff,
      data->mnu,
      ccl_mnu_sum_equal,
      data->w0,
      data->wa,
      data->h,
      data->A_s,
      data->n_s,
      data->bcm_log10Mc,
      data->bcm_etab,
      data->bcm_ks,
      data->mu_0,
      data->sigma_0,
      -1,
      NULL,
      NULL,
      &status);

  ASSERT_EQUAL(status, 0);

  test_general(params, data);

  ASSERT_EQUAL(params.N_nu_mass, 3);
  ASSERT_DBL_NEAR_TOL(params.sum_nu_masses, data->mnu[0], 1e-10);
  ASSERT_DBL_NEAR_TOL(params.mnu[0], data->mnu[0]/3, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.mnu[1], data->mnu[0]/3, 1e-10);
  ASSERT_DBL_NEAR_TOL(params.mnu[2], data->mnu[0]/3, 1e-10);
}


CTEST2(parameters, read_write) {
    char filename[32];
    snprintf(filename, 32, "ccl_test_params_rw_XXXXXX");
    mkstemp(filename);
    ccl_parameters params = ccl_parameters_create_flat_lcdm(data->Omega_c, data->Omega_b, data->h, data->A_s, data->n_s, &(data->status));
    int status = 0;
    ccl_parameters_write_yaml(&params, filename, &status);
    ASSERT_EQUAL(status, 0);
    ccl_parameters params2 = ccl_parameters_read_yaml(filename, &status);
    ASSERT_EQUAL(status, 0);
    ASSERT_DBL_NEAR_TOL(params2.Omega_c, data->Omega_c, 1e-10);
    ASSERT_DBL_NEAR_TOL(params2.Omega_k, 0.0, 1e-10);
    ASSERT_DBL_NEAR_TOL(params2.w0, -1.0, 1e-10);
    ASSERT_DBL_NEAR_TOL(params2.wa, 0.0, 1e-10);
    remove(filename);
}
