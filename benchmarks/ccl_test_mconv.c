#include "ccl.h"
#include "ctest.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

CTEST_DATA(mconv) {
  ccl_cosmology *cosmo;
};

CTEST_SETUP(mconv) {
  int status=0;
  ccl_parameters params = ccl_parameters_create_flat_lcdm(0.25,0.05,0.7,
							  2E-9,0.96, &status);
  data->cosmo = ccl_cosmology_create(params, default_config);
  ASSERT_NOT_NULL(data->cosmo);
};

CTEST_TEARDOWN(mconv) {
  ccl_parameters_free(&(data->cosmo->params));
  ccl_cosmology_free(data->cosmo);
}

CTEST2(mconv,sanity) {
  double c_new[3];
  double c_old[3]={9.,10.,11.};
  int ii,status=0;

  // No change expected
  ccl_get_new_concentration(data->cosmo,
			    200, 3, c_old,
			    200, c_new,
			    &status);
  for(ii=0;ii<3;ii++)
    ASSERT_TRUE(c_new[ii]==c_old[ii]);

  // Test against numerical solution from Mathematica.
  ccl_get_new_concentration(data->cosmo,
			    200, 3, c_old,
			    500, c_new,
			    &status);
  ASSERT_DBL_NEAR(c_new[0], 6.12194);
  ASSERT_DBL_NEAR(c_new[1], 6.82951);
  ASSERT_DBL_NEAR(c_new[2], 7.53797);
}
