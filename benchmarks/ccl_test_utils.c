#include <gsl/gsl_sf_bessel.h>
#include "ccl.h"
#include "ctest.h"
#include "ccl_core.h"

CTEST(spacing_tests, linear_spacing_simple) {
  double * m = ccl_linear_spacing(0.0, 1.0, 5);
  ASSERT_NOT_NULL(m);
  ASSERT_DBL_NEAR_TOL(0.0, m[0], 1e-10);
  ASSERT_DBL_NEAR_TOL(0.25, m[1], 1e-10);
  ASSERT_DBL_NEAR_TOL(0.5, m[2], 1e-10);
  ASSERT_DBL_NEAR_TOL(0.75, m[3], 1e-10);
  ASSERT_DBL_NEAR_TOL(1.0, m[4], 1e-10);
  free(m);
}

CTEST(spacing_tests, log_spacing_simple) {
  double * m = ccl_log_spacing(1.0, 16.0, 5);
  ASSERT_NOT_NULL(m);
  ASSERT_DBL_NEAR_TOL(1.0, m[0], 1e-10);
  ASSERT_DBL_NEAR_TOL(2.0, m[1], 1e-10);
  ASSERT_DBL_NEAR_TOL(4.0, m[2], 1e-10);
  ASSERT_DBL_NEAR_TOL(8.0, m[3], 1e-10);
  ASSERT_DBL_NEAR_TOL(16.0, m[4], 1e-10);
  free(m);
}

CTEST(spacing_tests, linlog_spacing_simple) {
  double * m = ccl_linlog_spacing(1.0, 16.0, 32.0, 5, 9);
  ASSERT_NOT_NULL(m);
  ASSERT_DBL_NEAR_TOL(1.0, m[0], 1e-10);
  ASSERT_DBL_NEAR_TOL(2.0, m[1], 1e-10);
  ASSERT_DBL_NEAR_TOL(4.0, m[2], 1e-10);
  ASSERT_DBL_NEAR_TOL(8.0, m[3], 1e-10);
  ASSERT_DBL_NEAR_TOL(16.0, m[4], 1e-10);
  ASSERT_DBL_NEAR_TOL(18.0, m[5], 1e-10);
  ASSERT_DBL_NEAR_TOL(20.0, m[6], 1e-10);
  ASSERT_DBL_NEAR_TOL(22.0, m[7], 1e-10);
  ASSERT_DBL_NEAR_TOL(24.0, m[8], 1e-10);
  ASSERT_DBL_NEAR_TOL(26.0, m[9], 1e-10);
  ASSERT_DBL_NEAR_TOL(28.0, m[10], 1e-10);
  ASSERT_DBL_NEAR_TOL(30.0, m[11], 1e-10);
  ASSERT_DBL_NEAR_TOL(32.0, m[12], 1e-10);
  free(m);
}

CTEST(spherical_bessel_tests, compare_gsl) {
  int l, i;
  double xmin = 0.0;
  double xmax = 10.0;
  int Nx = 10000;
  double x, dx;

  dx = (xmax - xmin) / (Nx - 1);

  for (l=0; l < 15; ++l) {
    for (i=0; i < Nx; ++i) {
      x = dx * i + xmin;
      ASSERT_DBL_NEAR_TOL(
        ccl_j_bessel(l, x),
        gsl_sf_bessel_jl(l, x),
        1e-4);
    }
  }
}
