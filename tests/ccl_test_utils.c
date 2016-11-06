#include "ccl.h"
#include "ctest.h"

CTEST(spacing_tests, linear_spacing_simple){
    double * m = ccl_linear_spacing(0.0, 1.0, 2);
    ASSERT_NOT_NULL(m);
    ASSERT_DBL_NEAR_TOL(0.0, m[0], 1e-10);
    ASSERT_DBL_NEAR_TOL(1.0, m[1], 1e-10);
    free(m);
}


CTEST(spacing_tests, linear_spacing_spline_limits){
    double * m = ccl_linear_spacing(A_SPLINE_MIN, A_SPLINE_MAX, A_SPLINE_NA);
    ASSERT_NOT_NULL(m);
    ASSERT_DBL_NEAR_TOL(A_SPLINE_MIN, m[0], 1e-5);
    ASSERT_DBL_NEAR_TOL(A_SPLINE_MAX, m[A_SPLINE_NA-1], 1e-5);
    ASSERT_TRUE(m[A_SPLINE_NA-1]<=1.0);
    free(m);
}

