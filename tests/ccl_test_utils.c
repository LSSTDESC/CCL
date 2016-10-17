#include "ccl.h"
#include "ctest.h"

CTEST(spacing_tests, linear_spacing_simple){
    int n;
    double * m = ccl_linear_spacing(0.0, 1.0, 1.0, &n);
    ASSERT_NOT_NULL(m);
    ASSERT_EQUAL(2, n);
    ASSERT_DBL_NEAR_TOL(0.0, m[0], 1e-10);
    ASSERT_DBL_NEAR_TOL(1.0, m[1], 1e-10);
    free(m);
}


CTEST(spacing_tests, linear_spacing_fail_partial_delta){
    int n;
    double * m = ccl_linear_spacing(0.0, 1.0, 0.75, &n);
    ASSERT_NULL(m);
    ASSERT_EQUAL(0,n);
    free(m);
}


CTEST(spacing_tests, linear_spacing_fail_large_delta){
    int n;
    double * m = ccl_linear_spacing(0.0, 1.0, 2.0, &n);
    ASSERT_EQUAL(0,n);
    ASSERT_NULL(m);
    free(m);
}

CTEST(spacing_tests, linear_spacing_spline_limits){

    int n;
    double * m = ccl_linear_spacing(A_SPLINE_MIN, A_SPLINE_MAX, A_SPLINE_DELTA, &n);
    ASSERT_TRUE(n>0);
    ASSERT_NOT_NULL(m);
    ASSERT_DBL_NEAR_TOL(A_SPLINE_MIN, m[0], 1e-5);
    ASSERT_DBL_NEAR_TOL(A_SPLINE_MAX, m[n-1], 1e-5);
    ASSERT_TRUE(m[n-1]<=1.0);
    free(m);
}

