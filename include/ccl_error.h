#pragma once
#include "ccl_core.h"

#define CCL_ERROR_MEMORY 1
#define CCL_ERROR_LINSPACE 2
#define CCL_ERROR_INCONSISTENT 3
#define CCL_ERROR_SPLINE 4
#define CCL_ERROR_SPLINE_EV 5
#define CCL_ERROR_INTEG 6
#define CCL_ERROR_ROOT 7
#define CCL_ERROR_CLASS 8
#define CCL_ERROR_COMPUTECHI 9
#define CCL_ERROR_MF 10
#define CCL_ERROR_HMF_INTERP 11
#define CCL_ERROR_PARAMETERS 12

typedef enum {
    CCL_ERRORHANDLING_POLICY_ABORT = 0,
    CCL_ERRORHANDLING_POLICY_UNWIND = 1,
} CCLErrorHandlingPolicy;

void ccl_check_status(ccl_cosmology *cosmo, int* status);
void ccl_errorhandling_set_policy(CCLErrorHandlingPolicy policy);
