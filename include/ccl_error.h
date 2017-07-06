/** @file */

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
#define CCL_ERROR_NU_INT 13

typedef enum {
    CCL_ERROR_POLICY_EXIT = 0,
    CCL_ERROR_POLICY_CONTINUE = 1,
} CCLErrorPolicy;

/** Raise an exception
 * Given a status, give an error message.
 * @return void
 */
void ccl_raise_exception(int err, char* msg);

/** Set the error policy
 * @oaram error_policy the error policy
 * @return void
 */
void ccl_set_error_policy(CCLErrorPolicy error_policy);

/** Check the error status
 * Given a status, check if any errors have occurred,
 * based on the CCL_ERRORs defined so far.
 * @return void
 */
void ccl_check_status(ccl_cosmology *cosmo, int* status);

/** Check the error status - no cosmology
 * Given a status, check if any errors have occurred,
 * based on the CCL_ERRORs defined so far.
 * @return void
 */
void ccl_check_status_nocosmo(int* status);
