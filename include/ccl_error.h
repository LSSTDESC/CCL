/** @file */
#ifndef __CCL_ERROR_H_INCLUDED__
#define __CCL_ERROR_H_INCLUDED__

CCL_BEGIN_DECLS

// Whether to do bounds checks on interpolated quantities
#define CCL_BOUNDS_CHECK_INTERP

#define CCL_ERROR_MEMORY 1025
#define CCL_ERROR_LINSPACE 1026
#define CCL_ERROR_INCONSISTENT 1027
#define CCL_ERROR_SPLINE 1028
#define CCL_ERROR_SPLINE_EV 1029
#define CCL_ERROR_INTEG 1030
#define CCL_ERROR_ROOT 1031
#define CCL_ERROR_CLASS 1032
#define CCL_ERROR_COMPUTECHI 1033
#define CCL_ERROR_MF 1034
#define CCL_ERROR_HMF_INTERP 1035
#define CCL_ERROR_PARAMETERS 1036
#define CCL_ERROR_NU_INT 1037
#define CCL_ERROR_EMULATOR_BOUND 1038
#define CCL_ERROR_NU_SOLVE 1039
#define CCL_ERROR_NOT_IMPLEMENTED 1040
#define CCL_ERROR_MNU_UNPHYSICAL 1041
#define CCL_ERROR_ANGPOW 1042
#define CCL_ERROR_MISSING_CONFIG_FILE 1043
#define CCL_ERROR_HALOCONC 1044
#define CCL_ERROR_HALOWIN 1045
#define CCL_ERROR_HMF_DV 1046
#define CCL_ERROR_CONC_DV 1047
#define CCL_ERROR_ONE_HALO_INT 1048
#define CCL_ERROR_TWO_HALO_INT 1049
#define CCL_ERROR_FILE_WRITE 1050
#define CCL_ERROR_FILE_READ 1051
#define CCL_ERROR_LOGSPACE 1052
#define CCL_ERROR_LINLOGSPACE 1053
#define CCL_ERROR_CONFIG_FILE 1054
#define CCL_ERROR_LOWMNU 1055

typedef enum {
  CCL_ERROR_POLICY_EXIT = 0,
  CCL_ERROR_POLICY_CONTINUE = 1,
} CCLErrorPolicy;

typedef enum {
  CCL_DEBUG_MODE_OFF = 0,
  CCL_DEBUG_MODE_ON = 1,
  CCL_DEBUG_MODE_WARNING = 2,
} CCLDebugModePolicy;

/** Raise an exception
 * Given a status, give an error message.
 * @return void
 */
void ccl_raise_exception(int err, const char* msg, ...);

/** Raise a warning
 * Given a status, give a warning message.
 * @return void
 */
void ccl_raise_warning(int err, const char* msg, ...);

/** Raise a warning based on a GSL error message
 * Given a GSL status, give a warning message.
 * @return void
 */
void ccl_raise_gsl_warning(int gslstatus, const char* msg, ...);

/** Set the error policy
 * @oaram error_policy the error policy
 * @return void
 */
void ccl_set_error_policy(CCLErrorPolicy error_policy);

/** Set the error policy
 * @oaram debug_policy the debug mode policy
 * @return void
 */
void ccl_set_debug_policy(CCLDebugModePolicy debug_policy);

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

CCL_END_DECLS

#endif
