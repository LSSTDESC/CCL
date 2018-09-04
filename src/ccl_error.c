#include "ccl_core.h"
#include "ccl_error.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_errno.h"

// Error handling policy: whether to exit on error (C default) or continue 
// (Python or other binding default)
static CCLErrorPolicy _ccl_error_policy = CCL_ERROR_POLICY_EXIT;

// Debug mode policy: whether to print error messages as they are raised. This 
// is useful for the Python wrapper, which normally allows errors to be 
// overwritten by the C code until control returns to Python. If debug mode is 
// switched on, the errors are always printed by the C code when they occur.
// Setting the debug mode to warning allows ccl_raise_warning to print warnings
// but keeps the behavior of ccl_raise_exception as if debug mode is set to off.
static CCLDebugModePolicy _ccl_debug_mode_policy = CCL_DEBUG_MODE_WARNING;

// Set error policy
void ccl_set_error_policy(CCLErrorPolicy error_policy)
{
  _ccl_error_policy = error_policy;
}

// Set debug mode policy
void ccl_set_debug_policy(CCLDebugModePolicy debug_policy)
{
    _ccl_debug_mode_policy = debug_policy;
}

// Convenience function to raise exceptions in an appropriate way
void ccl_raise_exception(int err, char* msg)
{  
  // Print error message and exit if fatal errors are enabled
  if ((_ccl_error_policy == CCL_ERROR_POLICY_EXIT) && (err)) {
    fprintf(stderr, "ERROR %d: %s\n", err, msg);
    exit(1);
  }
  // Print error message and exit if debug output is enabled
  else if ((_ccl_debug_mode_policy == CCL_DEBUG_MODE_ON) && (err)){
    fprintf(stderr, "ERROR %d: %s\n", err, msg);
  }
}

// Convenience function to handle warnings
void ccl_raise_warning(int err, char* msg)
{
  // For now just print warning to stderr if debug is enabled.
  // TODO: Implement some kind of error stack that can be passed on to, e.g.,
  // the python binding.
  char warning[256];
  snprintf(warning, 256, "WARNING: %s", msg);
  if( (_ccl_debug_mode_policy == CCL_DEBUG_MODE_ON) 
      || (_ccl_debug_mode_policy == CCL_DEBUG_MODE_WARNING) ) {
    fprintf(stderr, "%s\n", warning);
  }
}

// Convenience function to handle warnings
void ccl_raise_gsl_warning(int gslstatus, char* msg)
{
  char warning[256];
  snprintf(warning, 256, "%s GSL error: %s", msg, gsl_strerror(gslstatus));
  ccl_raise_warning(gslstatus, warning);
  return;
}

void ccl_check_status(ccl_cosmology *cosmo, int * status)
{
	
  switch (*status) {
  case 0: // all good, nothing to do
    return;
  case CCL_ERROR_LINSPACE:	// spacing allocation error, always terminate		
    ccl_raise_exception(*status, cosmo->status_message);
  case CCL_ERROR_SPLINE:	// spline allocation error, always terminate	
    ccl_raise_exception(*status, cosmo->status_message);
  case CCL_ERROR_COMPUTECHI:	// compute_chi error //RH
    ccl_raise_exception(*status, cosmo->status_message);
  case CCL_ERROR_HMF_INTERP: // terminate if hmf definition not supported
    ccl_raise_exception(*status, cosmo->status_message);
  case CCL_ERROR_NU_INT: // error in getting the neutrino integral spline: exit. No status_message in cosmo because can't pass cosmology to the function.
    ccl_raise_exception(*status, "Error, in ccl_neutrinos.c. ccl_calculate_nu_phasespace_spline(): Error in setting neutrino phasespace spline.");
  case CCL_ERROR_NU_SOLVE: // error in converting Omeganuh2-> Mnu: exit. No status_message in cosmo because can't pass cosmology to the function.
    ccl_raise_exception(*status, "Error, in ccl_neutrinos.c. Omeganuh2_to_Mnu(): Root finding did not converge.");
    // TODO: Implement softer error handling, e.g. for integral convergence here	
  default: 
    ccl_raise_exception(*status, cosmo->status_message);
  }
}

/* ------- ROUTINE: ccl_check_status_nocosmo ------
   INPUTS: pointer to a status integer
   TASK: Perform a check on status for the case where it is not possible to have a cosmology object where the status check is required.
*/
void ccl_check_status_nocosmo(int * status)
{
  switch (*status) {
  case 0: // Nothing to do
    return;
  case CCL_ERROR_LINSPACE:
    // Spacing allocation error, always terminate
    ccl_raise_exception(*status, "CCL_ERROR_LINSPACE: Spacing allocation error.");
  case CCL_ERROR_SPLINE:
    // Spline allocation error, always terminate
    ccl_raise_exception(*status, "CCL_ERROR_SPLINE: Spline allocation error.");
  case CCL_ERROR_COMPUTECHI:
    // Compute_chi error
    ccl_raise_exception(*status, 
             "CCL_ERROR_COMPUTECHI: Comoving distance chi computation failed.");
  case CCL_ERROR_HMF_INTERP:
    // Terminate if hmf definition not supported
    ccl_raise_exception(*status, 
          "CCL_ERROR_HMF_INTERP: Halo mass function definition not supported.");
  case CCL_ERROR_NU_INT:
    // Error in getting the neutrino integral spline: exit. No status_message 
    // in cosmo because can't pass cosmology to the function.
    ccl_raise_exception(*status, 
      "CCL_ERROR_NU_INT: Error getting the neutrino phase-space integral spline.");
  case CCL_ERROR_NU_SOLVE:
    // Error in converting Omeganuh2-> Mnu: exit. No status_message in cosmo 
    // because can't pass cosmology to the function.
    ccl_raise_exception(*status, 
                      "CCL_ERROR_NU_SOLVE: Error converting Omeganuh2 -> Mnu.");
  case CCL_ERROR_MNU_UNPHYSICAL:
    // Error in the sum of mnu or Omeganu passed for the hierarchy requested.
	  ccl_raise_exception(*status, 
      "CCL_ERROR_MNU_UNPHYSICAL: Sum of neutrinos masses for this Omeganu value is incompatible with the requested mass hierarchy.");
  case CCL_ERROR_NOT_IMPLEMENTED: 
    ccl_raise_exception(*status, 
      "CCL_ERROR_NOT_IMPLEMENTED: the type of m_nu specified is not supported.");
  default:
    ccl_raise_exception(*status, 
             "Unrecognized error code (see gsl_errno.h for error codes 1-32).");
  }
}
