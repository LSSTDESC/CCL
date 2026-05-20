#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include <gsl/gsl_errno.h>

#include "ccl.h"

// Debug mode policy: whether to print error messages as they are raised.
// Defualt is ON.
static CCLDebugModePolicy _ccl_debug_mode_policy = CCL_DEBUG_MODE_OFF;

// Set debug mode policy
void ccl_set_debug_policy(CCLDebugModePolicy debug_policy) {
    _ccl_debug_mode_policy = debug_policy;
}

// Convenience function to handle warnings
void ccl_raise_warning(int err, const char* msg, ...) {
  char message[256];

  va_list va;
  va_start(va, msg);
  vsnprintf(message, 250, msg, va);
  va_end(va);

  // For now just print warning to stderr if debug is enabled.
  // TODO: Implement some kind of error stack that can be passed on to, e.g.,
  // the python binding.
  if (_ccl_debug_mode_policy == CCL_DEBUG_MODE_ON) {
    fprintf(stderr, "WARNING %d: %s\n", err, message);
  }
}

// Convenience function to handle warnings
void ccl_raise_gsl_warning(int gslstatus, const char* msg, ...) {
  char message[256];

  va_list va;
  va_start(va, msg);
  vsnprintf(message, 250, msg, va);
  va_end(va);

  ccl_raise_warning(gslstatus, "%s: GSL ERROR: %s", message, gsl_strerror(gslstatus));
  return;
}
