%module ccllib
/* master file for the CCL swig module;
 * all other .i files are included by this file
 * producing a single .c file that is compiled to
 * a python extension module. */

%pythonbegin %{
import numpy
from .errors import CCLError
%}

%{
/* this is the master .c file; need an init function */
#define SWIG_FILE_WITH_INIT
/* must include the file explicitly */
#include "../include/ccl.h"
%}

// Enable numpy array support and Python exception handling
%include "numpy.i"
%init %{
    import_array();
    // Tell CCL library not to quit when an error is thrown (to let Python
    // exception handler take over)
    ccl_set_error_policy(CCL_ERROR_POLICY_CONTINUE);
    // Tell CCL to not print to stdout/stderr for debugging.
    ccl_set_debug_policy(CCL_DEBUG_MODE_OFF);
%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

// Flag status variable as input/output variable
%apply (int* INOUT) {(int * status)};

/* must scan this file for other scans to work */
/* although ccl.h includes ccl_defs.h swig does not remember macros defined nestedly. */
%include "../include/ccl_defs.h"

%include "../include/ccl.h"

%include "ccl_core.i"
%include "ccl_p2d.i"
%include "ccl_background.i"
%include "ccl_power.i"
%include "ccl_correlation.i"
%include "ccl_massfunc.i"
%include "ccl_cls.i"
%include "ccl_constants.i"
%include "ccl_lsst_specs.i"
%include "ccl_neutrinos.i"
%include "ccl_halomod.i"
%include "ccl_params.i"

/* list header files not yet having a .i file here */
%include "../include/ccl_config.h"
%include "../include/ccl_error.h"
%include "../include/ccl_utils.h"
