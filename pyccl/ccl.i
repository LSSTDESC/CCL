%module ccllib

%{

#define SWIG_FILE_WITH_INIT
#include "../include/ccl.h"
#include "../include/ccl_config.h"
#include "../include/ccl_error.h"
#include "../include/ccl_utils.h"

%}

// Enable numpy array support and Python exception handling
%include "numpy.i"
%init %{
    import_array();
    // Tell CCL library not to quit when an error is thrown (to let Python 
    // exception handler take over)
    ccl_set_error_policy(CCL_ERROR_POLICY_CONTINUE);
%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

// Flag status variable as input/output variable
%apply (int* INOUT) {(int * status)};

%include "../include/ccl.h"

%include "ccl_core.i"
%include "ccl_background.i"
%include "ccl_power.i"
%include "ccl_correlation.i"
%include "ccl_massfunc.i"
%include "ccl_cls.i"
%include "ccl_constants.i"
%include "ccl_lsst_specs.i"
%include "ccl_neutrinos.i"
%include "ccl_params.i"

%include "../include/ccl_config.h"
%include "../include/ccl_error.h"
%include "../include/ccl_utils.h"

