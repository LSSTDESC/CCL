%module ccllib

%{

#define SWIG_FILE_WITH_INIT
#include "../include/ccl.h"
#include "../include/ccl_config.h"
#include "../include/ccl_error.h"
#include "../include/ccl_utils.h"

#include "../class/include/class.h"

%}

// Enable numpy array support
%include "numpy.i"
%init %{
    import_array();
%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

// Flag status variable as input/output variable
%apply (int* INOUT) {(int * status)};

%include "../include/ccl.h"
//%include "../include/ccl_background.h"

%include "ccl_core.i"
%include "ccl_background.i"
%include "ccl_power.i"
%include "ccl_massfunc.i"
%include "ccl_cls.i"
%include "ccl_constants.i"
%include "ccl_lsst_specs.i"

%include "../include/ccl_config.h"
%include "../include/ccl_error.h"
%include "../include/ccl_utils.h"

%import "../class/include/class.h"

// We need this construct to handle some memory allocation scariness. By 
// specifying the size of the output array in advance, we can avoid having to 
// manage the memory manually, as swig will just do the right thing when this 
// information is available. (Manual memory management is possible, but the 
// Python/swig docs say it's dangerous.) Unfortunately, this construct means we 
// have to pass an integer specifying the output array size every time. So, 
// this requires an extra level of wrapping, in which we just pass the size of 
// the input array in as that argument.
/*
%inline %{
void function_vec(ccl_cosmology * cosmo, 
                               double* a, int na,
                               double* output, int nout) {
    ccl_function(cosmo, na, a, output);
}
%}
*/

