%module ccllib

%{

#define SWIG_FILE_WITH_INIT
#include "../include/ccl.h"
#include "../include/ccl_background.h"
#include "../include/ccl_cls.h"
#include "../include/ccl_config.h"
#include "../include/ccl_constants.h"
#include "../include/ccl_core.h"
#include "../include/ccl_error.h"
#include "../include/ccl_lsst_specs.h"
#include "../include/ccl_massfunc.h"
#include "../include/ccl_placeholder.h"
#include "../include/ccl_utils.h"
#include "../include/ccl_power.h"

%}

// Enable numpy array support
%include "numpy.i"
%init %{
    import_array();
%}

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* a, int na)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout)};

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

%include "../include/ccl.h"
%include "../include/ccl_background.h"
%include "../include/ccl_cls.h"
%include "../include/ccl_config.h"
%include "../include/ccl_constants.h"
%include "../include/ccl_core.h"
%include "../include/ccl_error.h"
%include "../include/ccl_lsst_specs.h"
%include "../include/ccl_massfunc.h"
%include "../include/ccl_placeholder.h"
%include "../include/ccl_utils.h"
%include "../include/ccl_power.h"

// We need this construct to handle some memory allocation scariness. By 
// specifying the size of the output array in advance, we can avoid having to 
// manage the memory manually, as swig will just do the right thing when this 
// information is available. (Manual memory management is possible, but the 
// Python/swig docs say it's dangerous.) Unfortunately, this construct means we 
// have to pass an integer specifying the output array size every time. So, 
// this requires an extra level of wrapping, in which we just pass the size of 
// the input array in as that argument.
%inline %{
void growth_factor_vec(ccl_cosmology * cosmo, 
                               double* a, int na,
                               double* output, int nout) {
    ccl_growth_factors(cosmo, na, a, output);
}
%}

