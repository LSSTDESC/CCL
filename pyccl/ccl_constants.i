%module ccl_constants

%{

#define SWIG_FILE_WITH_INIT
#include "../include/ccl_constants.h"

%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

%include "../include/ccl_constants.h"

