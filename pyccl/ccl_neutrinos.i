%module ccl_neutrinos

%{
#define SWIG_FILE_WITH_INIT
#include "../include/ccl_neutrinos.h"
%}

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

%include "../include/ccl_neutrinos.h"

