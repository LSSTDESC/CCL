%module ccl_constants

%{
/* put additional #include here */
%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

/* FIXME: do we need to scan this gsl module ? */
%include "gsl/gsl_const_mksa.h"
%include "../include/ccl_constants.h"
