%module ccl_constants

%{
/* put additional #include here */
%}

/* FIXME: do we need to scan this gsl module ? */
%include "gsl/gsl_const_mksa.h"
%include "../include/ccl_constants.h"
