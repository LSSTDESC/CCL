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
