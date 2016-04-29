#pragma once
#include "ccl_core.h"
double ccl_bbks_power(ccl_parameters * params, double k);
double ccl_sigma8(gsl_spline * P, double h, int * status);
