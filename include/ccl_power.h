#pragma once

#include "ccl_core.h"

double ccl_linear_matter_power(ccl_cosmology * cosmo, double a, double k);
double ccl_nonlin_matter_power(ccl_cosmology * cosmo, double a, double k);
void ccl_linear_matter_powers(ccl_cosmology * cosmo, int n, double a[n], double k[n], double output[n]);

double ccl_sigmaR(ccl_cosmology *cosmo, double R);
double ccl_sigma8(ccl_cosmology *cosmo);

