#pragma once

#include "ccl_core.h"

// Power function examples
void ccl_cosmology_compute_power_bbks(ccl_cosmology * cosmo, int *status);
void ccl_cosmology_compute_power_class(ccl_cosmology * cosmo, int *status);

double ccl_linear_matter_power(ccl_cosmology * cosmo, double a, double k, int * status);
double ccl_nonlin_matter_power(ccl_cosmology * cosmo, double a, double k, int * status);
int ccl_linear_matter_powers(ccl_cosmology * cosmo, int n, double a[n], double k[n], double output[n]);

