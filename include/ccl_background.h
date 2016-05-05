#pragma once
#include "ccl_core.h"

//TODO: why is there no status here?
double ccl_comoving_radial_distance(ccl_cosmology * cosmo, double a);
int ccl_comoving_radial_distances(ccl_cosmology * cosmo, int na, double a[na], double output[na]);

double ccl_luminosity_distance(ccl_cosmology * cosmo, double a);
int ccl_luminosity_distances(ccl_cosmology * cosmo, int na, double a[na], double output[na]);

double ccl_growth_factor(ccl_cosmology * cosmo, double a, int *status);
int ccl_growth_factors(ccl_cosmology * cosmo, int na, double a[na], double output[na]);

double ccl_growth_rate(ccl_cosmology * cosmo, double a, int *status);
int ccl_growth_rates(ccl_cosmology * cosmo, int na, double a[na], double output[na]);

