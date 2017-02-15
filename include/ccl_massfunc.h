#pragma once

#include "ccl_core.h"

void ccl_cosmology_compute_sigma(ccl_cosmology * cosmo, int * status);
double ccl_massfunc(ccl_cosmology * cosmo, double smooth_mass, double redshift, int * status);
double ccl_halo_bias(ccl_cosmology *cosmo, double smooth_mass, double redshift, int * status);
double ccl_massfunc_m2r(ccl_cosmology * cosmo, double smooth_mass, int * status);
double ccl_sigmaM(ccl_cosmology * cosmo, double smooth_mass, double redshift, int * status);
