#pragma once

#include "ccl_core.h"

void ccl_cosmology_compute_sigma(ccl_cosmology * cosmo);
double ccl_massfunc(ccl_cosmology * cosmo, double halo_mass, double redshift);
double ccl_massfunc_m2r(ccl_cosmology * cosmo, double halo_mass);
double ccl_sigmaM(ccl_cosmology * cosmo, double halo_mass, double redshift);
