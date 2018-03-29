/** @file */
#ifdef __cplusplus
extern "C" {
#endif

#pragma once

#include "ccl_core.h"

  /*
double u_nfw_c(ccl_cosmology *cosmo, double c, double halomass, double k, double a, int * status);
  */

double ccl_p_1h(ccl_cosmology *cosmo, double k, double a, int * status);

double ccl_p_2h(ccl_cosmology *cosmo, double k, double a, int * status);

double ccl_p_halomod(ccl_cosmology *cosmo, double k, double a, int * status);

double ccl_halo_concentration(ccl_cosmology *cosmo, double halomass, double a, int * status);
  
#ifdef __cplusplus
}
#endif
