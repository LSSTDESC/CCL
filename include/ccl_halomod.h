/** @file */
#ifdef __cplusplus
extern "C" {
#endif

#pragma once

#include "ccl_core.h"

double u_nfw_c(ccl_cosmology *cosmo, double c, double halomass, double k, double a, int * status);

#ifdef __cplusplus
}
#endif
