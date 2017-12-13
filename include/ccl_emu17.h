#pragma once

#define A_MIN_EMU 1./3.
#define K_MAX_EMU 5.0
#define K_MIN_EMU 1.0000000474974513E-003

#pragma once
#include "ccl_core.h"

void ccl_pkemu(double *xstarin, double **Pkemu, int *status, ccl_cosmology* cosmo);
