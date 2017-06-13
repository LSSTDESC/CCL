#pragma once

#define A_MIN_EMU 1./3.
#define K_MAX_EMU 5.0
#define K_MIN_EMU 1.e-3

void ccl_pkemu(double *xstarin, double **Pkemu, int *status);
