%module ccl_power

%{
#define SWIG_FILE_WITH_INIT
#include "../include/ccl_power.h"
%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* k, int nk)};
%apply (double* IN_ARRAY1, int DIM1) {(double* R, int nR)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout)};

%include "../include/ccl_power.h"

%inline %{
void linear_matter_power_vec(
                        ccl_cosmology * cosmo,
                        double a,
                        double* k, int nk,
                        double* output, int nout,
                        int* status)
{
    assert(nout == nk);
    for(int i=0; i < nk; i++){
      output[i] = ccl_linear_matter_power(cosmo, k[i], a, status);
    }
}

void nonlin_matter_power_vec(
                        ccl_cosmology * cosmo,
                        double a,
                        double* k, int nk,
                        double* output, int nout,
                        int* status)
{
    assert(nout == nk);
    for(int i=0; i < nk; i++){
      output[i] = ccl_nonlin_matter_power(cosmo, k[i], a, status);
    }
}

void sigmaR_vec(ccl_cosmology * cosmo, 
                        double* R, int nR,
                        double* output, int nout, int *status)
{
    assert(nout == nR);
    for(int i=0; i < nR; i++){
        output[i] = ccl_sigmaR(cosmo, R[i], status);
    }
}

%}
