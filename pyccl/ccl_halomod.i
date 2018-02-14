%module ccl_halomod

%{
#define SWIG_FILE_WITH_INIT
#include "../include/ccl_halomod.h"
%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

%include "../include/ccl_halomod.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* k, int nm)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout)};

%inline %{
void p_1h_vec(ccl_cosmology * cosmo,
                    double* k, int nm, double a,
                    double* output, int nout,
                    int* status)
{
    assert(nout == nm);
    for(int i=0; i < nm; i++){
        output[i] = p_1h(cosmo, k[i], a, status);
    }
}
%}
