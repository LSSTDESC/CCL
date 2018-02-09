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
%apply (double* IN_ARRAY1, int DIM1) {(double* halo_mass, int nm)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout)};

%inline %{
void u_nfw_c_vec(ccl_cosmology * cosmo,
                    double c, double* halo_mass,
                    int nm, double k, double a,
                    double* output, int nout,
                    int* status)
{
    assert(nout == nm);
    for(int i=0; i < nm; i++){
        output[i] = ccl_u_nfw_c(cosmo, c, halo_mass[i], k, a, status);
    }
}
%}
