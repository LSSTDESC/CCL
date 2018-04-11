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
%apply (double* IN_ARRAY1, int DIM1) {(double* k, int nk)};
%apply (double* IN_ARRAY1, int DIM1) {(double* halo_mass, int nm)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout)};

%inline %{
void p_1h_vec(ccl_cosmology * cosmo,
                    double a,
                    double* k, int nk,
                    double* output, int nout,
                    int* status)
{
    assert(nout == nk);
    for(int i=0; i < nk; i++){
        output[i] = ccl_p_1h(cosmo, k[i], a, status);
    }
}

void p_2h_vec(ccl_cosmology * cosmo,
                    double a,
                    double* k, int nk,
                    double* output, int nout,
                    int* status)
{
    assert(nout == nk);
    for(int i=0; i < nk; i++){
        output[i] = ccl_p_2h(cosmo, k[i], a, status);
    }
}

void p_halomod_vec(ccl_cosmology * cosmo,
                     double a,
                     double* k, int nk,
                     double* output, int nout,
                     int* status)
{
    assert(nout == nk);
    for(int i=0; i < nk; i++){
        output[i] = ccl_p_halomod(cosmo, k[i], a, status);
    }
}

void halo_concentration_vec(ccl_cosmology * cosmo,
                                 double a,
                                 double* halo_mass, int nm,
                                 double* output, int nout,
                                 int* status)
{
    assert(nout == nm);
    for(int i=0; i < nm; i++){
        output[i] = ccl_halo_concentration(cosmo, halo_mass[i], a, status);
    }
}

%}
