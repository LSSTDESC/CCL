%module ccl_lsst_specs

%{
#define SWIG_FILE_WITH_INIT
#include "../include/ccl_lsst_specs.h"
%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

%include "../include/ccl_lsst_specs.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {
        (double* a, int na),
        (double* z, int nz)
};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout)};
%apply double* OUTPUT { double* tomoout };

%include "../include/ccl_lsst_specs.h"

%inline %{

void specs_bias_clustering_vec(
                        ccl_cosmology * cosmo,
                        double* a, int na,
                        double* output, int nout)
{
    assert(nout == na);
    for(int i=0; i < na; i++){
        output[i] = ccl_specs_bias_clustering(cosmo, a[i]);
    }
}

void specs_sigmaz_clustering_vec(
                        double* z, int nz,
                        double* output, int nout)
{
    assert(nout == nz);
    for(int i=0; i < nz; i++){
        output[i] = ccl_specs_sigmaz_clustering(z[i]);
    }
}

void specs_sigmaz_sources_vec(
                        double* z, int nz,
                        double* output, int nout)
{
    assert(nout == nz);
    for(int i=0; i < nz; i++){
        output[i] = ccl_specs_sigmaz_sources(z[i]);
    }
}

%}
