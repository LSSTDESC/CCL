%module ccl_massfunc

%{
#define SWIG_FILE_WITH_INIT
#include "../include/ccl_massfunc.h"
%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

%include "../include/ccl_massfunc.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* halo_mass, int nm)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout)};

%inline %{
void massfunc_vec(ccl_cosmology * cosmo,
                    double redshift,
                    double* halo_mass, int nm,
                    double* output, int nout)
{
    assert(nout == nm);
    for(int i=0; i < nm; i++){
        output[i] = ccl_massfunc(cosmo, halo_mass[i], redshift);
    }
}

void massfunc_m2r_vec(ccl_cosmology * cosmo,
                        double* halo_mass, int nm,
                        double* output, int nout)
{
    assert(nout == nm);
    for(int i=0; i < nm; i++){
        output[i] = ccl_massfunc_m2r(cosmo, halo_mass[i]);
    }
}

void sigmaM_vec(ccl_cosmology * cosmo,
                    double redshift,
                    double* halo_mass, int nm,
                    double* output, int nout)
{
    assert(nout == nm);
    for(int i=0; i < nm; i++){
        output[i] = ccl_sigmaM(cosmo, halo_mass[i], redshift);
    }
}
%}
