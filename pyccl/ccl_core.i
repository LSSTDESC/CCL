%module ccl_core

%{
#define SWIG_FILE_WITH_INIT
#include "../include/ccl_core.h"
%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

%include "../include/ccl_core.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {
            (double* zarr, int nz),
            (double* dfarr, int nf)
};

%inline %{
ccl_parameters parameters_create_vec(
                        double Omega_c, double Omega_b, double Omega_k, 
                        double Omega_n, double w0, double wa, double h, 
                        double norm_pk, double n_s, 
                        double* zarr, int nz,
                        double* dfarr, int nf)
{
    assert(nz == nf);
    if (nz == 0){ nz = -1; }
    return ccl_parameters_create(Omega_c, Omega_b, Omega_k, Omega_n, 
                                 w0, wa, h, norm_pk, n_s, 
                                 nz, zarr, dfarr);
}
%}
