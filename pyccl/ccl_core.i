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
                        double N_nu_rel, double N_nu_mass, double M_nu, double w0, double wa, double h, 
                        double norm_pk, double n_s, double bcm_log10Mc, double bcm_etab, double bcm_ks, 
                        double* zarr, int nz,
                        double* dfarr, int nf,int* status)
{
    assert(nz == nf);
    if (nz == 0){ nz = -1; }
    return ccl_parameters_create(Omega_c, Omega_b, Omega_k, N_nu_rel, N_nu_mass, M_nu, 
                                 w0, wa, h, norm_pk, n_s, bcm_log10Mc, bcm_etab, bcm_ks,
                                 nz, zarr, dfarr, status);
    
    
}
%}
