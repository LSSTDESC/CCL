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
            (double* dfarr, int nf),
            (double* m_nu, int n_m)
};

%inline %{

ccl_parameters parameters_create_nu(
                        double Omega_c, double Omega_b, double Omega_k, 
                        double Neff, double w0, double wa, double h, 
                        double norm_pk, double n_s, double bcm_log10Mc, 
                        double bcm_etab, double bcm_ks, int mnu_is_sum, 
                        double* m_nu, int n_m, int* status)
{
    return ccl_parameters_create(
                        Omega_c, Omega_b, Omega_k, Neff, m_nu, mnu_is_sum, 
                        w0, wa, h, norm_pk, n_s, bcm_log10Mc, bcm_etab, 
                        bcm_ks, -1, NULL, NULL, status );
}


ccl_parameters parameters_create_nu_vec(
                        double Omega_c, double Omega_b, double Omega_k, 
                        double Neff, double w0, double wa, double h, 
                        double norm_pk, double n_s, double bcm_log10Mc, 
                        double bcm_etab, double bcm_ks, 
                        double* zarr, int nz,
                        double* dfarr, int nf, int mnu_is_sum, double* m_nu, 
                        int n_m, int* status)
{
    assert(nz == nf);
    if (nz == 0){ nz = -1; }
    return ccl_parameters_create(
                        Omega_c, Omega_b, Omega_k, Neff, m_nu, mnu_is_sum, 
                        w0, wa, h, norm_pk, n_s, bcm_log10Mc, bcm_etab, bcm_ks,
                        nz, zarr, dfarr, status);
}

%}
