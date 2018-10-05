%module ccl_core

%{
/* put additional #include here */
%}

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

%}

/* The python code here will be executed before all of the functions that
   follow this directive. */
%feature("pythonprepend") %{
    if numpy.shape(zarr) != numpy.shape(dfarr):
        raise CCLError("Input shape for `zarr` must match `dfarr`!")
%}

%inline %{

ccl_parameters parameters_create_nu_vec(
                        double Omega_c, double Omega_b, double Omega_k,
                        double Neff, double w0, double wa, double h,
                        double norm_pk, double n_s, double bcm_log10Mc,
                        double bcm_etab, double bcm_ks,
                        double* zarr, int nz,
                        double* dfarr, int nf, int mnu_is_sum, double* m_nu,
                        int n_m, int* status)
{
    if (nz == 0){ nz = -1; }
    return ccl_parameters_create(
                        Omega_c, Omega_b, Omega_k, Neff, m_nu, mnu_is_sum,
                        w0, wa, h, norm_pk, n_s, bcm_log10Mc, bcm_etab, bcm_ks,
                        nz, zarr, dfarr, status);
}

%}

/* The directive gets carried between files, so we reset it at the end. */
%feature("pythonprepend") %{ %}
