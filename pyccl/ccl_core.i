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
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%inline %{

void parameters_get_nu_masses(ccl_parameters *params, int nout, double* output) {
    output[0] = 0;
    output[1] = 0;
    output[2] = 0;

    for (int i=0; i<params->N_nu_mass; ++i) {
        output[i] = params->m_nu[i];
    }
}

ccl_parameters parameters_create_nu(
                        double Omega_c, double Omega_b, double Omega_k,
                        double Neff, double w0, double wa, double h,
                        double norm_pk, double n_s, double bcm_log10Mc,
                        double bcm_etab, double bcm_ks, double mu_0,
                        double sigma_0, double* m_nu, int n_m, int* status)
{
    return ccl_parameters_create(
                        Omega_c, Omega_b, Omega_k, Neff, m_nu, n_m,
                        w0, wa, h, norm_pk, n_s, bcm_log10Mc, bcm_etab,
                        bcm_ks, mu_0, sigma_0, -1, NULL, NULL, status );
}

%}

%feature("pythonprepend") parameters_create_nu_vec %{
    if numpy.shape(zarr) != numpy.shape(dfarr):
        raise CCLError("Input shape for `zarr` must match `dfarr`!")
%}

%inline %{
ccl_parameters parameters_create_nu_vec(
                        double Omega_c, double Omega_b, double Omega_k,
                        double Neff, double w0, double wa, double h,
                        double norm_pk, double n_s, double bcm_log10Mc,
                        double bcm_etab, double bcm_ks, double mu_0,
                        double sigma_0, double* zarr, int nz,
                        double* dfarr, int nf, double* m_nu,
                        int n_m, int* status)
{
    if (nz == 0){ nz = -1; }
    return ccl_parameters_create(
                        Omega_c, Omega_b, Omega_k, Neff, m_nu, n_m,
                        w0, wa, h, norm_pk, n_s, bcm_log10Mc, bcm_etab, bcm_ks,
                        mu_0, sigma_0, nz, zarr, dfarr, status);
}

%}

/* The directive gets carried between files, so we reset it at the end. */
%feature("pythonprepend") %{ %}
