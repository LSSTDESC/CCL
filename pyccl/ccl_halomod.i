%module ccl_halomod

%{
/* put additional #include here */
#include "../include/ccl_halomod.h"
%}

%include "../include/ccl_halomod.h"

 // Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* k, int nk)};
%apply (double* IN_ARRAY1, int DIM1) {(double* halo_mass, int nm)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

/* The python code here will be executed before all of the functions that
   follow this directive. */
%feature("pythonprepend") %{
    if numpy.shape(k) != (nout,):
        raise CCLError("Input shape for `k` must match `(nout,)`!")
%}

%inline %{

void onehalo_matter_power_vec(ccl_cosmology *cosmo, double a, double* k, int nk,
                              int nout, double* output, int *status) {
    for(int i=0; i < nk; i++) {
        output[i] = ccl_onehalo_matter_power(cosmo, k[i], a, status);
    }
}

void twohalo_matter_power_vec(ccl_cosmology *cosmo, double a, double* k, int nk,
                              int nout, double* output, int *status) {
    for(int i=0; i < nk; i++) {
        output[i] = ccl_twohalo_matter_power(cosmo, k[i], a, status);
    }
}

void halomodel_matter_power_vec(ccl_cosmology *cosmo, double a, double* k, int nk,
                                int nout, double* output, int *status) {
    for(int i=0; i < nk; i++) {
        output[i] = ccl_halomodel_matter_power(cosmo, k[i], a, status);
    }
}

%}

%feature("pythonprepend") halo_concentration_vec %{
    if numpy.shape(halo_mass) != (nout,):
        raise CCLError("Input shape for `halo_mass` must match `(nout,)`!")
%}

%inline %{
void halo_concentration_vec(ccl_cosmology *cosmo, double a, double odelta,
                            double* halo_mass, int nm, int nout, double* output,
                            int *status) {
    for(int i=0; i < nm; i++) {
        output[i] = ccl_halo_concentration(cosmo, halo_mass[i], a, odelta, status);
    }
}

%}

/* The directive gets carried between files, so we reset it at the end. */
%feature("pythonprepend") %{ %}
