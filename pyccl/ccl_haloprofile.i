%module ccl_haloprofile

#include "../include/ccl_haloprofile.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* r, int nr)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};


/* The python code here will be executed before all of the functions that
   follow this directive. */
%feature("pythonprepend") %{
    if numpy.shape(r) != (nout,):
        raise CCLError("Input shape for `r` must match `(nout,)`!")
%}

%inline %{

void halo_profile_nfw_vec(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a,
                                double* r, int nr, int nout, double* output, int *status) {
        ccl_halo_profile_nfw(cosmo, c, halomass, massdef_delta_m, a, r, nr, output, status);
}

void projected_halo_profile_nfw_vec(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a,
                                double* r, int nr, int nout, double* output, int *status){
        ccl_projected_halo_profile_nfw(cosmo, c, halomass, massdef_delta_m, a, r, nr, output, status);
}

void halo_profile_einasto_vec(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a,
                                double* r, int nr, int nout, double* output, int *status) {
        ccl_halo_profile_einasto(cosmo, c, halomass, massdef_delta_m, a, r, nr, output, status);
}

void halo_profile_hernquist_vec(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a,
                                double* r, int nr, int nout, double* output, int *status) {
        ccl_halo_profile_hernquist(cosmo, c, halomass, massdef_delta_m, a, r, nr, output, status);
}

%}


/* The directive gets carried between files, so we reset it at the end. */
%feature("pythonprepend") %{ %}
