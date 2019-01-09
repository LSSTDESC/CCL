%module ccl_massfunc

%{
/* put additional #include here */
%}

%include "../include/ccl_massfunc.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* halo_mass, int nm)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

/* The python code here will be executed before all of the functions that
   follow this directive. */
%feature("pythonprepend") %{
    if numpy.shape(halo_mass) != (nout,):
        raise CCLError("Input shape for `halo_mass` must match `(nout,)`!")
%}

%inline %{
void massfunc_vec(ccl_cosmology * cosmo, double a, double odelta,
                  double* halo_mass, int nm, int nout, double* output, int* status) {
    for(int i=0; i < nm; i++) {
        output[i] = ccl_massfunc(cosmo, halo_mass[i], a, odelta, status);
    }
}

void massfunc_m2r_vec(ccl_cosmology * cosmo, double* halo_mass, int nm,
                      int nout, double* output, int* status) {
    for(int i=0; i < nm; i++) {
        output[i] = ccl_massfunc_m2r(cosmo, halo_mass[i], status);
    }
}

void sigmaM_vec(ccl_cosmology * cosmo, double a, double* halo_mass, int nm,
                int nout, double* output, int* status) {
    for(int i=0; i < nm; i++) {
        output[i] = ccl_sigmaM(cosmo, halo_mass[i], a, status);
    }
}

void halo_bias_vec(ccl_cosmology * cosmo, double a, double odelta,
                   double* halo_mass, int nm, int nout, double* output,
                   int* status) {
    for(int i=0; i < nm; i++) {
        output[i] = ccl_halo_bias(cosmo, halo_mass[i], a, odelta, status);
    }
}

%}

/* The directive gets carried between files, so we reset it at the end. */
%feature("pythonprepend") %{ %}
