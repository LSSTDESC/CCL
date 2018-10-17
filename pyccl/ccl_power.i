%module ccl_power

%{
/* put additional #include here */
%}

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* k, int nk)};
%apply (double* IN_ARRAY1, int DIM1) {(double* R, int nR)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%include "../include/ccl_power.h"

/* The python code here will be executed before all of the functions that
   follow this directive. */
%feature("pythonprepend") %{
    if numpy.shape(k) != (nout,):
        raise CCLError("Input shape for `k` must match `(nout,)`!")
%}

%inline %{
void linear_matter_power_vec(ccl_cosmology * cosmo, double a, double* k, int nk,
                             int nout, double* output, int* status) {
    for(int i=0; i < nk; i++){
      output[i] = ccl_linear_matter_power(cosmo, k[i], a, status);
    }
}

void nonlin_matter_power_vec(ccl_cosmology * cosmo, double a, double* k, int nk,
                             int nout, double* output, int* status) {
    for(int i=0; i < nk; i++){
      output[i] = ccl_nonlin_matter_power(cosmo, k[i], a, status);
    }
}

%}

/* The python code here will be executed before all of the functions that
   follow this directive. */
%feature("pythonprepend") %{
    if numpy.shape(R) != (nout,):
        raise CCLError("Input shape for `R` must match `(nout,)`!")
%}

%inline %{

void sigmaR_vec(ccl_cosmology * cosmo, double a, double* R, int nR,
                int nout, double* output, int *status) {
    for(int i=0; i < nR; i++){
      output[i] = ccl_sigmaR(cosmo, R[i], a, status);
    }
}

void sigmaV_vec(ccl_cosmology * cosmo, double a, double* R, int nR,
                int nout, double* output, int *status) {
    for(int i=0; i < nR; i++){
      output[i] = ccl_sigmaV(cosmo, R[i], a, status);
    }
}

%}

/* The directive gets carried between files, so we reset it at the end. */
%feature("pythonprepend") %{ %}
