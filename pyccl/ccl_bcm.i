%module ccl_bcm

%{
/* put additional #include here */
%}

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* k, int nk)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%include "../include/ccl_bcm.h"

/* The python code here will be executed before all of the functions that
   follow this directive. */
%feature("pythonprepend") %{
    if numpy.shape(k) != (nout,):
        raise CCLError("Input shape for `k` must match `(nout,)`!")
%}

%inline %{
void bcm_model_fka_vec(ccl_cosmology * cosmo, double a, double* k, int nk,
                             int nout, double* output, int* status) {
    for(int i=0; i < nk; i++){
      output[i] = ccl_bcm_model_fka(cosmo, k[i], a, status);
    }
}

%}

/* The directive gets carried between files, so we reset it at the end. */
%feature("pythonprepend") %{ %}
