%module ccl_power

%{
/* put additional #include here */
%}

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* k, int nk)};
%apply (double* IN_ARRAY1, int DIM1) {(double* R, int nR)};
%apply (double* IN_ARRAY1, int DIM1) {(double* a, int na)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%include "../include/ccl_power.h"

/* The python code here will be executed before all of the functions that
   follow this directive. */
%feature("pythonprepend") %{
    if numpy.shape(R) != (nout,):
        raise CCLError("Input shape for `R` must match `(nout,)`!")
%}

%inline %{

void sigmaR_vec(ccl_cosmology * cosmo, ccl_f2d_t *psp,
                double a, double* R, int nR,
                int nout, double* output, int *status) {
    for(int i=0; i < nR; i++){
      output[i] = ccl_sigmaR(cosmo, R[i], a, psp, status);
    }
}

void sigmaV_vec(ccl_cosmology * cosmo, ccl_f2d_t *psp,
                double a, double* R, int nR,
                int nout, double* output, int *status) {
    for(int i=0; i < nR; i++){
      output[i] = ccl_sigmaV(cosmo, R[i], a, psp, status);
    }
}

%}


/* The python code here will be executed before all of the functions that
   follow this directive. */
%feature("pythonprepend") %{
    if numpy.shape(a) != (nout,):
        raise CCLError("Input shape for `a` must match `(nout,)`!")
%}

%inline %{

void kNL_vec(ccl_cosmology * cosmo, ccl_f2d_t *psp,
             double* a,  int na,
             int nout, double* output, int *status) {
    assert(nout == na);
    for(int i=0; i < na; i++){
      output[i] = ccl_kNL(cosmo, a[i], psp, status);
    }
}

%}

/* The directive gets carried between files, so we reset it at the end. */
%feature("pythonprepend") %{ %}
