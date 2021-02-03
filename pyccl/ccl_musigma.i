%module ccl_musigma

%{
/* put additional #includes here */
%}

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* a, int na)};
%apply (double* IN_ARRAY1, int DIM1) {(double* k, int nk)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%include "../include/ccl_musigma.h"

/* The python code here will be executed before all of the functions that
   follow this directive. */
/* unit tests and MG functions definitions */
%feature("pythonprepend") %{
    if (len(a)*len(k)) != nout:
        raise CCLError("Input length for `a` times `k` must match `nout`!")
%}

%inline %{
void mu_MG_vec(ccl_cosmology * cosmo, double* a, int na, double* k, int nk,
               int nout, double* output, int *status) {
    int index;
    int _status;
    assert(nout == na*nk);
    for(int j=0; j < nk; j++){
            index = j * na; // to get correct indexing.
            for(int i=0; i < na; i++){
                _status = 0;
                output[i+index] = ccl_mu_MG(cosmo, a[i], k[j], &_status);
                *status |= _status;
// printf("index-i=%d, index-j=%d, index=%d, Sigma=%f, a=%f, k=%f \n", i, j, index, output[i+index], a[i], k[j]);
            }
    }
}

void Sig_MG_vec(ccl_cosmology * cosmo, double* a, int na, double* k, int nk,
                int nout, double* output, int *status) {
    int index;
    int _status;
    assert(nout == na*nk);
    for(int j=0; j < nk; j++){
            index = j * na; // to get correct indexing.
            for(int i=0; i < na; i++){
                _status = 0;
                output[i+index] = ccl_Sig_MG(cosmo, a[i], k[j], &_status);
                *status |= _status;
//printf("index-i=%d, index-j=%d, index=%d, Sigma=%f, a=%f, k=%f \n", i, j, index, output[i+index], a[i], k[j]);
            }
     }
}
%}
/* End of MG functions and unit tests */
