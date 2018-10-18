%module ccl_correlation

%{
/* put additional #include here */
%}

%include "../include/ccl_correlation.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {
    (double* larr, int nlarr),
    (double* clarr, int nclarr),
    (double* theta, int nt),
    (double* r, int nr)}
%apply (int DIM1, double* ARGOUT_ARRAY1) {
    (int nout, double* output),
    (int nxi, double* xi)};

%feature("pythonprepend") correlation_vec %{
    if numpy.shape(larr) != numpy.shape(clarr):
        raise CCLError("Input shape for `larr` must match `clarr`!")

    if numpy.shape(theta) != (nout,):
        raise CCLError("Input shape for `theta` must match `(nout,)`!")
%}

%feature("pythonprepend") correlation_3d_vec %{
    if numpy.shape(r) != (nxi,):
        raise CCLError("Input shape for `r` must match `(nxi,)`!")
%}


%inline %{

void correlation_vec(ccl_cosmology *cosmo, double* larr, int nlarr,
                     double* clarr, int nclarr, double* theta, int nt,
                     int corr_type, int method, int nout, double* output,
                     int *status) {
    ccl_correlation(
        cosmo, nlarr, larr, clarr, nt, theta,
        output, corr_type, 0, NULL, method, status);
}

void correlation_3d_vec(ccl_cosmology *cosmo,double a, double* r, int nr,
                        int nxi, double* xi, int *status) {
  ccl_correlation_3d(cosmo, a, nr, r, xi, 0, NULL, status);
}

%}
