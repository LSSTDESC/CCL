%module ccl_correlation

%{
/* put additional #include here */
%}

%include "../include/ccl_correlation.h"

// Enable vectorised arguments for arrays
%apply (int DIM1, double* IN_ARRAY1) {
    (int nlarr, double* larr),
    (int nclarr, double* clarr),
    (int nt, double *theta),
    (int nr, double *r)}
%apply (double* ARGOUT_ARRAY1, int DIM1) {
    (double* output, int nout),
    (double* xi, int nxi)};

%feature("pythonprepend") correlation_vec %{
    if numpy.shape(larr) != numpy.shape(clarr):
        raise CCLError("Input shape for `larr` must match `clarr`!")

    if numpy.shape(theta) != (output,):
        raise CCLError("Input shape for `theta` must match `(output,)`!")
%}

%feature("pythonprepend") correlation_3d_vec %{
    if numpy.shape(r) != (xi,):
        raise CCLError("Input shape for `r` must match `(xi,)`!")
%}


%inline %{

void correlation_vec(ccl_cosmology *cosmo,
                     int nlarr, double *larr,
                     int nclarr, double *clarr,
                     int nt, double *theta,
                     int corr_type, int method,
                     double *output, int nout,
                     int *status) {
    ccl_correlation(
        cosmo, nlarr, larr, clarr, nt, theta,
        output, corr_type, 0, NULL, method, status);
}

void correlation_3d_vec(ccl_cosmology *cosmo,double a,
                        int nr, double *r,
                        double *xi, int nxi,
                        int *status)
{
  ccl_correlation_3d(cosmo, a, nr, r, xi, 0, NULL, status);
}
%}
