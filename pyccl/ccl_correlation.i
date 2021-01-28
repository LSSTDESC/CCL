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
    (double* r, int nr),
    (double* s, int ns),
    (double* sig, int nsig)}
%apply (int DIM1, double* ARGOUT_ARRAY1) {
    (int nout, double* output),
    (int nxi, double* xi),
    (int nxis, double* xis)};

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

%feature("pythonprepend") correlation_multipole_vec %{
    if numpy.shape(s) != (nxis,):
        raise CCLError("Input shape for `s` must match `(nxis,)`!")
%}

%feature("pythonprepend") correlation_3dRsd_vec %{
    if numpy.shape(s) != (nxis,):
        raise CCLError("Input shape for `s` must match `(nxis,)`!")
%}

%feature("pythonprepend") correlation_3dRsd_avgmu_vec %{
    if numpy.shape(s) != (nxis,):
        raise CCLError("Input shape for `s` must match `(nxis,)`!")
%}

%feature("pythonprepend") correlation_pi_sigma_vec %{
    if numpy.shape(sig) != (nxis,):
        raise CCLError("Input shape for `sig` must match `(nxis,)`!")
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

void correlation_3d_vec(ccl_cosmology *cosmo,ccl_f2d_t *psp,
                        double a, double* r, int nr,
                        int nxi, double* xi, int *status) {
  ccl_correlation_3d(cosmo, psp, a, nr, r, xi, 0, NULL, status);
}

void correlation_multipole_vec(ccl_cosmology *cosmo,ccl_f2d_t *psp,
                               double a,double beta,
                               int l,double *s,int ns,
                               int nxis,double *xis,
         		       int *status){
  ccl_correlation_multipole(cosmo,psp,a,beta,l,ns,s,xis,status);
}

void correlation_3dRsd_vec(ccl_cosmology *cosmo,ccl_f2d_t *psp,
                           double a,double mu,double beta,
                           double *s,int ns,
                           int nxis,double *xis,int use_spline,
                           int *status){
  ccl_correlation_3dRsd(cosmo,psp,a,ns,s,mu,beta,xis,use_spline,status);
}

void correlation_3dRsd_avgmu_vec(ccl_cosmology *cosmo,ccl_f2d_t *psp,
                                 double a,double beta,double *s,int ns,
                                 int nxis,double *xis,int *status){
  ccl_correlation_3dRsd_avgmu(cosmo,psp,a,ns,s,beta,xis,status);
}

void correlation_pi_sigma_vec(ccl_cosmology *cosmo,ccl_f2d_t *psp,
                              double a,double beta,double pie,double *sig,
                              int nsig,int nxis,double* xis,int use_spline,
                              int *status){
  ccl_correlation_pi_sigma(cosmo,psp,a,beta,pie,nsig,sig,xis,use_spline,status);
}
%}
