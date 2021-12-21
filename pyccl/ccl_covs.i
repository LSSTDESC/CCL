%module ccl_covs

%{
/* put additional #include here */
%}

%include "../include/ccl_cls.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* ell1, int nell1)};
%apply (double* IN_ARRAY1, int DIM1) {(double* ell2, int nell2)};
%apply (double* IN_ARRAY1, int DIM1) {(double* s2b, int ns2b)};
%apply (double* IN_ARRAY1, int DIM1) {(double* a, int na)};
%apply (double* IN_ARRAY1, int DIM1) {(double* R, int nR)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%feature("pythonprepend") sigma2b_vec %{
    if len(R) != nout:
        raise CCLError("Input shape for `R` must match `(nout,)`!")
%}

%inline %{

void sigma2b_vec(ccl_cosmology * cosmo,
                 double *a, int na,
                 double *R, int nR,
                 ccl_f2d_t *psp,
                 int nout, double* output,
                 int *status)
{
  ccl_sigma2Bs(cosmo, na, a, R, output, psp, status);
}

%}

%feature("pythonprepend") angular_cov_vec %{
    if len(ell1)*len(ell2) != nout:
        raise CCLError("Input shape for `ell1` and `ell2` must match `(nout,)`!")
%}

%inline %{

void angular_cov_vec(ccl_cosmology * cosmo,
                     ccl_cl_tracer_collection_t *clt1,
                     ccl_cl_tracer_collection_t *clt2,
                     ccl_cl_tracer_collection_t *clt3,
                     ccl_cl_tracer_collection_t *clt4,
                     ccl_f3d_t *tspec,
                     double* ell1, int nell1,
                     double* ell2, int nell2,
                     int integration_type,
                     int chi_exponent, double prefac,
                     int nout, double* output,
                     int *status)
{
  ccl_angular_cl_covariance(cosmo, clt1, clt2, clt3, clt4, tspec,
                            nell1,  ell1, nell2, ell2, output,
                            integration_type, chi_exponent, NULL,
                            prefac, status);
}

%}

%feature("pythonprepend") angular_cov_ssc_vec %{
    if len(ell1)*len(ell2) != nout:
        raise CCLError("Input shape for `ell1` and `ell2` must match `(nout,)`!")
%}

%inline %{

void angular_cov_ssc_vec(ccl_cosmology * cosmo,
                         ccl_cl_tracer_collection_t *clt1,
                         ccl_cl_tracer_collection_t *clt2,
                         ccl_cl_tracer_collection_t *clt3,
                         ccl_cl_tracer_collection_t *clt4,
                         ccl_f3d_t *tspec,
                         double *a, int na, 
                         double *s2b, int ns2b, 
                         double* ell1, int nell1,
                         double* ell2, int nell2,
                         int integration_type,
                         int chi_exponent, double prefac,
                         int nout, double* output,
                         int *status)
{
  ccl_f1d_t *s2b_f=ccl_f1d_t_new(na, a, s2b,
                                 s2b[0], s2b[na-1],
                                 0, 0, status);
  ccl_angular_cl_covariance(cosmo, clt1, clt2, clt3, clt4, tspec,
                            nell1,  ell1, nell2, ell2, output,
                            integration_type, chi_exponent, s2b_f,
                            prefac, status);
  ccl_f1d_t_free(s2b_f);
}

%}
