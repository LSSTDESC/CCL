%module ccl_covs

%{
/* put additional #include here */
%}

%include "../include/ccl_cls.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* ell1, int nell1)};
%apply (double* IN_ARRAY1, int DIM1) {(double* ell2, int nell2)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%feature("pythonprepend") angular_cl_vec %{
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
