%module ccl_cls

%{
/* put additional #include here */
%}

%include "../include/ccl_cls.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* ell, int nell)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%feature("pythonprepend") angular_cl_vec_limber %{
    if numpy.shape(ell) != (nout,):
        raise CCLError("Input shape for `ell` must match `(nout,)`!")
%}

%inline %{

void angular_cl_vec_limber(ccl_cosmology * cosmo,
                    ccl_cl_tracer_collection_t *clt1,
                    ccl_cl_tracer_collection_t *clt2,
                    ccl_f2d_t *pspec, 
                    double* ell, int nell,
                    int integration_type,
                    int nout, double* output,
                    int *status) {
  // Since N5K integration, this piece does Limber integration only
  // Compute Limber across the full ell range

    ccl_angular_cls_limber(cosmo, clt1, clt2, pspec, nell, ell, output,
                           integration_type, status);

}

%}
