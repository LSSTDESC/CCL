%module ccl_cls

%{
/* put additional #include here */
%}

%include "../include/ccl_cls.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* ell, int nell)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%feature("pythonprepend") angular_cl_vec %{
    if numpy.shape(ell) != (nout,):
        raise CCLError("Input shape for `ell` must match `(nout,)`!")
%}

%inline %{

void angular_cl_vec(ccl_cosmology * cosmo,
                    ccl_cl_tracer_collection_t *clt1,
                    ccl_cl_tracer_collection_t *clt2,
                    ccl_f2d_t *pspec, double l_limber,
                    double* ell, int nell,
                    int integration_type,
                    int nout, double* output,
                    int *status) {

  // Check if we need non-Limber power spectra
  int index_nonlimber_last = -1;
  for(int i=0; i < nell; i++) {
    if(ell[i] < l_limber)
      index_nonlimber_last = i;
    else
      break;
  }

  // Compute non-Limber power spectra
  if(index_nonlimber_last >= 0) {
    // Cast ells as integers
    int *ell_int = malloc((index_nonlimber_last+1) * sizeof(int));

    if (ell_int == NULL) {
      *status = CCL_ERROR_MEMORY;
    }
    else {
      for(int i=0; i <= index_nonlimber_last; i++)
        ell_int[i] = (int)(ell[i]);

      // Non-Limber computation
      ccl_angular_cls_nonlimber(cosmo, clt1, clt2, pspec,
                                index_nonlimber_last+1, ell_int, output, status);
      free(ell_int);
    }
  }

  // Compute Limber part
  double *_ell = NULL;
  double *_cl_ell = NULL;

  if (*status == 0) {
    _ell = malloc((nell - (index_nonlimber_last+1)) * sizeof(double));
    _cl_ell = malloc((nell - (index_nonlimber_last+1)) * sizeof(double));

    if ((_ell == NULL) || (_cl_ell == NULL)) {
      *status = CCL_ERROR_MEMORY;
     }
  }

  if (*status == 0) {
    for (int i=index_nonlimber_last+1; i < nell; i++)
      _ell[i - (index_nonlimber_last+1)] = ell[i];

    ccl_angular_cls_limber(cosmo, clt1, clt2, pspec, (nell - (index_nonlimber_last+1)), _ell, _cl_ell,
                           integration_type, status);

    for (int i=index_nonlimber_last+1; i < nell; i++)
      output[i] = _cl_ell[i - (index_nonlimber_last+1)];
  }

  free(_ell);
  free(_cl_ell);
}

%}
