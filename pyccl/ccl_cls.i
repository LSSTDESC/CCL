%module ccl_cls

%{
/* put additional #include here */
%}

%include "../include/ccl_cls.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {
    (double* z_n, int nz_n),
    (double* z_b, int nz_b),
    (double* z_s, int nz_s),
    (double* z_ba, int nz_ba),
    (double* z_rf, int nz_rf),
    (double* n, int nn),
    (double* b, int nb),
    (double* s, int ns),
    (double* ba, int nba),
    (double* rf, int nrf)}
%apply (double* IN_ARRAY1, int DIM1) {
    (double* ell, int nell),
    (double* aarr, int na)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};


%feature("pythonprepend") cl_tracer_new_wrapper %{
    if numpy.shape(z_n) != numpy.shape(n):
        raise CCLError("Input shape for `z_n` must match `n`!")

    if numpy.shape(z_b) != numpy.shape(b):
        raise CCLError("Input shape for `z_b` must match `b`!")

    if numpy.shape(z_s) != numpy.shape(s):
        raise CCLError("Input shape for `z_s` must match `s`!")

    if numpy.shape(z_ba) != numpy.shape(ba):
        raise CCLError("Input shape for `z_ba` must match `ba`!")

    if numpy.shape(z_rf) != numpy.shape(rf):
        raise CCLError("Input shape for `z_rf` must match `rf`!")
%}

%inline %{

CCL_ClTracer* cl_tracer_new_wrapper(ccl_cosmology *cosmo, int tracer_type,
                                    int has_density, int has_rsd, int has_magnification,
				    int has_shear, int has_intrinsic_alignment,
                                    double* z_n, int nz_n, double* n, int nn,
                                    double* z_b, int nz_b, double* b, int nb,
                                    double* z_s, int nz_s, double* s, int ns,
                                    double* z_ba, int nz_ba, double* ba, int nba,
                                    double* z_rf, int nz_rf, double* rf, int nrf,
                                    double z_source, int *status) {
    return ccl_cl_tracer(
        cosmo,
        tracer_type,
        has_density, has_rsd, has_magnification,
        has_shear, has_intrinsic_alignment,
        nz_n, z_n, n,
        nz_b, z_b, b,
        nz_s, z_s, s,
        nz_ba, z_ba, ba,
        nz_rf, z_rf, rf,
        z_source,
        status);
}

%}

%feature("pythonprepend") angular_cl_vec %{
    if numpy.shape(ell) != (nout,):
        raise CCLError("Input shape for `ell` must match `(nout,)`!")
%}

%inline %{

void angular_cl_vec(ccl_cosmology * cosmo, CCL_ClTracer *clt1, CCL_ClTracer *clt2,
		    ccl_f2d_t *pspec,
                    double l_limber, double l_logstep, double l_linstep,
                    double* ell, int nell, int nout, double* output, int *status) {
  //Check if we need non-Limber power spectra
  int index_nonlimber_last=-1;
  for(int i=0; i < nell; i++) {
    if(ell[i]<l_limber)
      index_nonlimber_last=i;
    else
      break;
  }

  //Compute non-Limber power spectra
  if(index_nonlimber_last>=0) {
    //Cast ells as integers
    int *ell_int = malloc((index_nonlimber_last+1) * sizeof(int));
    for(int i=0; i <= index_nonlimber_last; i++)
      ell_int[i] = (int)(ell[i]);
    // Non-Limber computation
    ccl_angular_cls_nonlimber(cosmo, l_logstep, l_linstep, clt1, clt2, pspec,
			      index_nonlimber_last+1, ell_int, output, status);
    free(ell_int);
  }

  //Compute Limber part
  for(int i=index_nonlimber_last+1;i<nell;i++)
    output[i]=ccl_angular_cl_limber(cosmo, clt1, clt2, pspec, ell[i], status);
}

%}

%feature("pythonprepend") clt_fa_vec %{
    if numpy.shape(aarr) != (nout,):
        raise CCLError("Input shape for `aarr` must match `(nout,)`!")
%}

%inline %{

void clt_fa_vec(ccl_cosmology *cosmo, CCL_ClTracer *clt, int func_code,
                double* aarr, int na, int nout, double* output, int *status) {
    ccl_get_tracer_fas(cosmo, clt, na, aarr, output, func_code, status);
}

%}
