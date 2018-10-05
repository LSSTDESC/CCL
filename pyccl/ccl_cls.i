%module ccl_cls

%{
/* put additional #include here */
%}

%include "../include/ccl_cls.h"

// Enable vectorised arguments for arrays
%apply (int DIM1, double* IN_ARRAY1) {
                                      (int nz_n, double* z_n),
                                      (int nz_b, double* z_b),
                                      (int nz_s, double* z_s),
                                      (int nz_ba, double* z_ba),
                                      (int nz_rf, double* z_rf),
                                      (int nn, double* n),
                                      (int nb, double* b),
                                      (int ns, double* s),
                                      (int nba, double* ba),
                                      (int nrf, double* rf) }
%apply (double* IN_ARRAY1, int DIM1) {(double* ell, int nell),
                                      (double *aarr,int na)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout)};


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

CCL_ClTracer* cl_tracer_new_wrapper(
        ccl_cosmology *cosmo, int tracer_type,
        int has_rsd, int has_magnification, int has_intrinsic_alignment,
        int nz_n, double *z_n, int nn, double *n,
        int nz_b, double *z_b, int nb, double *b,
        int nz_s, double *z_s, int ns, double *s,
        int nz_ba, double *z_ba, int nba, double *ba,
        int nz_rf, double *z_rf, int nrf, double *rf,
        double z_source,
        int* status) {
    return ccl_cl_tracer(
        cosmo,
        tracer_type,
        has_rsd, has_magnification,
        has_intrinsic_alignment,
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
    if numpy.shape(ell) != (output,):
        raise CCLError("Input shape for `ell` must match `(output,)`!")
%}

%inline %{

void angular_cl_vec(ccl_cosmology * cosmo,
                    CCL_ClTracer *clt1, CCL_ClTracer *clt2,
                    double l_limber,double l_logstep,double l_linstep,
                    double dchi, double dlk, double zmin,
                    int method,
                    double* ell, int nell,
                    double* output, int nout,
                    int* status)
{
  //Cast ells as integers
  int *ell_int = malloc(nell * sizeof(int));
  CCL_ClWorkspace *w = ccl_cl_workspace_default(
        (int)(ell[nell - 1]) + 1,
        (int)l_limber,
        method,
        l_logstep,
        (int)l_linstep,
        dchi,
        dlk,
        zmin,
        status);

  for(int i=0; i < nell; i++)
    ell_int[i] = (int)(ell[i]);

  //Compute C_ells
  ccl_angular_cls(cosmo, w, clt1, clt2, nell, ell_int, output, status);

  free(ell_int);
  ccl_cl_workspace_free(w);
}

%}

%feature("pythonprepend") clt_fa_vec %{
    if numpy.shape(aarr) != (output,):
        raise CCLError("Input shape for `aarr` must match `(output,)`!")
%}

%inline %{

void clt_fa_vec(
        ccl_cosmology *cosmo, CCL_ClTracer *clt, int func_code,
        double *aarr, int na,
        double *output, int nout,
        int *status) {
    assert(nout==na);
    ccl_get_tracer_fas(cosmo, clt, na, aarr, output, func_code, status);
}

%}
