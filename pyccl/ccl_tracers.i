%module ccl_tracers

%{
/* put additional #include here */
%}

%include "../include/ccl_tracers.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {
    (double* z_n, int nz_n),
    (double* z_b, int nz_b),
    (double* n, int nn),
    (double* b, int nb)}
//%apply (double* IN_ARRAY1, int DIM1) {
//    (double* ell, int nell),
//    (double* aarr, int na)};
//%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};


%feature("pythonprepend") tracer_get_nc_dens %{
    if numpy.shape(z_n) != numpy.shape(n):
        raise CCLError("Input shape for `z_n` must match `n`!")

    if numpy.shape(z_b) != numpy.shape(b):
        raise CCLError("Input shape for `z_b` must match `b`!")
%}

%inline %{

ccl_cl_tracer_t *tracer_get_nc_dens(ccl_cosmology *cosmo,
				      double* z_n, int nz_n, double *n, int nn,
				      double* z_b, int nz_b, double *b, int nb,
				      int *status)
{
  ccl_cl_tracer_t *t=ccl_nc_dens_tracer_new(cosmo,nz_n,z_n,n,nz_b,z_b,b,1,status);
  return t;
}

%}

%feature("pythonprepend") tracer_get_nc_rsd %{
    if numpy.shape(z_n) != numpy.shape(n):
        raise CCLError("Input shape for `z_n` must match `n`!")
%}

%inline %{

ccl_cl_tracer_t *tracer_get_nc_rsd(ccl_cosmology *cosmo,
				   double* z_n, int nz_n, double *n, int nn,
				   int *status)
{
  ccl_cl_tracer_t *t=ccl_nc_rsd_tracer_new(cosmo,nz_n,z_n,n,1,status);
  return t;
}

%}

%feature("pythonprepend") tracer_get_nc_mag %{
    if numpy.shape(z_n) != numpy.shape(n):
        raise CCLError("Input shape for `z_n` must match `n`!")

    if numpy.shape(z_b) != numpy.shape(b):
        raise CCLError("Input shape for `z_b` must match `b`!")
%}

%inline %{

ccl_cl_tracer_t *tracer_get_nc_mag(ccl_cosmology *cosmo,
				   double* z_n, int nz_n, double *n, int nn,
				   double* z_b, int nz_b, double *b, int nb,
				   int *status)
{
  ccl_cl_tracer_t *t=ccl_nc_mag_tracer_new(cosmo,nz_n,z_n,n,nz_b,z_b,b,1,status);
  return t;
}

%}

%feature("pythonprepend") tracer_get_wl_ia %{
    if numpy.shape(z_n) != numpy.shape(n):
        raise CCLError("Input shape for `z_n` must match `n`!")

    if numpy.shape(z_b) != numpy.shape(b):
        raise CCLError("Input shape for `z_b` must match `b`!")
%}

%inline %{

ccl_cl_tracer_t *tracer_get_wl_ia(ccl_cosmology *cosmo,
				   double* z_n, int nz_n, double *n, int nn,
				   double* z_b, int nz_b, double *b, int nb,
				   int *status)
{
  ccl_cl_tracer_t *t=ccl_wl_ia_tracer_new(cosmo,nz_n,z_n,n,nz_b,z_b,b,1,status);
  return t;
}

%}

%feature("pythonprepend") tracer_get_wl_shear %{
    if numpy.shape(z_n) != numpy.shape(n):
        raise CCLError("Input shape for `z_n` must match `n`!")
%}

%inline %{

ccl_cl_tracer_t *tracer_get_wl_shear(ccl_cosmology *cosmo,
				     double* z_n, int nz_n, double *n, int nn,
				     int *status)
{
  ccl_cl_tracer_t *t=ccl_wl_shear_tracer_new(cosmo,nz_n,z_n,n,1,status);
  return t;
}

%}

%feature("pythonprepend") tracer_get_kappa %{
    if z_source<0:
        raise CCLError("Source redshift cannot be negative")
%}

%inline %{

ccl_cl_tracer_t *tracer_get_kappa(ccl_cosmology *cosmo,double z_source,
				  int *status)
{
  ccl_cl_tracer_t *t=ccl_kappa_tracer_new(cosmo,z_source,100,status);
  return t;
}

%}
