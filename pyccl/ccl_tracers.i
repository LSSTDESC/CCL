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
    (double* b, int nb),
    (double* chi_s, int nchi),
    (double* wchi_s, int nwchi),
    (double* lk_s, int nlk),
    (double* a_s, int na),
    (double* tka_s, int ntka),
    (double* tk_s, int ntk),
    (double* ta_s, int nta)}
//%apply (double* IN_ARRAY1, int DIM1) {
//    (double* ell, int nell),
//    (double* aarr, int na)};
//%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%feature("pythonprepend") cl_tracer_t_new_wrapped %{
    if numpy.shape(chi_s) != numpy.shape(wchi_s):
        raise CCLError("Input shape for `chi_s` must match `wchi_s`!")
%}

%inline %{
ccl_cl_tracer_t *cl_tracer_t_new_wrapper(ccl_cosmology *cosmo,
					 int der_bessel,int der_angles,
					 double *chi_s,int nchi,
					 double *wchi_s,int nwchi,
					 double *a_s,int na,
					 double *lk_s,int nlk,
					 double *tka_s,int ntka,
					 double *tk_s,int ntk,
					 double *ta_s,int nta,
					 int is_logt,int is_factorizable,
					 int is_k_constant,int is_a_constant,
					 int is_kernel_constant,
					 int extrap_order_lok,
					 int extrap_order_hik,
					 int *status)
{
  double *chi_w,*w_w;
  double *a_ka,*lk_ka;
  double *fka_arr,*fk_arr,*fa_arr;

  // Set appropriate arrays to NULL
  if(is_kernel_constant) {
    chi_w=NULL;
    w_w=NULL;
  }
  else {
    chi_w=chi_s;
    w_w=wchi_s;
  }
  if(is_factorizable) {
    fka_arr=NULL;
    if(is_k_constant) { 
      lk_ka=NULL;
      fk_arr=NULL;
    }
    else {
      lk_ka=lk_s;
      fk_arr=tk_s;
    }
    if(is_a_constant) {
      a_ka=NULL;
      fa_arr=NULL;
    }
    else {
      a_ka=a_s;
      fa_arr=ta_s;
    }
  }
  else {
    fk_arr=NULL;
    fa_arr=NULL;
    fka_arr=tka_s;
  }

  //Initialize tracer
  ccl_cl_tracer_t *t=ccl_cl_tracer_t_new(cosmo,der_bessel,der_angles,
					 nchi,chi_w,w_w,
					 na,a_ka,
					 nlk,lk_ka,
					 fka_arr,fk_arr,fa_arr,
					 is_logt,is_factorizable,
					 extrap_order_lok,
					 extrap_order_hik,
					 status);

  return t;
}
%}

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
