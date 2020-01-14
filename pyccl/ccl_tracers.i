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
    (double* ell_s, int nell),
    (double* chi_s, int nchi),
    (double* wchi_s, int nwchi),
    (double* lk_s, int nlk),
    (double* a_s, int na),
    (double* tka_s, int ntka),
    (double* tk_s, int ntk),
    (double* ta_s, int nta)}
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%inline %{
int get_nchi_lensing_kernel_wrapper(double *z_n, int nz_n)
{
  int status=0;
  return ccl_get_nchi_lensing_kernel(nz_n,z_n,&status);
}
%}

%inline %{
void get_chis_lensing_kernel_wrapper(ccl_cosmology *cosmo,
				     double z_max,
				     int nout,double *output,
				     int *status)
{
  ccl_get_chis_lensing_kernel(cosmo,nout,z_max,output,status);
}
%}

%feature("pythonprepend") get_lensing_kernel_wrapper %{
    if numpy.shape(z_n) != numpy.shape(n):
        raise CCLError("Input shape for `z_n` must match `n`!")

    if numpy.shape(z_b) != numpy.shape(b):
        raise CCLError("Input shape for `z_b` must match `b`!")
%}

%inline %{
void get_lensing_kernel_wrapper(ccl_cosmology *cosmo,
				double *z_n, int nz_n,
				double *n, int nn,
				double z_max,
				int has_magbias,
				double *z_b, int nz_b,
				double *b, int nb,
				double *chi_s, int nchi,
				int nout,double *output,
				int *status)
{
  int nz_s=-1;
  double *zs_arr=NULL;
  double *sz_arr=NULL;

  if(has_magbias) {
    nz_s=nz_b;
    zs_arr=z_b;
    sz_arr=b;
  }
  ccl_get_lensing_mag_kernel(cosmo,
			     nz_n, z_n, n, 1, z_max,
			     nz_s,zs_arr,sz_arr,
			     nchi,chi_s,output,status);
}
%}

%inline %{
void get_kappa_kernel_wrapper(ccl_cosmology *cosmo,double chi_source,
			      double* chi_s, int nchi,
			      int nout,double *output,
			      int *status)
{
  ccl_get_kappa_kernel(cosmo,chi_source,nchi,chi_s,output,status);
}
%}

%feature("pythonprepend") get_number_counts_kernel_wrapper %{
    if numpy.shape(z_n) != numpy.shape(n):
        raise CCLError("Input shape for `z_n` must match `n`!")
%}

%inline %{
void get_number_counts_kernel_wrapper(ccl_cosmology *cosmo,
				      double *z_n, int nz_n,
				      double *n, int nn,
				      int nout,double *output,
				      int *status)
{
  ccl_get_number_counts_kernel(cosmo,nz_n,z_n,n,1,output,status);
}
%}

%feature("pythonprepend") cl_tracer_get_kernel %{
    if chi_s.size != nout:
        raise CCLError("Input shape for `chi_s` must match `nout`")
%}

%inline %{
void cl_tracer_get_kernel(ccl_cl_tracer_t *tr,
			  double *chi_s, int nchi,
			  int nout, double *output,
			  int *status)
{
  int ii;
  for(ii=0; ii<nchi; ii++) {
    output[ii] = ccl_cl_tracer_t_get_kernel(tr, chi_s[ii],
					    status);
  }
}
%}

%feature("pythonprepend") cl_tracer_get_f_ell %{
    if ell_s.size != nout:
        raise CCLError("Input shape for `ell_s` must match `nout`")
%}

%inline %{
void cl_tracer_get_f_ell(ccl_cl_tracer_t *tr,
			 double *ell_s, int nell,
			 int nout, double *output,
			 int *status)
{
  int ii;
  for(ii=0; ii<nell; ii++) {
    output[ii] = ccl_cl_tracer_t_get_f_ell(tr, ell_s[ii],
					   status);
  }
}
%}

%feature("pythonprepend") cl_tracer_get_transfer %{
    if a_s.size * lk_s.size != nout:
        raise CCLError("`nout` must match the shapes of `k_s` times `a_s`")
%}

%inline %{
void cl_tracer_get_transfer(ccl_cl_tracer_t *tr,
			    double *lk_s, int nlk,
			    double *a_s, int na,
			    int nout, double *output,
			    int *status)
{
  int ik;
  for(ik=0; ik<nlk; ik++) {
    int ia;
    double lk = lk_s[ik];
    for(ia=0; ia<na; ia++) {
      double a = a_s[ia];
      int ii = ia + na * ik;
      output[ii] = ccl_cl_tracer_t_get_transfer(tr, lk, a,
						status);
    }
  }
}
%}

%feature("pythonprepend") cl_tracer_t_new_wrapper %{
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
    lk_ka=lk_s;
    a_ka=a_s;
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
