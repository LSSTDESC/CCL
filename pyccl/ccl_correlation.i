%module ccl_correlation

%{
/* put additional #include here */
%}

%include "../include/ccl_correlation.h"

// Enable vectorised arguments for arrays
%apply (int DIM1,double* IN_ARRAY1) {
                                     (int nlarr,double* larr),
                                     (int nclarr,double* clarr),
                                     (int nt,double *theta),
                                     (int nr,double *r),
                                     (int ns,double *s),
                                     (int nsig,double *sig)}
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout),(double* xi, int nxi),(double* xis, int nxis)};

%inline %{

void correlation_vec(ccl_cosmology *cosmo,
		     int nlarr,double *larr,
		     int nclarr,double *clarr,
		     int nt,double *theta,
		     int corr_type,int method,
		     double *output,int nout,
		     int *status)
{
  assert(nlarr==nclarr);
  assert(nt==nout);

  ccl_correlation(cosmo,nlarr,larr,clarr,nt,theta,output,corr_type,0,NULL,method,status);
}

void correlation_3d_vec(ccl_cosmology *cosmo,double a,
		     int nr,double *r,
                     double *xi,int nxi,
		     int *status)
{ 
  assert(nr==nxi);

  ccl_correlation_3d(cosmo,a,nr,r,xi,0,NULL,status);
}

void correlation_multipole_vec(ccl_cosmology *cosmo,double a,double beta,
			       int l,int ns,double *s,
                               double *xis,int nxis,
         		       int *status)
{ 
  assert(ns==nxis);

  ccl_correlation_multipole(cosmo,a,beta,l,ns,s,xis,status);
}

void correlation_3dRsd_vec(ccl_cosmology *cosmo,double a,double mu,double beta,
			       int ns,double *s,
                               double *xis,int nxis,int use_spline,
         		       int *status)
{ 
  assert(ns==nxis);

  ccl_correlation_3dRsd(cosmo,a,ns,s,mu,beta,xis,use_spline,status);
}

void correlation_pi_sigma_vec(ccl_cosmology *cosmo,double a,double beta,
			   double pie,int nsig,double *sig,double* xis, int nxis,int use_spline,
			   int *status)
{
    assert(nsig==nxis);

    ccl_correlation_pi_sigma(cosmo,a,beta,pie,nsig,sig,xis,use_spline,status);
}

void correlation_multipole_spline_free_vec()
{
    ccl_correlation_multipole_spline_free();
}
%}
