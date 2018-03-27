%module ccl_cls

%{
#define SWIG_FILE_WITH_INIT
#include "../include/ccl_cls.h"
%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

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

%inline %{

CCL_ClTracer* cl_tracer_new_wrapper(ccl_cosmology *cosmo,int tracer_type,
				int has_rsd, int has_magnification, int has_intrinsic_alignment,
				int nz_n, double *z_n, int nn, double *n,
				int nz_b, double *z_b, int nb, double *b,
				int nz_s, double *z_s, int ns, double *s,
				int nz_ba, double *z_ba, int nba, double *ba,
				int nz_rf, double *z_rf, int nrf, double *rf,
				double z_source,
				int* status){
    
    assert(nz_n == nn);
    assert(nz_b == nb);
    assert(nz_s == ns);
    assert(nz_ba == nba);
    assert(nz_rf == nrf);
    
    // Check for null arrays
    
    
    return ccl_cl_tracer_new(cosmo, tracer_type,
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


void angular_cl_vec(ccl_cosmology * cosmo,
                    CCL_ClTracer *clt1, CCL_ClTracer *clt2,
                    double* ell, int nell,
                    double* output, int nout,
                    int* status)
{
    assert(nout == nell);
    for(int i=0; i < nell; i++){
        output[i] = ccl_angular_cl(cosmo, ell[i], clt1, clt2, status);
    }
}

void clt_fa_vec(ccl_cosmology *cosmo,CCL_ClTracer *clt,int func_code,
		double *aarr,int na,double *output,int nout,int *status)
{
  assert(nout==na);
  ccl_get_tracer_fas(cosmo,clt,na,aarr,output,func_code,status);
}
 
%}
