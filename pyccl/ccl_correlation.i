%module ccl_correlation

%{
#define SWIG_FILE_WITH_INIT
#include "../include/ccl_correlation.h"
%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* theta, int nt)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout)};

%include "../include/ccl_correlation.h"

%inline %{
void correlation_vec(
                        ccl_cosmology * cosmo,
			CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel,
			double* theta, int nt,
                        double* output, int nout)
{
    for(int i=0; i < nout; i++){
      output[i] = ccl_single_tracer_corr(theta[i],cosmo,ct1,ct2,i_bessel);
    }
}
 %}
