%module ccl_background

%{
#define SWIG_FILE_WITH_INIT
#include "../include/ccl_background.h"
%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* a, int na)};
%apply (double* IN_ARRAY1, int DIM1) {(double* chi, int nchi)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout)};
// Flag status variable as input/output variable
/*%apply (int* INOUT) {(int * status)};*/

%include "../include/ccl_background.h"

%inline %{
void growth_factor_vec(ccl_cosmology * cosmo, 
                        double* a, int na,
                        double* output, int nout,
                        int* status)
{
    assert(nout == na);
    ccl_growth_factors(cosmo, na, a, output, status);
}

void growth_factor_unnorm_vec(ccl_cosmology * cosmo, 
                        double* a, int na,
                        double* output, int nout,
                        int* status){
    assert(nout == na);
    ccl_growth_factors_unnorm(cosmo, na, a, output, status);
}

void growth_rate_vec(ccl_cosmology * cosmo, 
                        double* a, int na,
                        double* output, int nout,
                        int* status) {
    assert(nout == na);
    ccl_growth_rates(cosmo, na, a, output, status);
}

void comoving_radial_distance_vec(ccl_cosmology * cosmo, 
                        double* a, int na,
                        double* output, int nout,
                        int* status) {
    assert(nout == na);
    ccl_comoving_radial_distances(cosmo, na, a, output, status);
}

void comoving_angular_distance_vec(ccl_cosmology * cosmo, 
                       double* a, int na,
                       double* output, int nout,
                       int* status) {
    assert(nout == na);
    ccl_comoving_angular_distances(cosmo, na, a, output, status);
}

void h_over_h0_vec(ccl_cosmology * cosmo, 
                       double* a, int na,
                       double* output, int nout,
                       int* status) {
    assert(nout == na);
    ccl_h_over_h0s(cosmo, na, a, output, status);
}

void luminosity_distance_vec(ccl_cosmology * cosmo, 
                       double* a, int na,
                       double* output, int nout,
                       int* status) {
    assert(nout == na);
    ccl_luminosity_distances(cosmo, na, a, output, status);
}

void scale_factor_of_chi_vec(ccl_cosmology * cosmo, 
                       double* chi, int nchi,
                       double* output, int nout,
                       int* status) {
    assert(nout == nchi);
    ccl_scale_factor_of_chis(cosmo, nchi, chi, output, status);
}

void omega_x_vec(ccl_cosmology * cosmo, int label, 
		       double* a,  int na,
		   double* output, int nout, int *status) {
    assert(nout == na);
    for(int i=0; i < na; i++){
      output[i] = ccl_omega_x(cosmo, a[i], label, status);
    }
}


%}
