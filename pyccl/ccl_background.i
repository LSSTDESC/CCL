%module ccl_background

%{
/* put additional #includes here */
%}

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* a, int na)};
%apply (double* IN_ARRAY1, int DIM1) {(double* chi, int nchi)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout)};

%include "../include/ccl_background.h"

/* The python code here will be executed before all of the functions that
   follow this directive. */
%feature("pythonprepend") %{
    if numpy.shape(a) != (output,):
        raise CCLError("Input shape for `a` must match `(output,)`!")
%}

%inline %{
void growth_factor_vec(ccl_cosmology * cosmo,
                       double* a, int na,
                       double* output, int nout,
                       int* status) {
    ccl_growth_factors(cosmo, na, a, output, status);
}

void growth_factor_unnorm_vec(ccl_cosmology * cosmo,
                        double* a, int na,
                        double* output, int nout,
                        int* status){
    ccl_growth_factors_unnorm(cosmo, na, a, output, status);
}

void growth_rate_vec(ccl_cosmology * cosmo,
                        double* a, int na,
                        double* output, int nout,
                        int* status) {
    ccl_growth_rates(cosmo, na, a, output, status);
}

void comoving_radial_distance_vec(ccl_cosmology * cosmo,
                        double* a, int na,
                        double* output, int nout,
                        int* status) {
    ccl_comoving_radial_distances(cosmo, na, a, output, status);
}

void comoving_angular_distance_vec(ccl_cosmology * cosmo,
                       double* a, int na,
                       double* output, int nout,
                       int* status) {
    ccl_comoving_angular_distances(cosmo, na, a, output, status);
}

void h_over_h0_vec(ccl_cosmology * cosmo,
                       double* a, int na,
                       double* output, int nout,
                       int* status) {
    ccl_h_over_h0s(cosmo, na, a, output, status);
}

void luminosity_distance_vec(ccl_cosmology * cosmo,
                       double* a, int na,
                       double* output, int nout,
                       int* status) {
    ccl_luminosity_distances(cosmo, na, a, output, status);
}

void distance_modulus_vec(ccl_cosmology * cosmo,
                             double* a, int na,
                             double* output, int nout,
                             int* status) {
    ccl_distance_moduli(cosmo, na, a, output, status);
}


void omega_x_vec(ccl_cosmology * cosmo, int label,
		       double* a,  int na,
		   double* output, int nout, int *status) {
    for(int i=0; i < na; i++){
      output[i] = ccl_omega_x(cosmo, a[i], label, status);
    }
}

void rho_x_vec(ccl_cosmology * cosmo, int label, int is_comoving,
		double* a, int na,
		double* output, int nout, int *status) {
    for(int i=0; i < na; i++){
      output[i] = ccl_rho_x(cosmo, a[i], label, is_comoving, status);
    }
}

%}

/* Now we change the directive for `chi` instead of `a`. */
%feature("pythonprepend") %{
    if numpy.shape(chi) != (output,):
        raise CCLError("Input shape for `chi` must match `(output,)`!")
%}

%inline %{

void scale_factor_of_chi_vec(ccl_cosmology * cosmo,
                       double* chi, int nchi,
                       double* output, int nout,
                       int* status) {
    ccl_scale_factor_of_chis(cosmo, nchi, chi, output, status);
}

%}

/* The directive gets carried between files, so we reset it at the end. */
%feature("pythonprepend") %{ %}
