%module ccl_halomod

%{
#define SWIG_FILE_WITH_INIT
#include "../include/ccl_params.h"
#include "../include/ccl_halomod.h"
%}

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

%include "../include/ccl_halomod.h"

 // Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* k, int nk)};
%apply (double* IN_ARRAY1, int DIM1) {(double* halo_mass, int nm)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout)};

%inline %{
  void onehalo_matter_power_vec(ccl_cosmology *cosmo,
				double a,
				double *k, int nk,
				double *output, int nout,
				int *status)
  {
    assert(nout == nk);
    for(int i=0; i < nk; i++){
      output[i] = ccl_onehalo_matter_power(cosmo, k[i], a, status);
    }
  }

  void twohalo_matter_power_vec(ccl_cosmology *cosmo,
				double a,
				double *k, int nk,
				double *output, int nout,
				int *status)
  {
    assert(nout == nk);
    for(int i=0; i < nk; i++){
      output[i] = ccl_twohalo_matter_power(cosmo, k[i], a, status);
    }
  }

  void halomodel_matter_power_vec(ccl_cosmology *cosmo,
				  double a,
				  double *k, int nk,
				  double *output, int nout,
				  int *status)
  {
    assert(nout == nk);
    for(int i=0; i < nk; i++){
      output[i] = ccl_halomodel_matter_power(cosmo, k[i], a, status);
    }
  }

  void halo_concentration_vec(ccl_cosmology *cosmo,
			      double a,
			      double odelta,
			      double *halo_mass, int nm,
			      double *output, int nout,
			      int *status)
  {
    assert(nout == nm);
    for(int i=0; i < nm; i++){
      output[i] = ccl_halo_concentration(cosmo, halo_mass[i], a, odelta, status);
    }
  }

  %}
