#pragma once
#include "gsl/gsl_spline.h"
#include "ccl_config.h"
#include "ccl_neutrinos.h"
#include "ccl_constants.h"
#include <stdbool.h>

typedef struct ccl_parameters {

  // Densities
    double Omega_c;
    double Omega_b;
    double Omega_m;
    double Omega_n;
    double Omega_k;

    // Hubble parameters
    double H0;
    double h;

    // Neutrino properties
    // Number of different species of neutrinos (i.e. 1 for one massive neutrinos)
    int N_nu_species;
    double Neff_partial[CCL_MAX_NU_SPECIES];
    double mnu[CCL_MAX_NU_SPECIES];
    
    // Radiation parameters
    double Omega_g;
    double T_CMB;

    // Dark Energy
    double w0;
    double wa;

    // Primordial power spectra
    double A_s;
    double n_s;

    // Derived parameters
    double sigma_8;
    double Omega_l;
    double z_star;

} ccl_parameters;



typedef struct ccl_data{
  // These are all functions of the scale factor a.
  // Distances are defined in EITHER Mpc or Mpc/h (TBC)
  gsl_spline * chi;
  gsl_spline * growth;
  gsl_spline * fgrowth;
  gsl_spline * E;

  // All these splines use the same accelerator so that
  // if one calls them successively with the same a value
  // they will be much faster.
  gsl_interp_accel *accelerator;
  //TODO: why not use interpolation accelerators?

  // Function of Halo mass M
  gsl_spline * sigma;

  // These are all functions of the wavenumber k and the scale factor a.
  gsl_spline * p_lin;
  gsl_spline * p_nl;

  // neutrino phase-space integral splined in mnu/T units
  gsl_spline * nu_pspace_int;

} ccl_data;

typedef struct ccl_cosmology
{
    ccl_parameters    params;
    ccl_configuration config;
    ccl_data          data;

    bool computed_distances;
    bool computed_power;
    bool computed_sigma;

    // other flags?
} ccl_cosmology;


// Initialization and life cycle of objects
ccl_cosmology * ccl_cosmology_create(ccl_parameters params, ccl_configuration config);

// User-facing creation routines
// Most general case
ccl_parameters ccl_parameters_create(double Omega_c, double Omega_b, double Omega_k, double Omega_n, double w0, double wa, double h, double A_s, double n_s);
// Specific sub-models
ccl_parameters ccl_parameters_create_flat_lcdm(double Omega_c, double Omega_b, double h, double A_s, double n_s);
ccl_parameters ccl_parameters_create_flat_wcdm(double Omega_c, double Omega_b, double w0, double h, double A_s, double n_s);
ccl_parameters ccl_parameters_create_flat_wacdm(double Omega_c, double Omega_b, double w0,double wa, double h, double A_s, double n_s);
ccl_parameters ccl_parameters_create_lcdm(double Omega_c, double Omega_b, double Omega_k, double h, double A_s, double n_s);


void ccl_cosmology_free(ccl_cosmology * cosmo);

void ccl_cosmology_compute_distances(ccl_cosmology * cosmo, int *status);
void ccl_cosmology_compute_power(ccl_cosmology * cosmo, int *status);
// Internal(?)

// Distance-like function examples

