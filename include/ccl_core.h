#pragma once
#include "gsl/gsl_spline.h"
#include "ccl_config.h"
#include "ccl_constants.h"
#include <stdbool.h>



typedef struct ccl_parameters {
    // Densities
	double Omega_c;
    double Omega_b;
    double Omega_m;
    double Omega_n;
    double Omega_k;

    // Dark Energy
    double w0;
    double wa;

    // Hubble parameters
    double H0;
    double h;

    // Primordial power spectra
    double A_s;
    double n_s;

    // Radiation parameters
    double Omega_g;
    double T_CMB;

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
    gsl_spline * E;

    // Function of Halo mass M
    gsl_spline * sigma;

    // These are all functions of the wavenumber k and the scale factor a.
    gsl_spline * p_lin;
    gsl_spline * p_nl;

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
ccl_parameters ccl_parameters_flat_lcdm(double Omega_c, double Omega_b, double h, double A_s, double n_s);
void ccl_cosmology_free(ccl_cosmology * cosmo);

void ccl_cosmology_compute_distances(ccl_cosmology * cosmo, int *status);
void ccl_cosmology_compute_power(ccl_cosmology * cosmo, int *status);
// Internal(?)

// Distance-like function examples

