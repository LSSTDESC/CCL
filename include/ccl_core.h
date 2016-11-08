#pragma once
#include "gsl/gsl_spline.h"
#include "ccl_config.h"
#include "ccl_constants.h"
#include <stdbool.h>

typedef struct ccl_parameters {
    // Densities: CDM, baryons, total matter, neutrinos, curvature
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

    //Modified growth rate
  bool has_mgrowth;
  int nz_mgrowth;
  double *z_mgrowth;
  double *df_mgrowth;

} ccl_parameters;



typedef struct ccl_data{
    // These are all functions of the scale factor a.
    // Distances are defined in EITHER Mpc or Mpc/h (TBC)
  double growth0;
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

} ccl_data;

typedef struct ccl_cosmology
{
  ccl_parameters    params;
  ccl_configuration config;
  ccl_data          data;
  
  bool computed_distances;
  bool computed_growth;
  bool computed_power;
  bool computed_sigma;

  int status;
  //this is optional - less tedious than tracking all numerical values for status in error handler function
  char status_message[500];

  // other flags?
} ccl_cosmology;


// Initialization and life cycle of objects
ccl_cosmology * ccl_cosmology_create(ccl_parameters params, ccl_configuration config);

// User-facing creation routines
// Most general case
ccl_parameters ccl_parameters_create(double Omega_c, double Omega_b, double Omega_k, double Omega_n, double w0, double wa, double h, double A_s, double n_s,int nz_mgrowth,double *zarr_mgrowth,double *dfarr_mgrowth);
// Specific sub-models
ccl_parameters ccl_parameters_create_flat_lcdm(double Omega_c, double Omega_b, double h, double A_s, double n_s);
ccl_parameters ccl_parameters_create_flat_wcdm(double Omega_c, double Omega_b, double w0, double h, double A_s, double n_s);
ccl_parameters ccl_parameters_create_flat_wacdm(double Omega_c, double Omega_b, double w0,double wa, double h, double A_s, double n_s);
ccl_parameters ccl_parameters_create_lcdm(double Omega_c, double Omega_b, double Omega_k, double h, double A_s, double n_s);


void ccl_cosmology_free(ccl_cosmology * cosmo);

void ccl_cosmology_compute_distances(ccl_cosmology * cosmo);
void ccl_cosmology_compute_growth(ccl_cosmology * cosmo);
void ccl_cosmology_compute_power(ccl_cosmology * cosmo);
// Internal(?)

// Distance-like function examples

