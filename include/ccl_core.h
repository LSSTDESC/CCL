/** @file */
#ifndef __CCL_CORE_H_INCLUDED__
#define __CCL_CORE_H_INCLUDED__

#include <stdbool.h>
#include <stdio.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

CCL_BEGIN_DECLS

/**
 * Struct that contains all the parameters needed to create certain splines.
 * This includes splines for the scale factor, masses, and power spectra.
 */
typedef struct ccl_spline_params {
  // scale factor splines
  int A_SPLINE_NA;
  double A_SPLINE_MIN;
  double A_SPLINE_MINLOG_PK;
  double A_SPLINE_MIN_PK;
  double A_SPLINE_MAX;
  double A_SPLINE_MINLOG;
  int A_SPLINE_NLOG;

  //Mass splines
  double LOGM_SPLINE_DELTA;
  int LOGM_SPLINE_NM;
  double LOGM_SPLINE_MIN;
  double LOGM_SPLINE_MAX;

  //PS a and k spline
  int A_SPLINE_NA_PK;
  int A_SPLINE_NLOG_PK;

  //k-splines and integrals
  double K_MAX_SPLINE;
  double K_MAX;
  double K_MIN;
  int N_K;
  int N_K_3DCOR;

  //Correlation function parameters
  double ELL_MIN_CORR;
  double ELL_MAX_CORR;
  int N_ELL_CORR;

} ccl_spline_params;


/**
 * Struct that contains parameters that control the accuracy of various GSL
 * routines.
 */
typedef struct ccl_gsl_params {
  // General parameters. If not otherwise specified, those will be copied to the
  // more specialised cases.
  double EPSREL;
  size_t N_ITERATION;

  // Integration
  int INTEGRATION_GAUSS_KRONROD_POINTS;
  double INTEGRATION_EPSREL;
  // Limber integration
  int INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS;
  double INTEGRATION_LIMBER_EPSREL;
  // Distance integrals
  double INTEGRATION_DISTANCE_EPSREL;
  // dndz integrals
  double INTEGRATION_DNDZ_EPSREL;
  // sigma_R integral
  double INTEGRATION_SIGMAR_EPSREL;

  // Root finding
  double ROOT_EPSREL;
  int ROOT_N_ITERATION;

  // ODE
  double ODE_GROWTH_EPSREL;

  // growth
  double EPS_SCALEFAC_GROWTH;
} ccl_gsl_params;


/**
 * Struct containing the parameters defining a cosmology
 */
typedef struct ccl_parameters {

  // Densities: CDM, baryons, total matter, neutrinos, curvature
  double Omega_c; /**< Density of CDM relative to the critical density*/
  double Omega_b; /**< Density of baryons relative to the critical density*/
  double Omega_m; /**< Density of all matter relative to the critical density*/
  double Omega_k; /**< Density of curvature relative to the critical density*/
  double sqrtk; /**< Square root of the magnitude of curvature, k */ //TODO check
  int k_sign; /**<Sign of the curvature k */


  // Dark Energy
  double w0;
  double wa;

  // Hubble parameters
  double H0;
  double h;

  // Neutrino properties

  double Neff; // Effective number of relativistic neutrino species in the early universe.
  int N_nu_mass; // Number of species of neutrinos which are nonrelativistic today
  double N_nu_rel;  // Number of species of neutrinos which are relativistic  today
  double *mnu;  // total mass of massive neutrinos (This is a pointer so that it can hold multiple masses.)
  double sum_nu_masses; // sum of the neutrino masses.
  double Omega_n_mass; // Omega_nu for MASSIVE neutrinos
  double Omega_n_rel; // Omega_nu for MASSLESS neutrinos

  //double Neff_partial[CCL_MAX_NU_SPECIES];
  //double mnu[CCL_MAX_NU_SPECIES];

  // Primordial power spectra
  double A_s;
  double n_s;

  // Radiation parameters
  double Omega_g;
  double T_CMB;

  // BCM baryonic model parameters
  double bcm_log10Mc;
  double bcm_etab;
  double bcm_ks;

  // Derived parameters
  double sigma8;
  double Omega_l;
  double z_star;

  //Modified growth rate
  bool has_mgrowth;
  int nz_mgrowth;
  double *z_mgrowth;
  double *df_mgrowth;
} ccl_parameters;


/**
 * Struct containing references to gsl splines for distance and acceleration calculations
 */
typedef struct ccl_data{
  // These are all functions of the scale factor a.

  // Distances are defined in Mpc
  double growth0;
  gsl_spline * chi;
  gsl_spline * growth;
  gsl_spline * fgrowth;
  gsl_spline * E;
  gsl_spline * achi;

  // All these splines use the same accelerator so that
  // if one calls them successively with the same a value
  // they will be much faster.
  gsl_interp_accel *accelerator;
  gsl_interp_accel *accelerator_achi;
  gsl_interp_accel *accelerator_m;
  gsl_interp_accel *accelerator_d;
  //TODO: it seems like we're not really using this accelerator, and we should
  gsl_interp_accel *accelerator_k;

  // Function of Halo mass M

  gsl_spline * logsigma;
  gsl_spline * dlnsigma_dlogm;

  // splines for halo mass function
  gsl_spline * alphahmf;
  gsl_spline * betahmf;
  gsl_spline * gammahmf;
  gsl_spline * phihmf;
  gsl_spline * etahmf;

  // These are all functions of the wavenumber k and the scale factor a.
  gsl_spline2d * p_lin;
  gsl_spline2d * p_nl;
  double k_min_lin; //k_min  [1/Mpc] <- minimum wavenumber that the power spectrum has been computed to
  double k_min_nl;
  double k_max_lin;
  double k_max_nl;
} ccl_data;

/**
 * Sturct containing references to instances of the above structs, and boolean flags of precomputed values.
 */
typedef struct ccl_cosmology
{
  ccl_parameters    params;
  ccl_configuration config;
  ccl_data          data;
  ccl_spline_params spline_params;
  ccl_gsl_params    gsl_params;

  bool computed_distances;
  bool computed_growth;
  bool computed_power;
  bool computed_sigma;
  bool computed_hmfparams;

  int status;
  //this is optional - less tedious than tracking all numerical values for status in error handler function
  char status_message[500];

  // other flags?
} ccl_cosmology;

// Label for whether you are passing a pointer to a sum of neutrino masses or a pointer to a list of 3 masses.
typedef enum ccl_mnu_convention {
  ccl_mnu_list = 0,   // you pass a list of three neutrino masses
  ccl_mnu_sum = 1,  // sum, defaults to splitting with normal hierarchy
  ccl_mnu_sum_inverted = 2, //sum, split with inverted hierarchy
  ccl_mnu_sum_equal = 3, //sum, split into equal masses
  // More options could be added here
} ccl_mnu_convention;

// Initialization and life cycle of objects
ccl_cosmology * ccl_cosmology_create(ccl_parameters params, ccl_configuration config);

/* Internal function to set the status message safely. */
void ccl_cosmology_set_status_message(ccl_cosmology * cosmo, const char * status_message, ...);


// User-facing creation routines
/**
 * Create a cosmology
 * @param Omega_c Omega_c
 * @param Omega_b Omega_b
 * @param Omega_k Omega_k
 * @param Neff Number of relativistic neutrino species in the early universe
 * @param mnu neutrino mass, either sum or list of length 3
 * @param mnu_type determines neutrino mass convention (ccl_mnu_list, ccl_mnu_sum, ccl_mnu_sum_inverted, ccl_mnu_sum_equal)
 * @param w0 Dark energy EoS parameter
 * @param wa Dark energy EoS parameter
 * @param h Hubble constant in units of 100 km/s/Mpc
 * @param norm_pk the normalization of the power spectrum, either A_s or sigma8
 * @param n_s the power-law index of the power spectrum
 * @param bcm_log10Mc log10 cluster mass, one of the parameters of the BCM model
 * @param bcm_etab ejection radius parameter, one of the parameters of the BCM model
 * @param bcm_ks wavenumber for the stellar profile, one of the parameters of the BCM model
 * @param nz_mgrowth the number of redshifts where the modified growth is provided
 * @param zarr_mgrowth the array of redshifts where the modified growth is provided
 * @param dfarr_mgrowth the modified growth function vector provided
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
ccl_parameters ccl_parameters_create(double Omega_c, double Omega_b, double Omega_k,
                     double Neff, double* mnu, ccl_mnu_convention mnu_type,
                     double w0, double wa, double h, double norm_pk,
                     double n_s, double bcm_log10Mc, double bcm_etab, double bcm_ks,
                     int nz_mgrowth,double *zarr_mgrowth,
                     double *dfarr_mgrowth, int *status);


/* ------- ROUTINE: ccl_parameters_create_flat_lcdm --------
INPUT: some cosmological parameters needed to create a flat LCDM model
TASK: call ccl_parameters_create to produce an LCDM model
*/
ccl_parameters ccl_parameters_create_flat_lcdm(double Omega_c, double Omega_b, double h,
					       double norm_pk, double n_s, int *status);


/**
 * Free a parameters struct
 * @param params ccl_parameters struct
 * @return void
 */
void ccl_parameters_free(ccl_parameters * params);


/**
 * Write a cosmology parameters object to a file in yaml format, .
 * @param params Cosmological parameters
 * @param filename Name of file to create and write
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * @return void
 */
void ccl_parameters_write_yaml(ccl_parameters * params, const char * filename, int * status);

/**
 * Read a cosmology parameters object from a file in yaml format, .
 * @param filename Name of existing file to read from
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * @return cosmo Cosmological parameters
 */
ccl_parameters ccl_parameters_read_yaml(const char * filename, int *status);

/**
 * Free a cosmology struct
 * @param cosmo Cosmological parameters
 * @return void
 */
void ccl_cosmology_free(ccl_cosmology * cosmo);


CCL_END_DECLS

#endif
