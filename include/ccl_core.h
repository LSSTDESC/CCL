/** @file */
#ifndef __CCL_CORE_H_INCLUDED__
#define __CCL_CORE_H_INCLUDED__

#include <stdbool.h>
#include <stdio.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_const_mksa.h>

#include "ccl_utils.h"
#include "ccl_f1d.h"
#include "ccl_f2d.h"

CCL_BEGIN_DECLS

/**
 * Struct to hold physical constants.
 */
typedef struct ccl_physical_constants {
  /**
   * Lightspeed / H0 in units of Mpc/h (from CODATA 2014)
   */
  double CLIGHT_HMPC;

  /**
   * Newton's gravitational constant in units of m^3/Kg/s^2
   */
  double GNEWT;

  /**
   * Solar mass in units of kg (from GSL)
   */
  double SOLAR_MASS;

  /**
   * Mpc to meters (from PDG 2013)
   */
  double MPC_TO_METER;

  /**
   * pc to meters (from PDG 2013)
   */
  double PC_TO_METER;

  /**
   * Rho critical in units of M_sun/h / (Mpc/h)^3
   */
  double RHO_CRITICAL;

  /**
   * Boltzmann constant in units of J/K
  */
  double KBOLTZ;

  /**
   * Stefan-Boltzmann constant in units of kg/s^3 / K^4
   */
  double STBOLTZ;

  /**
   * Planck's constant in units kg m^2 / s
   */
  double HPLANCK;

  /**
   * The speed of light in m/s
   */
  double CLIGHT;

  /**
   * Electron volt to Joules convestion
   */
  double EV_IN_J;

  /**
   * Temperature of the CMB in K
   */
  double T_CMB;

  /**
   * T_ncdm, as taken from CLASS, explanatory.ini
   */
  double TNCDM;

  /**
   * neutrino mass splitting differences
   * See Lesgourgues and Pastor, 2012 for these values.
   * Adv. High Energy Phys. 2012 (2012) 608515,
   * arXiv:1212.6154, page 13
  */
  double DELTAM12_sq;
  double DELTAM13_sq_pos;
  double DELTAM13_sq_neg;
} ccl_physical_constants;

extern ccl_physical_constants ccl_constants;

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
  double A_SPLINE_MINLOG_SM;
  double A_SPLINE_MIN_SM;
  double A_SPLINE_MAX;
  double A_SPLINE_MINLOG;
  int A_SPLINE_NLOG;

  //Mass splines
  double LOGM_SPLINE_DELTA;
  int LOGM_SPLINE_NM;
  double LOGM_SPLINE_MIN;
  double LOGM_SPLINE_MAX;

  //PS a and k spline
  int A_SPLINE_NA_SM;
  int A_SPLINE_NLOG_SM;
  int A_SPLINE_NA_PK;
  int A_SPLINE_NLOG_PK;

  //k-splines and integrals
  double K_MAX_SPLINE;
  double K_MAX;
  double K_MIN;
  double DLOGK_INTEGRATION;
  double DCHI_INTEGRATION;
  int N_K;
  int N_K_3DCOR;

  //Correlation function parameters
  double ELL_MIN_CORR;
  double ELL_MAX_CORR;
  int N_ELL_CORR;

  // interpolation types
  const gsl_interp_type* A_SPLINE_TYPE;
  const gsl_interp_type* K_SPLINE_TYPE;
  const gsl_interp_type* M_SPLINE_TYPE;
  const gsl_interp_type* D_SPLINE_TYPE;
  const gsl_interp2d_type* PNL_SPLINE_TYPE;
  const gsl_interp2d_type* PLIN_SPLINE_TYPE;
  const gsl_interp_type* CORR_SPLINE_TYPE;
} ccl_spline_params;

extern const ccl_spline_params default_spline_params;

/**
 * Struct that contains parameters that control the accuracy of various GSL
 * routines.
 */
typedef struct ccl_gsl_params {
  // General parameters
  size_t N_ITERATION;

  // Integration
  int INTEGRATION_GAUSS_KRONROD_POINTS;
  double INTEGRATION_EPSREL;
  // Limber integration
  int INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS;
  double INTEGRATION_LIMBER_EPSREL;
  // Distance integrals
  double INTEGRATION_DISTANCE_EPSREL;
  // sigma_R integral
  double INTEGRATION_SIGMAR_EPSREL;
  // k_NL integral
  double INTEGRATION_KNL_EPSREL;

  // Root finding
  double ROOT_EPSREL;
  int ROOT_N_ITERATION;

  // ODE
  double ODE_GROWTH_EPSREL;

  // growth
  double EPS_SCALEFAC_GROWTH;

  // halo model
  double HM_MMIN;
  double HM_MMAX;
  double HM_EPSABS;
  double HM_EPSREL;
  size_t HM_LIMIT;
  int HM_INT_METHOD;

} ccl_gsl_params;

extern const ccl_gsl_params default_gsl_params;

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
  double *m_nu;  // total mass of massive neutrinos (This is a pointer so that it can hold multiple masses.)
  double sum_nu_masses; // sum of the neutrino masses.
  double Omega_nu_mass; // Omega_nu for MASSIVE neutrinos
  double Omega_nu_rel; // Omega_nu for MASSLESS neutrinos

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

  // mu / Sigma quasistatica parameterisation of modified gravity params
  double mu_0;
  double sigma_0;
  double c1_mg;
  double c2_mg;
  double lambda_mg;

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
typedef struct ccl_data {
  // These are all functions of the scale factor a.

  // Distances are defined in Mpc
  double growth0;
  gsl_spline * chi;
  gsl_spline * growth;
  gsl_spline * fgrowth;
  gsl_spline * E;
  gsl_spline * achi;

  // Function of Halo mass M
  gsl_spline2d * logsigma;

  // real-space splines for RSD
  ccl_f1d_t* rsd_splines[3];
  double rsd_splines_scalefactor;
} ccl_data;

/**
 * Sturct containing references to instances of the above structs, and boolean flags of precomputed values.
 */
typedef struct ccl_cosmology {
  ccl_parameters    params;
  ccl_configuration config;
  ccl_data          data;
  ccl_spline_params spline_params;
  ccl_gsl_params    gsl_params;

  bool computed_distances;
  bool computed_growth;
  bool computed_sigma;

  int status;
  //this is optional - less tedious than tracking all numerical values for status in error handler function
  char status_message[500];

  // other flags?
} ccl_cosmology;

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
                                     double Neff, double* mnu, int n_mnu,
                                     double w0, double wa, double h, double norm_pk,
                                     double n_s, double bcm_log10Mc, double bcm_etab, double bcm_ks,
                                     double mu_0, double sigma_0, double c1_mg, double c2_mg, double lambda_mg,
                                     int nz_mgrowth, double *zarr_mgrowth,
                                     double *dfarr_mgrowth, int *status);


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

int ccl_get_pk_spline_na(ccl_cosmology *cosmo);
int ccl_get_pk_spline_nk(ccl_cosmology *cosmo);
void ccl_get_pk_spline_a_array(ccl_cosmology *cosmo,int ndout,double* doutput,int *status);
void ccl_get_pk_spline_lk_array(ccl_cosmology *cosmo,int ndout,double* doutput,int *status);

CCL_END_DECLS

#endif
