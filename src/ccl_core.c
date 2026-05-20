#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_integration.h>

#include "ccl.h"

// Macros for replacing relative paths
#define EXPAND_STR(s) STRING(s)
#define STRING(s) #s


const ccl_configuration default_config = {ccl_boltzmann_class, ccl_halofit};


//Precision parameters
/**
 * Default relative precision if not otherwise specified
 */
#define GSL_EPSREL 1E-4

/**
 * Default number of iterations for integration and root-finding if not otherwise
 * specified
 */
#define GSL_N_ITERATION 1000

/**
 * Default number of Gauss-Kronrod points in QAG integration if not otherwise
 * specified
 */
#define GSL_INTEGRATION_GAUSS_KRONROD_POINTS GSL_INTEG_GAUSS41

/**
 * Relative precision in sigma_R calculations
 */
#define GSL_EPSREL_SIGMAR 1E-5

/**
 * Relative precision in k_NL calculations
 */
#define GSL_EPSREL_KNL 1E-5

/**
 * Relative precision in distance calculations
 */
#define GSL_EPSREL_DIST 1E-6

/**
 * Relative precision in growth calculations
 */
#define GSL_EPSREL_GROWTH 1E-6

/**
 * Relative precision in dNdz calculations
 */
#define GSL_EPSREL_DNDZ 1E-6

ccl_gsl_params ccl_user_gsl_params = {
  GSL_N_ITERATION,                     // N_ITERATION
  GSL_INTEGRATION_GAUSS_KRONROD_POINTS,// INTEGRATION_GAUSS_KRONROD_POINTS
  GSL_EPSREL,                          // INTEGRATION_EPSREL
  GSL_INTEGRATION_GAUSS_KRONROD_POINTS,// INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS
  GSL_EPSREL,                          // INTEGRATION_LIMBER_EPSREL
  GSL_EPSREL_DIST,                     // INTEGRATION_DISTANCE_EPSREL
  GSL_EPSREL_SIGMAR,                   // INTEGRATION_SIGMAR_EPSREL
  GSL_EPSREL_KNL,                      // INTEGRATION_KNL_EPSREL
  GSL_EPSREL,                          // ROOT_EPSREL
  GSL_N_ITERATION,                     // ROOT_N_ITERATION
  GSL_EPSREL_GROWTH,                   // ODE_GROWTH_EPSREL
  1E-6,                                // EPS_SCALEFAC_GROWTH
  true,                                // NZ_NORM_SPLINE_INTEGRATION
  true                                 // LENSING_KERNEL_SPLINE_INTEGRATION
  };

#undef GSL_EPSREL
#undef GSL_N_ITERATION
#undef GSL_INTEGRATION_GAUSS_KRONROD_POINTS
#undef GSL_EPSREL_SIGMAR
#undef GSL_EPSREL_KNL
#undef GSL_EPSREL_DIST
#undef GSL_EPSREL_GROWTH
#undef GSL_EPSREL_DNDZ


ccl_spline_params ccl_user_spline_params = {

  // scale factor spline params
  250,  // A_SPLINE_NA
  0.1,  // A_SPLINE_MIN
  0.01,  // A_SPLINE_MINLOG_PK
  0.1,  // A_SPLINE_MIN_PK,
  0.01,  // A_SPLINE_MINLOG_SM,
  0.1,  // A_SPLINE_MIN_SM,
  1.0,  // A_SPLINE_MAX,
  0.0001,  // A_SPLINE_MINLOG,
  250,  // A_SPLINE_NLOG,

  // mass splines
  0.025,  // LOGM_SPLINE_DELTA
  50,  // LOGM_SPLINE_NM
  6,  // LOGM_SPLINE_MIN
  17,  // LOGM_SPLINE_MAX

  // PS a and k spline
  13,  // A_SPLINE_NA_SM
  6,  // A_SPLINE_NLOG_SM
  40,  // A_SPLINE_NA_PK
  11,  // A_SPLINE_NLOG_PK

  // k-splines and integrals
  50,  // K_MAX_SPLINE
  1E3,  // K_MAX
  5E-5,  // K_MIN
  0.025,  // DLOGK_INTEGRATION
  5.,  // DCHI_INTEGRATION
  167,  // N_K
  100000,  // N_K_3DCOR

  // correlation function parameters
  0.01,  // ELL_MIN_CORR
  60000,  // ELL_MAX_CORR
  5000,  // N_ELL_CORR

  //Spline types
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL
};


ccl_physical_constants ccl_constants = {
  /**
   * Sidereal year (s/yr). [IERS2014 in J2000.0]
   */
  365.256363004 * 86400.,
  /**
   * Lightspeed / H0 in units of Mpc/h (from CODATA 2022)
   */
  2997.92458,

  /**
   * Newton's gravitational constant in units of m^3/Kg/s^2
   */
  //6.6738e-11,  /(from PDG 2013) in m^3/Kg/s^2
  //6.67428e-11, // CLASS VALUE
  6.67430e-11, // from CODATA 2022 

  /**
   * Solar mass in units of kg (from GSL)
   */
  //GSL_CONST_MKSA_SOLAR_MASS,
  //1.9885e30, //(from PDG 2015) in Kg
  //1.9884754153381438E+30, //from IAU 2015
  1.988409871E+30, // From IAU 2015 compatible with CODATA 2022 - Newton's Gravitational constant

  /**
   * Mpc to meters (from PDG 2016 and using M_PI)
   */
  3.085677581491367399198952281E+22,

  /**
   * Rho critical in units of M_sun/h / (Mpc/h)^3
   * Values compatible with CODATA 2022
   */
  ((3*100*100)/(8*M_PI*6.67430e-11)) * (1000*1000*3.085677581491367399198952281E+22/1.988409871E+30),

  /**
   * Boltzmann constant in units of J/K
  */
  //GSL_CONST_MKSA_BOLTZMANN,
  1.380649E-23, //from CODATA 2022

  /**
   * Stefan-Boltzmann constant in units of kg/s^3 / K^4
   */
  //GSL_CONST_MKSA_STEFAN_BOLTZMANN_CONSTANT,
  5.670374419E-8, //from CODATA 2022

  /**
   * Planck's constant in units kg m^2 / s
   */
  //GSL_CONST_MKSA_PLANCKS_CONSTANT_H,
  6.62607015E-34, //from CODATA 2022

  /**
   * The speed of light in m/s
   */
  //GSL_CONST_MKSA_SPEED_OF_LIGHT,
  299792458.0, //from CODATA 2022

  /**
   * Electron volt to Joules convestion
   */
  //GSL_CONST_MKSA_ELECTRON_VOLT,
  1.602176634E-19,  //from CODATA 2022

  /**
   * neutrino mass splitting differences
   * See Lesgourgues and Pastor, 2012 for these values.
   * Adv. High Energy Phys. 2012 (2012) 608515,
   * arXiv:1212.6154, page 13
  */
  7.62E-5,
  2.55E-3,
  -2.43E-3
};



/* ------- ROUTINE: ccl_cosmology_create ------
INPUTS: ccl_parameters params
        ccl_configuration config
TASK: creates the ccl_cosmology struct and passes some values to it
DEFINITIONS:
chi: comoving distance [Mpc]
growth: growth function (density)
fgrowth: logarithmic derivative of the growth (density) (dlnD/da?)
E: E(a)=H(a)/H0
growth0: growth at z=0, defined to be 1
sigma: ?
p_lnl: nonlinear matter power spectrum at z=0?
computed_distances, computed_growth,
computed_power, computed_sigma: store status of the computations
*/
ccl_cosmology * ccl_cosmology_create(ccl_parameters params, ccl_configuration config)
{
  #ifndef USE_GSL_ERROR
    gsl_set_error_handler_off();
  #endif

  ccl_cosmology * cosmo = malloc(sizeof(ccl_cosmology));
  cosmo->params = params;
  cosmo->config = config;
  cosmo->gsl_params = ccl_user_gsl_params;
  cosmo->spline_params = ccl_user_spline_params;
  cosmo->spline_params.A_SPLINE_TYPE = gsl_interp_akima;
  cosmo->spline_params.K_SPLINE_TYPE = gsl_interp_akima;
  cosmo->spline_params.M_SPLINE_TYPE = gsl_interp_akima;
  cosmo->spline_params.D_SPLINE_TYPE = gsl_interp_akima;
  cosmo->spline_params.PNL_SPLINE_TYPE = gsl_interp2d_bicubic;
  cosmo->spline_params.PLIN_SPLINE_TYPE = gsl_interp2d_bicubic;
  cosmo->spline_params.CORR_SPLINE_TYPE = gsl_interp_akima;

  cosmo->data.chi = NULL;
  cosmo->data.growth = NULL;
  cosmo->data.fgrowth = NULL;
  cosmo->data.E = NULL;
  cosmo->data.growth0 = 1.;
  cosmo->data.achi = NULL;

  cosmo->data.logsigma = NULL;

  cosmo->data.rsd_splines[0] = NULL;
  cosmo->data.rsd_splines[1] = NULL;
  cosmo->data.rsd_splines[2] = NULL;

  cosmo->computed_distances = false;
  cosmo->computed_growth = false;
  cosmo->computed_sigma = false;
  cosmo->status = 0;
  // Initialise as 0-length string
  cosmo->status_message[0] = '\0';

  if(cosmo->spline_params.A_SPLINE_MAX !=1.) {
    cosmo->status = CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo,
                                     "ccl_core.c: ccl_cosmology_create(): "
                                     "A_SPLINE_MAX needs to be 1.\n");
  }

  return cosmo;
}

/* ------ ROUTINE: ccl_parameters_fill_initial -------
INPUT: ccl_parameters: params
TASK: fill parameters not set by ccl_parameters_create with some initial values
DEFINITIONS:
Omega_g = (Omega_g*h^2)/h^2 is the radiation parameter; "g" is for photons, as in CLASS
T_CMB: CMB temperature in Kelvin
Omega_l: Lambda
A_s: amplitude of the primordial PS
sigma8: variance in 8 Mpc/h spheres for normalization of matter PS
z_star: recombination redshift
 */
void ccl_parameters_fill_initial(ccl_parameters * params, int *status)
{
  // Fixed radiation parameters
  // Omega_g * h**2 is known from T_CMB
  // kg / m^3
  double rho_g = 4. * ccl_constants.STBOLTZ / pow(ccl_constants.CLIGHT, 3) * pow(params->T_CMB, 4);
  // kg / m^3
  double rho_crit =
    ccl_constants.RHO_CRITICAL *
    ccl_constants.SOLAR_MASS/pow(ccl_constants.MPC_TO_METER, 3) *
    pow(params->h, 2);

  // Get the N_nu_rel from Neff and N_nu_mass
  params->N_nu_rel = params->Neff - params->N_nu_mass * pow(params->T_ncdm, 4) / pow(4./11.,4./3.);

  // Temperature of the relativistic neutrinos in K
  double T_nu= (params->T_CMB) * pow(4./11.,1./3.);
  // in kg / m^3
  double rho_nu_rel =
    params->N_nu_rel* 7.0/8.0 * 4. *
    ccl_constants.STBOLTZ / pow(ccl_constants.CLIGHT, 3) *
    pow(T_nu, 4);
  params-> Omega_nu_rel = rho_nu_rel/rho_crit;

  // If non-relativistic neutrinos are present, calculate the phase_space integral.
  if((params->N_nu_mass)>0) {
    params->Omega_nu_mass = ccl_Omeganuh2(
      1.0, params->N_nu_mass, params->m_nu, params->T_CMB, params->T_ncdm,
      status) / ((params->h)*(params->h));
  }
  else{
    params->Omega_nu_mass = 0.;
  }

  params->Omega_m = params->Omega_b + params-> Omega_c + params->Omega_nu_mass;
  params->Omega_l = 1.0 - params->Omega_m - rho_g/rho_crit - params->Omega_nu_rel - params->Omega_k;

  if (isnan(params->Omega_g)) {
    // No value passed for Omega_g
    params->Omega_g = rho_g/rho_crit;
  }
  else {
    // Omega_g was passed - modify Omega_l
    double total = rho_g/rho_crit + params->Omega_l;
    params->Omega_l = total - params->Omega_g;
  }

  // NULL to NAN in case it is not set
  params->z_star = NAN;

  if(fabs(params->Omega_k)<1E-6)
    params->k_sign=0;
  else if(params->Omega_k>0)
    params->k_sign=-1;
  else
    params->k_sign=1;
  params->sqrtk=sqrt(fabs(params->Omega_k))*params->h/ccl_constants.CLIGHT_HMPC;
}


/* ------ ROUTINE: ccl_parameters_create -------
INPUT: numbers for the basic cosmological parameters needed by CCL
TASK: fill params with some initial values provided by the user
DEFINITIONS:
Omega_c: cold dark matter
Omega_b: baryons
Omega_m: matter
Omega_k: curvature
little omega_x means Omega_x*h^2
Neff : Effective number of neutrino speces
mnu : Pointer to either sum of neutrino masses or list of three masses.
mnu_type : how the neutrino mass(es) should be treated
w0: Dark energy eq of state parameter
wa: Dark energy eq of state parameter, time variation
H0: Hubble's constant in km/s/Mpc.
h: Hubble's constant divided by (100 km/s/Mpc).
A_s: amplitude of the primordial PS
sigma8: variance of matter density fluctuations at 8 Mpc/h
n_s: index of the primordial PS
T_CMB: CMB temperature
Omega_g: radiation density parameter

 */
ccl_parameters ccl_parameters_create(double Omega_c, double Omega_b, double Omega_k,
				     double Neff, double* mnu, int n_mnu,
				     double w0, double wa, double h, double A_s, double sigma8,
				     double n_s, double T_CMB, double Omega_g, double T_ncdm,
				     double bcm_log10Mc, double bcm_etab, double bcm_ks,
				     double mu_0, double sigma_0,
				     double c1_mg, double c2_mg, double lambda_mg,
				     int nz_mgrowth, double *zarr_mgrowth,
				     double *dfarr_mgrowth, int *status)
{
  ccl_parameters params;
  // Initialize params
  params.m_nu = NULL;
  params.z_mgrowth=NULL;
  params.df_mgrowth=NULL;
  params.m_nu = malloc(n_mnu*sizeof(double));

  // Neutrinos
  params.sum_nu_masses = 0.;
  for(int i = 0; i<n_mnu; i=i+1){
     params.m_nu[i] = mnu[i];
     params.sum_nu_masses = params.sum_nu_masses + mnu[i];
  }
  if(params.sum_nu_masses<1e-15){
    params.N_nu_mass = 0;
  }else{
    params.N_nu_mass = n_mnu;
   }
  params.Neff = Neff;
  params.T_ncdm = T_ncdm;

  // Matter & curvature
  params.Omega_c = Omega_c;
  params.Omega_b = Omega_b;
  params.Omega_k = Omega_k;

  // Radiation
  params.T_CMB = T_CMB;
  params.Omega_g = Omega_g;

  // Dark Energy
  params.w0 = w0;
  params.wa = wa;

  // Hubble parameters
  params.h = h;
  params.H0 = h*100;

  // Primordial power spectra
  params.A_s = A_s;
  params.sigma8 = sigma8;
  params.n_s = n_s;

  //Baryonic params
  if(bcm_log10Mc<0)
    params.bcm_log10Mc=log10(1.2e14);
  else
    params.bcm_log10Mc=bcm_log10Mc;
  if(bcm_etab<0)
    params.bcm_etab=0.5;
  else
    params.bcm_etab=bcm_etab;
  if(bcm_ks<0)
    params.bcm_ks=55.0;
  else
    params.bcm_ks=bcm_ks;

  // Params of the mu / Sigma parameterisation of MG
  params.mu_0 = mu_0;
  params.sigma_0 = sigma_0;
  params.c1_mg = c1_mg;
  params.c2_mg = c2_mg;
  params.lambda_mg = lambda_mg;

  // Set remaining standard and easily derived parameters
  ccl_parameters_fill_initial(&params, status);

  //Trigger modified growth function if nz>0
  if(nz_mgrowth>0) {
    params.has_mgrowth=true;
    params.nz_mgrowth=nz_mgrowth;
    params.z_mgrowth=malloc(params.nz_mgrowth*sizeof(double));
    params.df_mgrowth=malloc(params.nz_mgrowth*sizeof(double));
    memcpy(params.z_mgrowth,zarr_mgrowth,params.nz_mgrowth*sizeof(double));
    memcpy(params.df_mgrowth,dfarr_mgrowth,params.nz_mgrowth*sizeof(double));
  }
  else {
    params.has_mgrowth=false;
    params.nz_mgrowth=0;
    params.z_mgrowth=NULL;
    params.df_mgrowth=NULL;
  }

  return params;
}


/* ------- ROUTINE: ccl_data_free --------
INPUT: ccl_data
TASK: free the input data
*/
void ccl_data_free(ccl_data * data) {
  //We cannot assume that all of these have been allocated
  //TODO: it would actually make more sense to do this within ccl_cosmology_free,
  //where we could make use of the flags "computed_distances" etc. to figure out
  //what to free up
  gsl_spline_free(data->chi);
  gsl_spline_free(data->growth);
  gsl_spline_free(data->fgrowth);
  gsl_spline_free(data->E);
  gsl_spline_free(data->achi);
  gsl_spline2d_free(data->logsigma);
  ccl_f1d_t_free(data->rsd_splines[0]);
  ccl_f1d_t_free(data->rsd_splines[1]);
  ccl_f1d_t_free(data->rsd_splines[2]);
}

/* ------- ROUTINE: ccl_cosmology_set_status_message --------
INPUT: ccl_cosmology struct, status_string
TASK: set the status message safely.
*/
void ccl_cosmology_set_status_message(ccl_cosmology * cosmo, const char * message, ...) {
  const int trunc = 480; /* must be < 500 - 4 */

  va_list va;
  va_start(va, message);

  #pragma omp critical
  {
    if(strlen(cosmo->status_message) != 0) {
      ccl_raise_warning(CCL_ERROR_OVERWRITE, "Status message being overwritten:");
      fprintf(stderr, "STATUS: %d. %s\n", cosmo->status, cosmo->status_message);
    }
    vsnprintf(cosmo->status_message, trunc, message, va);

    /* if truncation happens, message[trunc - 1] is not NULL, ... will show up. */
    strcpy(&cosmo->status_message[trunc], "...");
  }

  va_end(va);
}

/* ------- ROUTINE: ccl_parameters_free --------
INPUT: ccl_parameters struct
TASK: free allocated quantities in the parameters struct
*/
void ccl_parameters_free(ccl_parameters * params) {
  if (params->m_nu != NULL){
    free(params->m_nu);
    params->m_nu = NULL;
  }
  if (params->z_mgrowth != NULL){
    free(params->z_mgrowth);
    params->z_mgrowth = NULL;
  }
  if (params->df_mgrowth != NULL){
    free(params->df_mgrowth);
    params->df_mgrowth = NULL;
  }
}


/* ------- ROUTINE: ccl_cosmology_free --------
INPUT: ccl_cosmology struct
TASK: free the input data and the cosmology struct
*/
void ccl_cosmology_free(ccl_cosmology * cosmo) {
  if (cosmo != NULL)
    ccl_data_free(&cosmo->data);
  free(cosmo);
}


int ccl_get_pk_spline_na(ccl_cosmology *cosmo) {
  return cosmo->spline_params.A_SPLINE_NA_PK + cosmo->spline_params.A_SPLINE_NLOG_PK - 1;
}

void ccl_get_pk_spline_a_array(ccl_cosmology *cosmo,int ndout,double* doutput,int *status) {
  double *d = NULL;
  if (ndout != ccl_get_pk_spline_na(cosmo))
    *status = CCL_ERROR_INCONSISTENT;
  if (*status == 0) {
    d = ccl_linlog_spacing(cosmo->spline_params.A_SPLINE_MINLOG_PK,
      cosmo->spline_params.A_SPLINE_MIN_PK,
      cosmo->spline_params.A_SPLINE_MAX,
      cosmo->spline_params.A_SPLINE_NLOG_PK,
      cosmo->spline_params.A_SPLINE_NA_PK);
    if (d == NULL)
      *status = CCL_ERROR_MEMORY;
  }
  if(*status==0)
    memcpy(doutput, d, ndout*sizeof(double));

  free(d);
}

void ccl_get_pk_spline_a_array_from_params(ccl_spline_params *spline_params,
                                           int ndout, double *doutput, int *status) {
  double *d = NULL;
  if (*status == 0) {
    d = ccl_linlog_spacing(spline_params->A_SPLINE_MINLOG_PK,
      spline_params->A_SPLINE_MIN_PK,
      spline_params->A_SPLINE_MAX,
      spline_params->A_SPLINE_NLOG_PK,
      spline_params->A_SPLINE_NA_PK);
    if (d == NULL)
      *status = CCL_ERROR_MEMORY;
  }
  if(*status==0)
    memcpy(doutput, d, ndout*sizeof(double));

  free(d);
}

int ccl_get_pk_spline_nk(ccl_cosmology *cosmo) {
  double ndecades = log10(cosmo->spline_params.K_MAX) - log10(cosmo->spline_params.K_MIN);
  return (int)ceil(ndecades*cosmo->spline_params.N_K);
}

void ccl_get_pk_spline_lk_array(ccl_cosmology *cosmo,int ndout,double* doutput,int *status) {
  double *d = NULL;
  if (ndout != ccl_get_pk_spline_nk(cosmo))
    *status = CCL_ERROR_INCONSISTENT;
  if (*status == 0) {
    d = ccl_log_spacing(cosmo->spline_params.K_MIN, cosmo->spline_params.K_MAX, ndout);
    if (d == NULL)
      *status = CCL_ERROR_MEMORY;
  }
  if (*status == 0) {
    for(int ii=0; ii < ndout; ii++)
      doutput[ii] = log(d[ii]);
  }
  free(d);
}

void ccl_get_pk_spline_lk_array_from_params(ccl_spline_params *spline_params,
                                            int ndout, double *doutput, int *status) {
  double *d = NULL;
  if (*status == 0) {
    d = ccl_log_spacing(spline_params->K_MIN, spline_params->K_MAX, ndout);
    if (d == NULL)
      *status = CCL_ERROR_MEMORY;
  }
  if (*status == 0) {
    for(int ii=0; ii < ndout; ii++)
      doutput[ii] = log(d[ii]);
  }
  free(d);
}
