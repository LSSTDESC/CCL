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

//
// Macros for replacing relative paths
#define EXPAND_STR(s) STRING(s)
#define STRING(s) #s


const ccl_configuration default_config = {
  ccl_boltzmann_class, ccl_halofit, ccl_nobaryons,
  ccl_tinker10, ccl_duffy2008, ccl_emu_strict};


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

const ccl_gsl_params default_gsl_params = {
  GSL_N_ITERATION,                     // N_ITERATION
  GSL_INTEGRATION_GAUSS_KRONROD_POINTS,// INTEGRATION_GAUSS_KRONROD_POINTS
  GSL_EPSREL,                          // INTEGRATION_EPSREL
  GSL_INTEGRATION_GAUSS_KRONROD_POINTS,// INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS
  GSL_EPSREL,                          // INTEGRATION_LIMBER_EPSREL
  GSL_EPSREL_DIST,                     // INTEGRATION_DISTANCE_EPSREL
  GSL_EPSREL_SIGMAR,                   // INTEGRATION_SIGMAR_EPSREL
  GSL_EPSREL,                          // ROOT_EPSREL
  GSL_N_ITERATION,                     // ROOT_N_ITERATION
  GSL_EPSREL_GROWTH,                   // ODE_GROWTH_EPSREL
  1E-6,                                // EPS_SCALEFAC_GROWTH
  1E7,                                 // HM_MMIN
  1E17,                                // HM_MMAX
  0.0,                                 // HM_EPSABS
  1E-4,                                // HM_EPSREL
  1000,                                // HM_LIMIT
  GSL_INTEG_GAUSS41                    // HM_INT_METHOD
  };

#undef GSL_EPSREL
#undef GSL_N_ITERATION
#undef GSL_INTEGRATION_GAUSS_KRONROD_POINTS
#undef GSL_EPSREL_SIGMAR
#undef GSL_EPSREL_DIST
#undef GSL_EPSREL_GROWTH
#undef GSL_EPSREL_DNDZ


const ccl_spline_params default_spline_params = {

  // scale factor spline params
  250,  // A_SPLINE_NA
  0.1,  // A_SPLINE_MIN
  0.01,  // A_SPLINE_MINLOG_PK
  0.1,  // A_SPLINE_MIN_PK,
  1.0,  // A_SPLINE_MAX,
  0.0001,  // A_SPLINE_MINLOG,
  250,  // A_SPLINE_NLOG,

  // mass splines
  0.025,  // LOGM_SPLINE_DELTA
  440,  // LOGM_SPLINE_NM
  6,  // LOGM_SPLINE_MIN
  17,  // LOGM_SPLINE_MAX

  // PS a and k spline
  40,  // A_SPLINE_NA_PK
  11,  // A_SPLINE_NLOG_PK

  // k-splines and integrals
  50,  // K_MAX_SPLINE
  1E3,  // K_MAX
  5E-5,  // K_MIN
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
   * Lightspeed / H0 in units of Mpc/h (from CODATA 2014)
   */
  2997.92458,

  /**
   * Newton's gravitational constant in units of m^3/Kg/s^2
   */
  //6.6738e-11,  /(from PDG 2013) in m^3/Kg/s^2
  //6.67428e-11, // CLASS VALUE
  6.67408e-11, // from CODATA 2014

  /**
   * Solar mass in units of kg (from GSL)
   */
  //GSL_CONST_MKSA_SOLAR_MASS,
  //1.9885e30, //(from PDG 2015) in Kg
  1.9884754153381438E+30, //from IAU 2015

  /**
   * Mpc to meters (from PDG 2016 and using M_PI)
   */
  3.085677581491367399198952281E+22,

  /**
   * pc to meters (from PDG 2016 and using M_PI)
   */
  3.085677581491367399198952281E+16,

  /**
   * Rho critical in units of M_sun/h / (Mpc/h)^3
   */
  ((3*100*100)/(8*M_PI*6.67408e-11)) * (1000*1000*3.085677581491367399198952281E+22/1.9884754153381438E+30),

  /**
   * Boltzmann constant in units of J/K
  */
  //GSL_CONST_MKSA_BOLTZMANN,
  1.38064852E-23, //from CODATA 2014

  /**
   * Stefan-Boltzmann constant in units of kg/s^3 / K^4
   */
  //GSL_CONST_MKSA_STEFAN_BOLTZMANN_CONSTANT,
  5.670367E-8, //from CODATA 2014

  /**
   * Planck's constant in units kg m^2 / s
   */
  //GSL_CONST_MKSA_PLANCKS_CONSTANT_H,
  6.626070040E-34, //from CODATA 2014

  /**
   * The speed of light in m/s
   */
  //GSL_CONST_MKSA_SPEED_OF_LIGHT,
  299792458.0, //from CODATA 2014

  /**
   * Electron volt to Joules convestion
   */
  //GSL_CONST_MKSA_ELECTRON_VOLT,
  1.6021766208e-19,  //from CODATA 2014

  /**
   * Temperature of the CMB in K
   */
  2.725,
  //2.7255, // CLASS value

  /**
   * T_ncdm, as taken from CLASS, explanatory.ini
   */
  0.71611,

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
p_lin: linear matter power spectrum at z=0?
p_lnl: nonlinear matter power spectrum at z=0?
computed_distances, computed_growth,
computed_power, computed_sigma: store status of the computations
*/
ccl_cosmology * ccl_cosmology_create(ccl_parameters params, ccl_configuration config)
{
  ccl_cosmology * cosmo = malloc(sizeof(ccl_cosmology));
  cosmo->params = params;
  cosmo->config = config;
  cosmo->gsl_params = default_gsl_params;
  cosmo->spline_params = default_spline_params;
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
  cosmo->data.dlnsigma_dlogm = NULL;

  // hmf parameter for interpolation
  cosmo->data.alphahmf = NULL;
  cosmo->data.betahmf = NULL;
  cosmo->data.gammahmf = NULL;
  cosmo->data.phihmf = NULL;
  cosmo->data.etahmf = NULL;

  cosmo->data.rsd_splines[0] = NULL;
  cosmo->data.rsd_splines[1] = NULL;
  cosmo->data.rsd_splines[2] = NULL;

  cosmo->data.p_lin = NULL;
  cosmo->data.p_nl = NULL;
  cosmo->computed_distances = false;
  cosmo->computed_growth = false;
  cosmo->computed_power = false;
  cosmo->computed_sigma = false;
  cosmo->computed_hmfparams = false;
  cosmo->status = 0;
  ccl_cosmology_set_status_message(cosmo, "");


  return cosmo;
}

/* ------ ROUTINE: ccl_parameters_fill_initial -------
INPUT: ccl_parameters: params
TASK: fill parameters not set by ccl_parameters_create with some initial values
DEFINITIONS:
Omega_g = (Omega_g*h^2)/h^2 is the radiation parameter; "g" is for photons, as in CLASS
T_CMB: CMB temperature in Kelvin
Omega_l: Lambda
A_s: amplitude of the primordial PS, enforced here to initially set to NaN
sigma8: variance in 8 Mpc/h spheres for normalization of matter PS, enforced here to initially set to NaN
z_star: recombination redshift
 */
void ccl_parameters_fill_initial(ccl_parameters * params, int *status)
{
  // Fixed radiation parameters
  // Omega_g * h**2 is known from T_CMB
  params->T_CMB =  ccl_constants.T_CMB;
  // kg / m^3
  double rho_g = 4. * ccl_constants.STBOLTZ / pow(ccl_constants.CLIGHT, 3) * pow(params->T_CMB, 4);
  // kg / m^3
  double rho_crit =
    ccl_constants.RHO_CRITICAL *
    ccl_constants.SOLAR_MASS/pow(ccl_constants.MPC_TO_METER, 3) *
    pow(params->h, 2);
  params->Omega_g = rho_g/rho_crit;

  // Get the N_nu_rel from Neff and N_nu_mass
  params->N_nu_rel = params->Neff - params->N_nu_mass * pow(ccl_constants.TNCDM, 4) / pow(4./11.,4./3.);

  // Temperature of the relativistic neutrinos in K
  double T_nu= (params->T_CMB) * pow(4./11.,1./3.);
  // in kg / m^3
  double rho_nu_rel =
    params->N_nu_rel* 7.0/8.0 * 4. *
    ccl_constants.STBOLTZ / pow(ccl_constants.CLIGHT, 3) *
    pow(T_nu, 4);
  params-> Omega_n_rel = rho_nu_rel/rho_crit;

  // If non-relativistic neutrinos are present, calculate the phase_space integral.
  if((params->N_nu_mass)>0) {
    params->Omega_n_mass = ccl_Omeganuh2(
      1.0, params->N_nu_mass, params->mnu, params->T_CMB, status) / ((params->h)*(params->h));
    ccl_check_status_nocosmo(status);
  }
  else{
    params->Omega_n_mass = 0.;
  }

  params->Omega_m = params->Omega_b + params-> Omega_c;
  params->Omega_l = 1.0 - params->Omega_m - params->Omega_g - params->Omega_n_rel -params->Omega_n_mass- params->Omega_k;
  // Initially undetermined parameters - set to nan to trigger
  // problems if they are mistakenly used.
  if (isfinite(params->A_s)) {params->sigma8 = NAN;}
  if (isfinite(params->sigma8)) {params->A_s = NAN;}
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
n_s: index of the primordial PS

 */
ccl_parameters ccl_parameters_create(
                     double Omega_c, double Omega_b, double Omega_k,
				     double Neff, double* mnu, ccl_mnu_convention mnu_type,
				     double w0, double wa, double h, double norm_pk,
				     double n_s, double bcm_log10Mc, double bcm_etab, 
				     double bcm_ks, double mu_0, double sigma_0, 
				     int nz_mgrowth, double *zarr_mgrowth,
				     double *dfarr_mgrowth, int *status)
{
  #ifndef USE_GSL_ERROR
    gsl_set_error_handler_off ();
  #endif

  ccl_parameters params;
  // Initialize params
  params.mnu = NULL;
  params.z_mgrowth=NULL;
  params.df_mgrowth=NULL;
  params.sigma8 = NAN;
  params.A_s = NAN;
  params.Omega_c = Omega_c;
  params.Omega_b = Omega_b;
  params.Omega_k = Omega_k;
  params.Neff = Neff;

    double *mnu_in = NULL;

  // Decide how to split sum of neutrino masses between 3 neutrinos. We use
  // a Newton's rule numerical solution (thanks M. Jarvis).

  if (mnu_type==ccl_mnu_sum){
	  // Normal hierarchy

	  // Set the sum of neutrino masses
      params.sum_nu_masses = *mnu;

	  mnu_in = malloc(3*sizeof(double));
	  
	  // Check if the sum is zero
	  if (*mnu<1e-15){
		  mnu_in[0] = 0.;
		  mnu_in[1] = 0.;
		  mnu_in[2] = 0.;
	  }
    else{
	      mnu_in[0] = 0.; // This is a starting guess.

	      double sum_check;
	      // Check that sum is consistent
	      mnu_in[1] = sqrt(ccl_constants.DELTAM12_sq);
	      mnu_in[2] = sqrt(ccl_constants.DELTAM13_sq_pos);
	      sum_check = mnu_in[0] + mnu_in[1] + mnu_in[2];
	      if (*mnu < sum_check){
		      *status = CCL_ERROR_MNU_UNPHYSICAL;
		      ccl_check_status_nocosmo(status);
          }
          double dsdm1;
          // This is the Newton's method
          while (fabs(*mnu - sum_check) > 1e-15){

              dsdm1 = 1. + mnu_in[0] / mnu_in[1] + mnu_in[0] / mnu_in[2];
              mnu_in[0] = mnu_in[0] - (sum_check - *mnu) / dsdm1;
              mnu_in[1] = sqrt(mnu_in[0]*mnu_in[0] + ccl_constants.DELTAM12_sq);
              mnu_in[2] = sqrt(mnu_in[0]*mnu_in[0] + ccl_constants.DELTAM13_sq_pos);
              sum_check = mnu_in[0] + mnu_in[1] + mnu_in[2];
          }
	  }

  } else if (mnu_type==ccl_mnu_sum_inverted){
	  // Inverted hierarchy

	  // Set the sum of neutrino masses
      params.sum_nu_masses = *mnu;

	  mnu_in = malloc(3*sizeof(double));

	  	  // Check if the sum is zero
	  if (*mnu<1e-15){
		  mnu_in[0] = 0.;
		  mnu_in[1] = 0.;
		  mnu_in[2] = 0.;
	  } else{

	      mnu_in[0] = 0.; // This is a starting guess.

	      double sum_check;
	      // Check that sum is consistent
	      mnu_in[1] = sqrt(-1.* ccl_constants.DELTAM13_sq_neg - ccl_constants.DELTAM12_sq);
	      mnu_in[2] = sqrt(-1.* ccl_constants.DELTAM13_sq_neg);
	      sum_check = mnu_in[0] + mnu_in[1] + mnu_in[2];
	      if (*mnu < sum_check){
		      *status = CCL_ERROR_MNU_UNPHYSICAL;
		      ccl_check_status_nocosmo(status);
          }
          double dsdm1;
          // This is the Newton's method
          while (fabs(*mnu- sum_check) > 1e-15){
              dsdm1 = 1. + (mnu_in[0] / mnu_in[1]) + (mnu_in[0] / mnu_in[2]);
              mnu_in[0] = mnu_in[0] - (sum_check - *mnu) / dsdm1;
              mnu_in[1] = sqrt(mnu_in[0]*mnu_in[0] + ccl_constants.DELTAM12_sq);
              mnu_in[2] = sqrt(mnu_in[0]*mnu_in[0] + ccl_constants.DELTAM13_sq_neg);
              sum_check = mnu_in[0] + mnu_in[1] + mnu_in[2];
          }

      }

  } else if (mnu_type==ccl_mnu_sum_equal){

	    // Set the sum of neutrino masses
        params.sum_nu_masses = *mnu;

	    // Split the sum of masses equally
	    mnu_in = malloc(3*sizeof(double));
	    mnu_in[0] = params.sum_nu_masses / 3.;
	    mnu_in[1] = params.sum_nu_masses / 3.;
	    mnu_in[2] = params.sum_nu_masses / 3.;
  } else if (mnu_type == ccl_mnu_list){
      // A list of neutrino masses was already passed in
	  params.sum_nu_masses = mnu[0] + mnu[1] + mnu[2];
	  mnu_in = malloc(3*sizeof(double));
	  for(int i=0; i<3; i++) mnu_in[i] = mnu[i];
  } else {
	  *status = CCL_ERROR_NOT_IMPLEMENTED;
  }

  // Check for errors in the neutrino set up (e.g. unphysical mnu)
  ccl_check_status_nocosmo(status);

  // Check which of the neutrino species are non-relativistic today
  int N_nu_mass = 0;
  for(int i = 0; i<3; i=i+1){
  	if (mnu_in[i] > 0.00017){ // Limit taken from Lesgourges et al. 2012
  		N_nu_mass = N_nu_mass + 1;
  	}
  }
  params.N_nu_mass = N_nu_mass;

  // Fill the array of massive neutrinos
  if (N_nu_mass>0){
  	params.mnu = malloc(params.N_nu_mass*sizeof(double));
  	int relativistic[3] = {0, 0, 0};
	for (int i = 0; i < N_nu_mass; i = i + 1){
		for (int j = 0; j<3; j = j +1){
			if ((mnu_in[j]>0.00017) && (relativistic[j]==0)){
				relativistic[j]=1;
				params.mnu[i] = mnu_in[j];
				break;
			}
		} // end loop over neutrinos
	} // end loop over massive neutrinos
  } else{
	  params.mnu = malloc(sizeof(double));
	  params.mnu[0] = 0.;
  }
  // Free mnu_in
  if (mnu_in != NULL) free(mnu_in);

  // Dark Energy
  params.w0 = w0;
  params.wa = wa;

  // Hubble parameters
  params.h = h;
  params.H0 = h*100;

  // Primordial power spectra
  if(norm_pk<1E-5)
    params.A_s=norm_pk;
  else
    params.sigma8=norm_pk;
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


/* ------- ROUTINE: ccl_parameters_create_flat_lcdm --------
INPUT: some cosmological parameters needed to create a flat LCDM model
TASK: call ccl_parameters_create to produce an LCDM model
*/
ccl_parameters ccl_parameters_create_flat_lcdm(double Omega_c, double Omega_b, double h,
                                               double norm_pk, double n_s, int *status)
{
  double Omega_k = 0.0;
  double Neff = 3.046;
  double w0 = -1.0;
  double wa = 0.0;
  double *mnu;
  double mnuval = 0.;  // a pointer to the variable is not kept past the lifetime of this function
  mnu = &mnuval;
  ccl_mnu_convention mnu_type = ccl_mnu_sum;
  double mu_0 = 0.;
  double sigma_0 = 0.;
  
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Neff,
						mnu, mnu_type, w0, wa, h, norm_pk, n_s, -1, -1, -1, mu_0, sigma_0, -1, NULL, NULL, status);
  return params;

}


/**
 * Write a cosmology parameters object to a file in yaml format.
 * @param cosmo Cosmological parameters
 * @param f FILE* pointer opened for reading
 * @return void
 */
void ccl_parameters_write_yaml(ccl_parameters * params, const char * filename, int *status)
{
  FILE * f = fopen(filename, "w");

  if (!f){
    *status = CCL_ERROR_FILE_WRITE;
    return;
  }

#define WRITE_DOUBLE(name) fprintf(f, #name ": %le\n",params->name)
#define WRITE_INT(name) fprintf(f, #name ": %d\n",params->name)

  // Densities: CDM, baryons, total matter, curvature
  WRITE_DOUBLE(Omega_c);
  WRITE_DOUBLE(Omega_b);
  WRITE_DOUBLE(Omega_m);
  WRITE_DOUBLE(Omega_k);
  WRITE_INT(k_sign);

  // Dark Energy
  WRITE_DOUBLE(w0);
  WRITE_DOUBLE(wa);

  // Hubble parameters
  WRITE_DOUBLE(H0);
  WRITE_DOUBLE(h);

  // Neutrino properties
  WRITE_DOUBLE(Neff);
  WRITE_INT(N_nu_mass);
  WRITE_DOUBLE(N_nu_rel);

  if (params->N_nu_mass>0){
    fprintf(f, "mnu: [");
    for (int i=0; i<params->N_nu_mass; i++){
      fprintf(f, "%le, ", params->mnu[i]);
    }
    fprintf(f, "]\n");
  }

  WRITE_DOUBLE(sum_nu_masses);
  WRITE_DOUBLE(Omega_n_mass);
  WRITE_DOUBLE(Omega_n_rel);

  // Primordial power spectra
  WRITE_DOUBLE(A_s);
  WRITE_DOUBLE(n_s);

  // Radiation parameters
  WRITE_DOUBLE(Omega_g);
  WRITE_DOUBLE(T_CMB);

  // BCM baryonic model parameters
  WRITE_DOUBLE(bcm_log10Mc);
  WRITE_DOUBLE(bcm_etab);
  WRITE_DOUBLE(bcm_ks);
  
  // Modified gravity parameters
  WRITE_DOUBLE(mu_0);
  WRITE_DOUBLE(sigma_0);

  // Derived parameters
  WRITE_DOUBLE(sigma8);
  WRITE_DOUBLE(Omega_l);
  WRITE_DOUBLE(z_star);

  WRITE_INT(has_mgrowth);
  WRITE_INT(nz_mgrowth);

  if (params->has_mgrowth){
    fprintf(f, "z_mgrowth: [");
    for (int i=0; i<params->nz_mgrowth; i++){
      fprintf(f, "%le, ", params->z_mgrowth[i]);
    }
    fprintf(f, "]\n");

    fprintf(f, "df_mgrowth: [");
    for (int i=0; i<params->nz_mgrowth; i++){
      fprintf(f, "%le, ", params->df_mgrowth[i]);
    }
    fprintf(f, "]\n");
  }

#undef WRITE_DOUBLE
#undef WRITE_INT

  fclose(f);

}

/**
 * Write a cosmology parameters object to a file in yaml format.
 * @param cosmo Cosmological parameters
 * @param f FILE* pointer opened for reading
 * @return void
 */
ccl_parameters ccl_parameters_read_yaml(const char * filename, int *status)
{
  FILE * f = fopen(filename, "r");

  if (!f){
    *status = CCL_ERROR_FILE_READ;
    ccl_parameters bad_params;
    ccl_raise_exception(CCL_ERROR_FILE_READ, "ccl_core.c: Failed to read parameters from file.");

    return bad_params;
  }

#define READ_DOUBLE(name) double name; *status |= (0==fscanf(f, #name ": %le\n",&name));
#define READ_INT(name) int name; *status |= (0==fscanf(f, #name ": %d\n",&name))

  // Densities: CDM, baryons, total matter, curvature
  READ_DOUBLE(Omega_c);
  READ_DOUBLE(Omega_b);
  READ_DOUBLE(Omega_m);
  READ_DOUBLE(Omega_k);
  READ_INT(k_sign);

  // Dark Energy
  READ_DOUBLE(w0);
  READ_DOUBLE(wa);

  // Hubble parameters
  READ_DOUBLE(H0);
  READ_DOUBLE(h);

  // Neutrino properties
  READ_DOUBLE(Neff);
  READ_INT(N_nu_mass);
  READ_DOUBLE(N_nu_rel);

  double mnu[3] = {0.0, 0.0, 0.0};
  if (N_nu_mass>0){
    *status |= (0==fscanf(f, "mnu: ["));
    for (int i=0; i<N_nu_mass; i++){
      *status |= (0==fscanf(f, "%le, ", mnu+i));
    }
    *status |= (0==fscanf(f, "]\n"));
  }

  READ_DOUBLE(sum_nu_masses);
  READ_DOUBLE(Omega_n_mass);
  READ_DOUBLE(Omega_n_rel);

  // Primordial power spectra
  READ_DOUBLE(A_s);
  READ_DOUBLE(n_s);

  // Radiation parameters
  READ_DOUBLE(Omega_g);
  READ_DOUBLE(T_CMB);

  // BCM baryonic model parameters
  READ_DOUBLE(bcm_log10Mc);
  READ_DOUBLE(bcm_etab);
  READ_DOUBLE(bcm_ks);
  
  // Modified gravity parameters
  READ_DOUBLE(mu_0);
  READ_DOUBLE(sigma_0);

  // Derived parameters
  READ_DOUBLE(sigma8);
  READ_DOUBLE(Omega_l);
  READ_DOUBLE(z_star);

  READ_INT(has_mgrowth);
  READ_INT(nz_mgrowth);

  double *z_mgrowth;
  double *df_mgrowth;


  if (has_mgrowth){
    z_mgrowth = malloc(nz_mgrowth*sizeof(double));
    df_mgrowth = malloc(nz_mgrowth*sizeof(double));
    *status |= (0==fscanf(f, "z_mgrowth: ["));
    for (int i=0; i<nz_mgrowth; i++){
      *status |= (0==fscanf(f, "%le, ", z_mgrowth+i));
    }
    *status |= (0==fscanf(f, "]\n"));

    *status |= (0==fscanf(f, "df_mgrowth: ["));
    for (int i=0; i<nz_mgrowth; i++){
      *status |= (0==fscanf(f, "%le, ", df_mgrowth+i));
    }
    *status |= (0==fscanf(f, "]\n"));
  }
  else{
    z_mgrowth = NULL;
    df_mgrowth = NULL;
  }

#undef READ_DOUBLE
#undef READ_INT

  fclose(f);


  if (status){
    char msg[256];
    snprintf(msg, 256, "ccl_core.c: Structure of YAML file incorrect: %s", filename);
    ccl_raise_exception(*status, msg);
  }

  double norm_pk;

  if (isnan(A_s)){
    norm_pk = sigma8;
  }
  else{
   norm_pk = A_s;
  }

  ccl_parameters params = ccl_parameters_create(
    Omega_c, Omega_b, Omega_k,
    Neff, mnu, ccl_mnu_list,
    w0, wa, h, norm_pk,
    n_s, bcm_log10Mc, bcm_etab,
    bcm_ks, mu_0, sigma_0, nz_mgrowth, z_mgrowth,
    df_mgrowth, status);

  if(z_mgrowth) free(z_mgrowth);
  if (df_mgrowth) free(df_mgrowth);
  
  return params;

}



/* ------- ROUTINE: ccl_data_free --------
INPUT: ccl_data
TASK: free the input data
*/
void ccl_data_free(ccl_data * data)
{
  //We cannot assume that all of these have been allocated
  //TODO: it would actually make more sense to do this within ccl_cosmology_free,
  //where we could make use of the flags "computed_distances" etc. to figure out
  //what to free up
  gsl_spline_free(data->chi);
  gsl_spline_free(data->growth);
  gsl_spline_free(data->fgrowth);
  gsl_spline_free(data->E);
  gsl_spline_free(data->achi);
  gsl_spline_free(data->logsigma);
  gsl_spline_free(data->dlnsigma_dlogm);
  ccl_p2d_t_free(data->p_lin);
  ccl_p2d_t_free(data->p_nl);
  gsl_spline_free(data->alphahmf);
  gsl_spline_free(data->betahmf);
  gsl_spline_free(data->gammahmf);
  gsl_spline_free(data->phihmf);
  gsl_spline_free(data->etahmf);
  ccl_spline_free(data->rsd_splines[0]);
  ccl_spline_free(data->rsd_splines[1]);
  ccl_spline_free(data->rsd_splines[2]);
}

/* ------- ROUTINE: ccl_cosmology_set_status_message --------
INPUT: ccl_cosmology struct, status_string
TASK: set the status message safely.
*/
void ccl_cosmology_set_status_message(ccl_cosmology * cosmo, const char * message, ...)
{
  const int trunc = 480; /* must be < 500 - 4 */
  va_list va;
  va_start(va, message);
  vsnprintf(cosmo->status_message, trunc, message, va);
  va_end(va);

  /* if truncation happens, message[trunc - 1] is not NULL, ... will show up. */
  strcpy(&cosmo->status_message[trunc], "...");
}

/* ------- ROUTINE: ccl_parameters_free --------
INPUT: ccl_parameters struct
TASK: free allocated quantities in the parameters struct
*/
void ccl_parameters_free(ccl_parameters * params)
{
  if (params->mnu != NULL){
    free(params->mnu);
    params->mnu = NULL;
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
void ccl_cosmology_free(ccl_cosmology * cosmo)
{
  ccl_data_free(&cosmo->data);
  free(cosmo);
}

int ccl_get_pk_spline_na(ccl_cosmology *cosmo)
{
  return cosmo->spline_params.A_SPLINE_NA_PK + cosmo->spline_params.A_SPLINE_NLOG_PK - 1;
}

void ccl_get_pk_spline_a_array(ccl_cosmology *cosmo,int ndout,double* doutput,int *status)
{
  double *d;
  if(ndout!=ccl_get_pk_spline_na(cosmo))
    *status=CCL_ERROR_INCONSISTENT;
  if(*status==0) {
    d=ccl_linlog_spacing(cosmo->spline_params.A_SPLINE_MINLOG_PK,
			 cosmo->spline_params.A_SPLINE_MIN_PK,
			 cosmo->spline_params.A_SPLINE_MAX,
			 cosmo->spline_params.A_SPLINE_NLOG_PK,
			 cosmo->spline_params.A_SPLINE_NA_PK);
    if(d==NULL)
      *status=CCL_ERROR_MEMORY;
  }
  if(*status==0)
    memcpy(doutput,d,ndout*sizeof(double));
  free(d);
}

int ccl_get_pk_spline_nk(ccl_cosmology *cosmo)
{
  double ndecades = log10(cosmo->spline_params.K_MAX) - log10(cosmo->spline_params.K_MIN);
  return (int)ceil(ndecades*cosmo->spline_params.N_K);
}

void ccl_get_pk_spline_lk_array(ccl_cosmology *cosmo,int ndout,double* doutput,int *status)
{
  double *d;
  if(ndout!=ccl_get_pk_spline_nk(cosmo))
    *status=CCL_ERROR_INCONSISTENT;
  if(*status==0) {
    d=ccl_log_spacing(cosmo->spline_params.K_MIN,cosmo->spline_params.K_MAX,ndout);
    if(d==NULL)
      *status=CCL_ERROR_MEMORY;
  }
  if(*status==0) {
    for(int ii=0;ii<ndout;ii++)
      doutput[ii]=log(d[ii]);
  }
  free(d);
}
