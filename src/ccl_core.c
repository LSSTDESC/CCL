#include "ccl_core.h"
#include "ccl_neutrinos.h"
#include "ccl_utils.h"
#include "ccl_constants.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "gsl/gsl_errno.h"
#include "gsl/gsl_odeiv.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_integration.h"
#include "ccl_params.h"
#include "ccl_error.h"
#include <stdlib.h>

const ccl_configuration default_config = {ccl_boltzmann_class, ccl_halofit, ccl_nobaryons, ccl_tinker10, ccl_emu_strict};

const ccl_gsl_params default_gsl_params = {GSL_EPSREL,                          // EPSREL
                                           GSL_N_ITERATION,                     // N_ITERATION
                                           GSL_INTEGRATION_GAUSS_KRONROD_POINTS,// INTEGRATION_GAUSS_KRONROD_POINTS
                                           GSL_EPSREL,                          // INTEGRATION_EPSREL
                                           GSL_INTEGRATION_GAUSS_KRONROD_POINTS,// INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS
                                           GSL_EPSREL,                          // INTEGRATION_LIMBER_EPSREL
                                           GSL_EPSREL_DIST,                     // INTEGRATION_DISTANCE_EPSREL
                                           GSL_EPSREL_DNDZ,                     // INTEGRATION_DNDZ_EPSREL
                                           GSL_EPSREL_SIGMAR,                   // INTEGRATION_SIGMAR_EPSREL
                                           GSL_EPSREL_NU,                       // INTEGRATION_NU_EPSREL
                                           GSL_EPSABS_NU,                       // INTEGRATION_NU_EPSABS
                                           GSL_EPSREL,                          // ROOT_EPSREL
                                           GSL_N_ITERATION,                     // ROOT_N_ITERATION
                                           GSL_EPSREL_GROWTH                    // ODE_GROWTH_EPSREL
                                          };

/* ------- ROUTINE: ccl_cosmology_read_config ------
   INPUTS: none, but will look for ini file in include/ dir
   TASK: fill out global variables of splines with user defined input.
   The variables are defined in ccl_params.h.
   
   The following are the relevant global variables:
*/

ccl_spline_params * ccl_splines=NULL; // Global variable
ccl_gsl_params * ccl_gsl=NULL; // Global variable

void ccl_cosmology_read_config(void)
{

  int CONFIG_LINE_BUFFER_SIZE=100;
  int MAX_CONFIG_VAR_LEN=100;
  FILE *fconfig;
  char buf[CONFIG_LINE_BUFFER_SIZE];
  char var_name[MAX_CONFIG_VAR_LEN];
  char* rtn;
  double var_dbl;
  
  // Get parameter .ini filename from environment variable or default location
  const char* param_file;
  const char* param_file_env = getenv("CCL_PARAM_FILE");
  if (param_file_env != NULL) {
    param_file = param_file_env;
  }
  else {
    // Use default ini file
    param_file = EXPAND_STR(__CCL_DATA_DIR__) "/ccl_params.ini";
  }
  
  if ((fconfig=fopen(param_file, "r")) == NULL) {
    char msg[256];
    snprintf(msg, 256, "ccl_core.c: Failed to open config file: %s", param_file);
    ccl_raise_exception(CCL_ERROR_MISSING_CONFIG_FILE, msg);
    return;
  }

  if(ccl_splines == NULL) {
    ccl_splines = malloc(sizeof(ccl_spline_params));
  }
  if(ccl_gsl == NULL) {
    ccl_gsl = malloc(sizeof(ccl_gsl_params));
    memcpy(ccl_gsl, &default_gsl_params, sizeof(ccl_gsl_params));
  }

  while(! feof(fconfig)) {
    rtn = fgets(buf, CONFIG_LINE_BUFFER_SIZE, fconfig);
    if (buf[0]==';' || buf[0]=='[' || buf[0]=='\n') {
      continue;
    }
    else {
      sscanf(buf, "%99[^=]=%le\n",var_name, &var_dbl);
      // Spline parameters
      if(strcmp(var_name,"A_SPLINE_NA")==0) ccl_splines->A_SPLINE_NA=(int) var_dbl; 
      if(strcmp(var_name,"A_SPLINE_NLOG")==0) ccl_splines->A_SPLINE_NLOG=(int) var_dbl;
      if(strcmp(var_name,"A_SPLINE_MINLOG")==0) ccl_splines->A_SPLINE_MINLOG=var_dbl;
      if(strcmp(var_name,"A_SPLINE_MIN")==0) ccl_splines->A_SPLINE_MIN=var_dbl;
      if(strcmp(var_name,"A_SPLINE_MINLOG_PK")==0) ccl_splines->A_SPLINE_MINLOG_PK=var_dbl;
      if(strcmp(var_name,"A_SPLINE_MIN_PK")==0) ccl_splines->A_SPLINE_MIN_PK=var_dbl;
      if(strcmp(var_name,"A_SPLINE_MAX")==0) ccl_splines->A_SPLINE_MAX=var_dbl;
      if(strcmp(var_name,"LOGM_SPLINE_DELTA")==0) ccl_splines->LOGM_SPLINE_DELTA=var_dbl;
      if(strcmp(var_name,"LOGM_SPLINE_NM")==0) ccl_splines->LOGM_SPLINE_NM=(int) var_dbl;
      if(strcmp(var_name,"LOGM_SPLINE_MIN")==0) ccl_splines->LOGM_SPLINE_MIN=var_dbl;
      if(strcmp(var_name,"LOGM_SPLINE_MAX")==0) ccl_splines->LOGM_SPLINE_MAX=var_dbl;
      if(strcmp(var_name,"A_SPLINE_NA_PK")==0) ccl_splines->A_SPLINE_NA_PK=(int) var_dbl;
      if(strcmp(var_name,"A_SPLINE_NLOG_PK")==0) ccl_splines->A_SPLINE_NLOG_PK=(int) var_dbl;
      if(strcmp(var_name,"K_MAX_SPLINE")==0) ccl_splines->K_MAX_SPLINE=var_dbl;
      if(strcmp(var_name,"K_MAX")==0) ccl_splines->K_MAX=var_dbl;
      if(strcmp(var_name,"K_MIN_DEFAULT")==0) ccl_splines->K_MIN_DEFAULT=var_dbl;
      if(strcmp(var_name,"N_K")==0) ccl_splines->N_K=(int) var_dbl;
      // 3dcorr parameters
      if(strcmp(var_name,"N_K_3DCOR")==0) ccl_splines->N_K_3DCOR=(int) var_dbl;     

      // GSL parameters
      if(strcmp(var_name,"GSL_EPSREL")==0) ccl_gsl->EPSREL=var_dbl;
      if(strcmp(var_name,"GSL_N_ITERATION")==0) ccl_gsl->N_ITERATION=(size_t) var_dbl;
      if(strcmp(var_name,"GSL_INTEGRATION_GAUSS_KRONROD_POINTS")==0) ccl_gsl->INTEGRATION_GAUSS_KRONROD_POINTS=(int) var_dbl;
      if(strcmp(var_name,"GSL_INTEGRATION_EPSREL")==0) ccl_gsl->INTEGRATION_EPSREL=var_dbl;
      if(strcmp(var_name,"GSL_INTEGRATION_DISTANCE_EPSREL")==0) ccl_gsl->INTEGRATION_DISTANCE_EPSREL=var_dbl;
      if(strcmp(var_name,"GSL_INTEGRATION_DNDZ_EPSREL")==0) ccl_gsl->INTEGRATION_DNDZ_EPSREL=var_dbl;
      if(strcmp(var_name,"GSL_INTEGRATION_SIGMAR_EPSREL")==0) ccl_gsl->INTEGRATION_SIGMAR_EPSREL=var_dbl;
      if(strcmp(var_name,"GSL_INTEGRATION_NU_EPSREL")==0) ccl_gsl->INTEGRATION_NU_EPSREL=var_dbl;
      if(strcmp(var_name,"GSL_INTEGRATION_NU_EPSABS")==0) ccl_gsl->INTEGRATION_NU_EPSABS=var_dbl;
      if(strcmp(var_name,"GSL_INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS")==0) ccl_gsl->INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS=(int) var_dbl;
      if(strcmp(var_name,"GSL_INTEGRATION_LIMBER_EPSREL")==0) ccl_gsl->INTEGRATION_LIMBER_EPSREL=var_dbl;
      if(strcmp(var_name,"GSL_ROOT_EPSREL")==0) ccl_gsl->ROOT_EPSREL=var_dbl;
      if(strcmp(var_name,"GSL_ROOT_N_ITERATION")==0) ccl_gsl->ROOT_N_ITERATION=(int) var_dbl;
      if(strcmp(var_name,"GSL_ODE_GROWTH_EPSREL")==0) ccl_gsl->ODE_GROWTH_EPSREL=var_dbl;
    }
  }

  fclose(fconfig);
}


/* ------- ROUTINE: ccl_cosmology_create ------
INPUTS: ccl_parameters params
        ccl_configuration config
TASK: creates the ccl_cosmology struct and passes some values to it
DEFINITIONS:
chi: comoving distance [Mpc]
growth: growth function (density)
fgrowth: logarithmic derivative of the growth (density) (dlnD/da?)
E: E(a)=H(a)/H0 
accelerator: interpolation accelerator for functions of a
accelerator_achi: interpolation accelerator for functions of chi
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

  cosmo->data.chi = NULL;
  cosmo->data.growth = NULL;
  cosmo->data.fgrowth = NULL;
  cosmo->data.E = NULL;
  cosmo->data.accelerator=NULL;
  cosmo->data.accelerator_achi=NULL;
  cosmo->data.accelerator_m=NULL;
  cosmo->data.accelerator_d=NULL;
  cosmo->data.accelerator_k=NULL;
  cosmo->data.growth0 = 1.;
  cosmo->data.achi=NULL;

  cosmo->data.logsigma = NULL;
  cosmo->data.dlnsigma_dlogm = NULL;

  // hmf parameter for interpolation
  cosmo->data.alphahmf = NULL;
  cosmo->data.betahmf = NULL;
  cosmo->data.gammahmf = NULL;
  cosmo->data.phihmf = NULL;
  cosmo->data.etahmf = NULL;

  cosmo->data.p_lin = NULL;
  cosmo->data.p_nl = NULL;
  //cosmo->data.nu_pspace_int = NULL;
  cosmo->computed_distances = false;
  cosmo->computed_growth = false;
  cosmo->computed_power = false;
  cosmo->computed_sigma = false;
  cosmo->computed_hmfparams = false;
  cosmo->status = 0;
  strcpy(cosmo->status_message, "");
  
  return cosmo;
}


/* ------- ROUTINE: ccl_cosmology_create_with_params ------
INPUTS: 
        Numbers for the basic cosmological parameters needed by CCL
        ccl_configuration config
TASK: Creates ccl_cosmology struct directly from a set of input cosmological 
      parameter values, without the need to create a separate ccl_parameters 
      struct.
DEFINITIONS:
Omega_c: cold dark matter
Omega_b: baryons
Omega_k: curvature
Neff: effective number of neutrinos species
mnu: neutrino mass(es)
mnu_type: how the neutrino mass(es) should be treated
w0: Dark energy eqn. of state parameter
wa: Dark energy eqn. of state parameter, time variation
h: Hubble's constant divided by (100 km/s/Mpc).
norm_pk: amplitude of the primordial PS (either A_s or sigma8)
n_s: index of the primordial PS
*/
ccl_cosmology * ccl_cosmology_create_with_params(double Omega_c, double Omega_b, double Omega_k,
						 double Neff, double* mnu, ccl_mnu_convention mnu_type,
						 double w0, double wa, double h, double norm_pk, double n_s,
						 double bcm_log10Mc, double bcm_etab, double bcm_ks,
						 int nz_mgrowth, double *zarr_mgrowth, 
						 double *dfarr_mgrowth, ccl_configuration config,
						 int *status)
{

  // Create ccl_parameters struct from input parameters
  ccl_parameters params;
  
  params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Neff, mnu, mnu_type, w0, wa,
				 h, norm_pk, n_s, bcm_log10Mc, bcm_etab, bcm_ks, nz_mgrowth, zarr_mgrowth, dfarr_mgrowth, status);
  // Check status
  ccl_check_status_nocosmo(status);
  
  // Create  ccl_cosmology struct
  ccl_cosmology *cosmo;
  cosmo = ccl_cosmology_create(params, config);

  return cosmo;
}

/* ------- ROUTINE: ccl_cosmology_create_with_lcdm_params ------
INPUTS: 
        Numbers for the basic LCDM cosmological parameters needed by CCL
        ccl_configuration config
TASK: Creates ccl_cosmology struct directly from a set of input cosmological 
      parameter values (for a flat LCDM model), without the need to create a 
      separate ccl_parameters struct.
DEFINITIONS:
Omega_c: cold dark matter
Omega_b: baryons
Omega_k: curvature
h: Hubble's constant divided by (100 km/s/Mpc).
norm_pk: amplitude of the primordial PS (either A_s or sigma8)
n_s: index of the primordial PS
*/
ccl_cosmology * ccl_cosmology_create_with_lcdm_params(double Omega_c, double Omega_b, double Omega_k,
						      double h, double norm_pk, double n_s,
						      ccl_configuration config, int *status)
{
  // Create ccl_parameters struct from input parameters
  ccl_parameters params;
  params = ccl_parameters_create_lcdm(Omega_c, Omega_b, Omega_k, h, norm_pk, n_s, status);
  // Check status
  ccl_check_status_nocosmo(status);
  
  // Create  ccl_cosmology struct
  ccl_cosmology *cosmo;
  cosmo = ccl_cosmology_create(params, config);
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
  params->T_CMB =  TCMB;
  // kg / m^3
  double rho_g = 4. * STBOLTZ / pow(CLIGHT, 3) * pow(params->T_CMB, 4);
  // kg / m^3
  double rho_crit = RHO_CRITICAL * SOLAR_MASS/pow(MPC_TO_METER, 3) * pow(params->h, 2);
  params->Omega_g = rho_g/rho_crit;
  
  // Get the N_nu_rel from Neff and N_nu_mass
  params->N_nu_rel = params->Neff - params->N_nu_mass * pow(TNCDM, 4) / pow(4./11.,4./3.);
  
  // Temperature of the relativistic neutrinos in K
  double T_nu= (params->T_CMB) * pow(4./11.,1./3.); 
  // in kg / m^3
  double rho_nu_rel = params->N_nu_rel* 7.0/8.0 * 4. * STBOLTZ / pow(CLIGHT, 3) * pow(T_nu, 4);
  params-> Omega_n_rel = rho_nu_rel/rho_crit;
    
  // If non-relativistic neutrinos are present, calculate the phase_space integral.
  if((params->N_nu_mass)>0) {
    // Pass NULL for the accelerator here because we don't have our cosmology object defined yet.
    params->Omega_n_mass = ccl_Omeganuh2(1.0, params->N_nu_mass, params->mnu, params->T_CMB, NULL, status) / ((params->h)*(params->h));
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
  params->sqrtk=sqrt(fabs(params->Omega_k))*params->h/CLIGHT_HMPC;
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
				     double bcm_ks, int nz_mgrowth, double *zarr_mgrowth,
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
  
  // Set the sum of neutrino masses
  params.sum_nu_masses = *mnu;
  double mnusum = *mnu;
  double *mnu_in = NULL;
  
  /* Check whether ccl_splines and ccl_gsl exist. If either is not set yet, load
     parameters from the config file. */
  if(ccl_splines==NULL || ccl_gsl==NULL) {
    ccl_cosmology_read_config();
  }
  /* Exit gracefully if config file can't be opened. */
  if(ccl_splines==NULL || ccl_gsl==NULL) {
    ccl_raise_exception(CCL_ERROR_MISSING_CONFIG_FILE, "ccl_core.c: Failed to read config file.");
    *status = CCL_ERROR_MISSING_CONFIG_FILE;
    return params;
  }
  
  // Decide how to split sum of neutrino masses between 3 neutrinos. See the 
  // CCL note for how we get these expressions for the neutrino masses in 
  // normal and inverted hierarchy.
  if (mnu_type==ccl_mnu_sum){
	  // Normal hierarchy
	  mnu_in = malloc(3*sizeof(double));
	  double nfac = -6.*DELTAM12_sq + 12.*DELTAM13_sq_pos + 4.*mnusum*mnusum;
	  
	  mnu_in[0] = 2./3. * mnusum - 1./6. * pow(nfac, 0.5) 
	            - 0.25 * DELTAM12_sq / (2./3.* mnusum - 1./6.*pow(nfac, 0.5));
	  mnu_in[1] = 2./3.* mnusum - 1./6. * pow(nfac, 0.5) 
	            + 0.25 * DELTAM12_sq / (2./3.* mnusum - 1./6. * pow(nfac, 0.5));
	  mnu_in[2] = -1./3. * mnusum + 1./3 * pow(nfac, 0.5); 
	  
	  if (mnu_in[0]<0 || mnu_in[1]<0 || mnu_in[2]<0){
	    // The user has provided a sum that is below the physical limit.
	    if (params.sum_nu_masses < 1e-14){
			mnu_in[0] = 0.; mnu_in[1] = 0.; mnu_in[2] = 0.;
		}else{
			*status = CCL_ERROR_MNU_UNPHYSICAL;
	    }
	  }
  } else if (mnu_type==ccl_mnu_sum_inverted){
		// Inverted hierarchy
		mnu_in = malloc(3*sizeof(double));
		double nfac = -6.*DELTAM12_sq + 12.*DELTAM13_sq_neg + 4.*mnusum*mnusum;
		
		mnu_in[0] = 2./3.* mnusum - 1./6.*pow(nfac, 0.5) 
	              - 0.25 * DELTAM12_sq / (2./3.* mnusum - 1./6.*pow(nfac, 0.5));
	    mnu_in[1] = 2./3.* mnusum - 1./6. * pow(nfac, 0.5) 
	              + 0.25 * DELTAM12_sq / (2./3.* mnusum - 1./6. * pow(nfac, 0.5));
	    mnu_in[2] = -1./3. * mnusum + 1./3 * pow(nfac, 0.5);
	    
	    if(mnu_in[0]<0 || mnu_in[1]<0 || mnu_in[2]<0){
	    // The user has provided a sum that is below the physical limit.
	    if (params.sum_nu_masses < 1e-14){
			mnu_in[0] = 0.; mnu_in[1] = 0.; mnu_in[2] = 0.;
		}else{
			*status = CCL_ERROR_MNU_UNPHYSICAL;
	    }
	    }
  } else if (mnu_type==ccl_mnu_sum_equal){
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
  double mnuval = 0.;
  mnu = &mnuval;
  ccl_mnu_convention mnu_type = ccl_mnu_sum;
  
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Neff,
						mnu, mnu_type, w0, wa, h, norm_pk, n_s, -1, -1, -1, -1, NULL, NULL, status);
			
  return params;

}



/* ------- ROUTINE: ccl_parameters_create_flat_lcdm -------- 
INPUT: some cosmological parameters needed to create a flat LCDM model 
TASK: call ccl_parameters_create to produce an LCDM model with baryonic effects
*/
ccl_parameters ccl_parameters_create_flat_lcdm_bar(double Omega_c, double Omega_b, double h,
						   double norm_pk, double n_s, double bcm_log10Mc,
						   double bcm_etab, double bcm_ks, int *status)
{
  double Omega_k = 0.0;
  double Neff = 3.046;
  double *mnu;
  double mnuval = 0.;
  mnu = &mnuval;
  ccl_mnu_convention mnu_type = ccl_mnu_sum;
  double w0 = -1.0;
  double wa = 0.0;
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Neff,
						mnu, mnu_type, w0, wa, h, norm_pk, n_s, bcm_log10Mc, bcm_etab,
						bcm_ks, -1, NULL, NULL, status);
  return params;

}

/* ------- ROUTINE: ccl_parameters_create_flat_lcdm_nu -------- 
INPUT: some cosmological parameters needed to create a flat LCDM model with neutrinos 
TASK: call ccl_parameters_create to produce an LCDM model
*/
ccl_parameters ccl_parameters_create_flat_lcdm_nu(double Omega_c, double Omega_b, double h, double norm_pk,
						  double n_s, double Neff, double *mnu, ccl_mnu_convention mnu_type,
						  int *status)
{
  double Omega_k = 0.0;
  double w0 = -1.0;
  double wa = 0.0;
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Neff, mnu, mnu_type, w0, wa,
						h, norm_pk, n_s, -1, -1, -1, -1, NULL, NULL, status);
  return params;

}


/* ------- ROUTINE: ccl_parameters_create_lcdm -------- 
INPUT: some cosmological parameters needed to create an LCDM model with curvature 
TASK: call ccl_parameters_create for this specific model
*/
ccl_parameters ccl_parameters_create_lcdm(double Omega_c, double Omega_b, double Omega_k, double h,
					  double norm_pk, double n_s, int *status)
{
  double Neff = 3.046;
  double w0 = -1.0;
  double wa = 0.0;
  double *mnu;
  double mnuval = 0.;
  mnu = &mnuval;
  ccl_mnu_convention mnu_type = ccl_mnu_sum;
  
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Neff, mnu, mnu_type, w0, wa,
						h, norm_pk, n_s, -1, -1, -1,-1,NULL,NULL, status);
  return params;
}


/* ------- ROUTINE: ccl_parameters_create_lcdm_nu -------- 
INPUT: some cosmological parameters needed to create an LCDM model with curvature and neutrinos
TASK: call ccl_parameters_create for this specific model
*/
ccl_parameters ccl_parameters_create_lcdm_nu(double Omega_c, double Omega_b, double Omega_k, double h,
					     double norm_pk, double n_s, double Neff,
					     double* mnu, ccl_mnu_convention mnu_type, int *status)
{
  double w0 = -1.0;
  double wa = 0.0; 

  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Neff, mnu, mnu_type, w0, wa,
						h, norm_pk, n_s, -1, -1, -1,-1,NULL,NULL, status);

  return params;

}

/* ------- ROUTINE: ccl_parameters_create_flat_wcdm -------- 
INPUT: some cosmological parameters needed to create an LCDM model with wa=0 but w0!=-1
TASK: call ccl_parameters_create for this specific model
*/
ccl_parameters ccl_parameters_create_flat_wcdm(double Omega_c, double Omega_b, double w0, double h,
					       double norm_pk, double n_s, int *status)
{

  double Omega_k = 0.0;
  double Neff = 3.046;
  double wa = 0.0;
  double *mnu;
  double mnuval = 0.;
  mnu = &mnuval;
  ccl_mnu_convention mnu_type = ccl_mnu_sum;

  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Neff, mnu, mnu_type, w0, wa,
						h, norm_pk, n_s, -1, -1, -1,-1,NULL,NULL, status);
  return params;
}


/* ------- ROUTINE: ccl_parameters_create_wcdm_nu -------- 
INPUT: some cosmological parameters needed to create an LCDM model with neutrinos, and wa=0 but w0!=-1
TASK: call ccl_parameters_create for this specific model
*/
ccl_parameters ccl_parameters_create_flat_wcdm_nu(double Omega_c, double Omega_b, double w0, double h,
						  double norm_pk, double n_s, double Neff, double *mnu, ccl_mnu_convention mnu_type, int *status)
{

  double Omega_k = 0.0;
  double wa = 0.0;
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Neff, mnu, mnu_type, w0, wa, 
						h, norm_pk, n_s, -1, -1, -1,-1,NULL,NULL, status);
  return params;
}


/* ------- ROUTINE: ccl_parameters_create_wacdm -------- 
INPUT: some cosmological parameters needed to create an LCDM model with curvature wa!=0 and and w0!=-1
TASK: call ccl_parameters_create for this specific model
*/
ccl_parameters ccl_parameters_create_flat_wacdm(double Omega_c, double Omega_b, double w0, double wa,
						double h, double norm_pk, double n_s, int *status)
{
  double Omega_k = 0.0;
  double Neff = 3.046;
  double *mnu;
  double mnuval = 0.;
  mnu = &mnuval;
  ccl_mnu_convention mnu_type = ccl_mnu_sum;
  
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k,Neff, mnu, mnu_type, w0, wa,
						h, norm_pk, n_s, -1, -1, -1,-1,NULL,NULL, status);
  return params;
}


/* ------- ROUTINE: ccl_parameters_create_wacdm_nu -------- 
INPUT: some cosmological parameters needed to create an LCDM model with neutrinoswith curvature wa!=0 and and w0!=-1
TASK: call ccl_parameters_create for this specific model
*/
ccl_parameters ccl_parameters_create_flat_wacdm_nu(double Omega_c, double Omega_b, double w0, double wa,
						   double h, double norm_pk, double n_s,
						   double Neff, double* mnu, ccl_mnu_convention mnu_type, int *status)
{

  double Omega_k = 0.0;
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k,Neff, mnu, mnu_type, w0, wa,
						h, norm_pk, n_s, -1, -1, -1,-1,NULL,NULL, status);
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
  if(data->chi!=NULL)
    gsl_spline_free(data->chi);
  if(data->growth!=NULL)
    gsl_spline_free(data->growth);
  if(data->fgrowth!=NULL)
    gsl_spline_free(data->fgrowth);
  if(data->accelerator!=NULL)
    gsl_interp_accel_free(data->accelerator);
  if(data->accelerator_achi!=NULL)
    gsl_interp_accel_free(data->accelerator_achi);
  if(data->E!=NULL)
    gsl_spline_free(data->E);
  if(data->achi!=NULL)
    gsl_spline_free(data->achi);
  if(data->logsigma!=NULL)
    gsl_spline_free(data->logsigma);
  if(data->dlnsigma_dlogm!=NULL)
    gsl_spline_free(data->dlnsigma_dlogm);
  if(data->p_lin!=NULL)
    gsl_spline2d_free(data->p_lin);
  if(data->p_nl!=NULL)
    gsl_spline2d_free(data->p_nl);
  if(data->alphahmf!=NULL)
    gsl_spline_free(data->alphahmf);
  if(data->betahmf!=NULL)
    gsl_spline_free(data->betahmf);
  if(data->gammahmf!=NULL)
    gsl_spline_free(data->gammahmf);
  if(data->phihmf!=NULL)
    gsl_spline_free(data->phihmf);
  if(data->etahmf!=NULL)
    gsl_spline_free(data->etahmf);
  if(data->accelerator_d!=NULL)
    gsl_interp_accel_free(data->accelerator_d);
  if(data->accelerator_m!=NULL)
    gsl_interp_accel_free(data->accelerator_m);
  if(data->accelerator_k!=NULL)
    gsl_interp_accel_free(data->accelerator_k);
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
