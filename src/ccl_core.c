#include "ccl_core.h"
#include "ccl_utils.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "gsl/gsl_errno.h"
#include "gsl/gsl_odeiv.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_integration.h"
#include "ccl_params.h"
#include <stdlib.h>

const ccl_configuration default_config = {ccl_boltzmann_class, ccl_halofit, ccl_tinker10};

/* ------- ROUTINE: ccl_cosmology_read_config ------
   INPUTS: none, but will look for ini file in include/ dir
   TASK: fill out global variables of splines with user defined input.
   The variables are defined in ccl_params.h.
   
   The following are the relevant global variables:
*/

ccl_spline_params * ccl_splines; // Global variable

void ccl_cosmology_read_config(){

  int CONFIG_LINE_BUFFER_SIZE=100;
  int MAX_CONFIG_VAR_LEN=100;
  FILE *fconfig;
  char buf[CONFIG_LINE_BUFFER_SIZE];
  char var_name[MAX_CONFIG_VAR_LEN];
  double var_dbl;
  
  ccl_splines = malloc(sizeof(ccl_spline_params));
  
  // Get parameter .ini filename from environment variable or default location
  const char* param_file;
  const char* param_file_env = getenv("CCL_PARAM_FILE");
  if (param_file_env != NULL){
    param_file = param_file_env;
  } else {
    // Use default ini file
    param_file = EXPAND_STR(__CCL_DATA_DIR__) "/ccl_params.ini";
  }
  
  if ((fconfig=fopen(param_file, "r")) == NULL) {
    fprintf(stderr, "ccl_core.c: Failed to open config file %s\n", param_file);
    exit(EXIT_FAILURE);
  } 

  while(! feof(fconfig)) {
  fgets(buf, CONFIG_LINE_BUFFER_SIZE, fconfig);
  if (buf[0]==';' || buf[0]=='[' || buf[0]=='\n') {
    continue;
  } else {
    sscanf(buf, "%99[^=]=%le\n",var_name, &var_dbl);
    if(strcmp(var_name,"A_SPLINE_DELTA")==0) ccl_splines->A_SPLINE_DELTA=var_dbl;
    if(strcmp(var_name,"A_SPLINE_NA")==0) ccl_splines->A_SPLINE_NA=(int) var_dbl;
    if(strcmp(var_name,"A_SPLINE_MIN")==0) ccl_splines->A_SPLINE_MIN=var_dbl;
    if(strcmp(var_name,"A_SPLINE_MAX")==0) ccl_splines->A_SPLINE_MAX=var_dbl;
    if(strcmp(var_name,"LOGM_SPLINE_DELTA")==0) ccl_splines->LOGM_SPLINE_DELTA=var_dbl;
    if(strcmp(var_name,"LOGM_SPLINE_NM")==0) ccl_splines->LOGM_SPLINE_NM=(int) var_dbl;
    if(strcmp(var_name,"LOGM_SPLINE_MIN")==0) ccl_splines->LOGM_SPLINE_MIN=var_dbl;
    if(strcmp(var_name,"LOGM_SPLINE_MAX")==0) ccl_splines->LOGM_SPLINE_MAX=var_dbl;
    if(strcmp(var_name,"N_A")==0) ccl_splines->N_A=(int) var_dbl;
    if(strcmp(var_name,"K_MAX_SPLINE")==0) ccl_splines->K_MAX_SPLINE=var_dbl;
    if(strcmp(var_name,"K_MAX")==0) ccl_splines->K_MAX=var_dbl;
    if(strcmp(var_name,"K_MIN_DEFAULT")==0) ccl_splines->K_MIN_DEFAULT=var_dbl;
    if(strcmp(var_name,"N_K")==0) ccl_splines->N_K=(int) var_dbl;     
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
  
  if(ccl_splines==NULL) ccl_cosmology_read_config();

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
  cosmo->data.growth0 = 1.;
  cosmo->data.achi=NULL;

  cosmo->data.logsigma = NULL;

  // hmf parameter for interpolation
  cosmo->data.alphahmf = NULL;
  cosmo->data.betahmf = NULL;
  cosmo->data.gammahmf = NULL;
  cosmo->data.phihmf = NULL;
  cosmo->data.etahmf = NULL;

  cosmo->data.p_lin = NULL;
  cosmo->data.p_nl = NULL;
  
  cosmo->computed_distances = false;
  cosmo->computed_growth = false;
  cosmo->computed_power = false;
  cosmo->computed_sigma = false;
  cosmo->computed_hmfparams = false;
  cosmo->status = 0;
  
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
Omega_n: neutrinos
Omega_k: curvature
w0: Dark energy eqn. of state parameter
wa: Dark energy eqn. of state parameter, time variation
h: Hubble's constant divided by (100 km/s/Mpc).
norm_pk: amplitude of the primordial PS (either A_s or sigma_8)
n_s: index of the primordial PS
*/

ccl_cosmology * ccl_cosmology_create_with_params(
        double Omega_c, double Omega_b, double Omega_k, double Omega_n, 
        double w0, double wa, double h, double norm_pk, double n_s,
        int nz_mgrowth, double *zarr_mgrowth, double *dfarr_mgrowth, 
        ccl_configuration config)
{
    // Create ccl_parameters struct from input parameters
    ccl_parameters params;
    params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Omega_n, w0, wa, 
                                   h, norm_pk, n_s, nz_mgrowth, zarr_mgrowth, 
                                   dfarr_mgrowth);
    
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
norm_pk: amplitude of the primordial PS (either A_s or sigma_8)
n_s: index of the primordial PS
*/

ccl_cosmology * ccl_cosmology_create_with_lcdm_params(
        double Omega_c, double Omega_b, double Omega_k, double h, 
        double norm_pk, double n_s,
        ccl_configuration config)
{
    // Create ccl_parameters struct from input parameters
    ccl_parameters params;
    params = ccl_parameters_create_lcdm(Omega_c, Omega_b, Omega_k, h, 
                                        norm_pk, n_s);
    
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
sigma_8: variance in 8 Mpc/h spheres for normalization of matter PS, enforced here to initially set to NaN
z_star: recombination redshift
 */

void ccl_parameters_fill_initial(ccl_parameters *params)
{
  // Fixed radiation parameters
  // Omega_g * h**2 is known from T_CMB
  params->T_CMB =  2.725;
  params->Omega_g = M_PI*M_PI*pow((params->T_CMB/11604.5),4.)/(15*8.098E-11*params->h*params->h);

  // Derived parameters
  params->Omega_l = 1.0 - params->Omega_m - params->Omega_g - params->Omega_n - params->Omega_k;
  // Initially undetermined parameters - set to nan to trigger
  // problems if they are mistakenly used.
  if (isfinite(params->A_s)){params->sigma_8 = NAN;}
  if (isfinite(params->sigma_8)){params->A_s = NAN;}
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
Omega_n: neutrinos
Omega_k: curvature
little omega_x means Omega_x*h^2
w0: Dark energy eq of state parameter
wa: Dark energy eq of state parameter, time variation
H0: Hubble's constant in km/s/Mpc.
h: Hubble's constant divided by (100 km/s/Mpc).
A_s: amplitude of the primordial PS
n_s: index of the primordial PS
 */

ccl_parameters ccl_parameters_create(double Omega_c, double Omega_b, double Omega_k, double Omega_n, double w0, double wa, double h, double norm_pk, double n_s,int nz_mgrowth,double *zarr_mgrowth,double *dfarr_mgrowth){
  ccl_parameters params;
  params.sigma_8 = NAN;
  params.A_s = NAN;
  params.Omega_c = Omega_c;
  params.Omega_b = Omega_b;
  params.Omega_m = Omega_b + Omega_c;
  params.Omega_n = Omega_n;
  params.Omega_k = Omega_k;

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
    params.sigma_8=norm_pk;
  params.n_s = n_s;

  // Set remaining standard and easily derived parameters
  ccl_parameters_fill_initial(&params);

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

ccl_parameters ccl_parameters_create_flat_lcdm(double Omega_c, double Omega_b, double h, double norm_pk, double n_s)
{
  double Omega_k = 0.0;
  double Omega_n = 0.0;
  double w0 = -1.0;
  double wa = 0.0;
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Omega_n, w0, wa, h, norm_pk, n_s, -1, NULL, NULL);
  return params;

}


/* ------- ROUTINE: ccl_parameters_create_lcdm -------- 
INPUT: some cosmological parameters needed to create an LCDM model with curvature 
TASK: call ccl_parameters_create for this specific model
*/

ccl_parameters ccl_parameters_create_lcdm(double Omega_c, double Omega_b, double Omega_k, double h, double norm_pk, double n_s)
{
  double Omega_n = 0.0;
  double w0 = -1.0;
  double wa = 0.0;
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Omega_n, w0, wa, h, norm_pk, n_s,-1,NULL,NULL);
  return params;

}


/* ------- ROUTINE: ccl_parameters_create_wcdm -------- 
INPUT: some cosmological parameters needed to create an LCDM model with curvature and wa=0 but w0!=-1
TASK: call ccl_parameters_create for this specific model
*/


ccl_parameters ccl_parameters_create_flat_wcdm(double Omega_c, double Omega_b, double w0, double h, double norm_pk, double n_s)
{

  double Omega_k = 0.0;
  double Omega_n = 0.0;
  double wa = 0.0;
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Omega_n, w0, wa, h, norm_pk, n_s,-1,NULL,NULL);
  return params;
}


/* ------- ROUTINE: ccl_parameters_create_wacdm -------- 
INPUT: some cosmological parameters needed to create an LCDM model with curvature wa!=0 and and w0!=-1
TASK: call ccl_parameters_create for this specific model
*/

ccl_parameters ccl_parameters_create_flat_wacdm(double Omega_c, double Omega_b, double w0, double wa, double h, double norm_pk, double n_s)
{

  double Omega_k = 0.0;
  double Omega_n = 0.0;
  ccl_parameters params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Omega_n, w0, wa, h, norm_pk, n_s,-1,NULL,NULL);
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
