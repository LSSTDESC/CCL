#include "ccl_core.h"
#include "ccl_utils.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_interp.h"
#include "gsl/gsl_spline.h"
//#include "gsl/gsl_interp2d.h"
//#include "gsl/gsl_spline2d.h"
#include "ccl_placeholder.h"
#include "ccl_background.h"
#include "ccl_power.h"
#include "ccl_error.h"
#include "../class/include/class.h"

/*------ ROUTINE: ccl_cosmology_compute_power_class ----- 
INPUT: ccl_cosmology * cosmo
TASK: call CLASS to obtain power spectra
*/

void ccl_cosmology_compute_power_class(ccl_cosmology * cosmo){
  
  struct precision pr;        // for precision parameters 
  struct background ba;       // for cosmological background 
  struct thermo th;           // for thermodynamics 
  struct perturbs pt;         // for source functions 
  struct transfers tr;        // for transfer functions 
  struct primordial pm;       // for primordial spectra 
  struct spectra sp;          // for output spectra 
  struct nonlinear nl;        // for non-linear spectra 
  struct lensing le;
  struct output op;
  ErrorMsg errmsg;            // for error messages 
  struct file_content fc;
  // generate file_content structure 
  // CLASS configuration parameters will be passed through this structure,
  // to avoid writing and reading .ini files for every call
  if (parser_init(&fc,15,"none",errmsg) == _FAILURE_){
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): parser init error:%s\n",errmsg);
    return;
  }
  // basic CLASS configuration parameters
  // these need to be decided once, and they unchanged for (most) CLASS calls from CCL
  strcpy(fc.name[0],"output");
  strcpy(fc.value[0],"mPk");

  strcpy(fc.name[1],"non linear");
  if (cosmo->config.matter_power_spectrum_method == ccl_halofit){ strcpy(fc.value[1],"Halofit"); }
  else {strcpy(fc.value[1]," ");}

  strcpy(fc.name[2],"P_k_max_1/Mpc");
  sprintf(fc.value[2],"%e",K_MAX); //in units of 1/Mpc, corroborated with ccl_constants.h

  strcpy(fc.name[3],"z_max_pk");
  sprintf(fc.value[3],"%e",1./A_SPLINE_MIN-1.);

  strcpy(fc.name[4],"modes");
  strcpy(fc.value[4],"s");

  strcpy(fc.name[5],"lensing");
  strcpy(fc.value[5],"no");

  // now, copy over cosmology parameters
  strcpy(fc.name[6],"h");
  sprintf(fc.value[6],"%e",cosmo->params.h);

  strcpy(fc.name[7],"Omega_cdm");
  sprintf(fc.value[7],"%e",cosmo->params.Omega_c);

  strcpy(fc.name[8],"Omega_b");
  sprintf(fc.value[8],"%e",cosmo->params.Omega_b);

  strcpy(fc.name[9],"Omega_k");
  sprintf(fc.value[9],"%e",cosmo->params.Omega_k);

  strcpy(fc.name[10],"n_s");
  sprintf(fc.value[10],"%e",cosmo->params.n_s);

  if (isfinite(cosmo->params.sigma_8) && isfinite(cosmo->params.A_s)){
      cosmo->status = 11;
      strcpy(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error initialzing CLASS pararmeters: both sigma_8 and A_s defined\n");
    return;
  }
  if (isfinite(cosmo->params.sigma_8)){
    strcpy(fc.name[11],"sigma_8");
    sprintf(fc.value[11],"%e",cosmo->params.sigma_8);
  }
  else if (isfinite(cosmo->params.A_s)){ 
    strcpy(fc.name[11],"A_s");
    sprintf(fc.value[11],"%e",cosmo->params.A_s);
  }
  else{
      cosmo->status = 11;
      strcpy(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error initialzing CLASS pararmeters: neither sigma_8 nor A_s defined\n");
    return;
  }

//cosmological constant?
// set Omega_Lambda = 0.0 if w !=-1
  if ((cosmo->params.w0 !=-1.0) || (cosmo->params.wa !=0)){
    strcpy(fc.name[12],"Omega_Lambda");
    sprintf(fc.value[12],"%e",0.0);

    strcpy(fc.name[13],"w0_fld");
    sprintf(fc.value[13],"%e",cosmo->params.w0);

    strcpy(fc.name[14],"wa_fld");
    sprintf(fc.value[14],"%e",cosmo->params.wa);
  }


  if (input_init(&fc,&pr,&ba,&th,&pt,&tr,&pm,&sp,&nl,&le,&op,errmsg) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS input:%s\n",errmsg);
    return;
  }

  if (background_init(&pr,&ba) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS background:%s\n",errmsg);
    return;
  }

  if (thermodynamics_init(&pr,&ba,&th) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS thermodynamics:%s\n",errmsg);
    return;
  }

  if (perturb_init(&pr,&ba,&th,&pt) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS pertub:%s\n",errmsg);
    return;
  }

  if (primordial_init(&pr,&pt,&pm) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS primordial:%s\n",errmsg);
    return;
  }

  if (nonlinear_init(&pr,&ba,&th,&pt,&pm,&nl) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS nonlinear:%s\n",errmsg);
    return;
  }
  if (transfer_init(&pr,&ba,&th,&pt,&nl,&tr) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS transfer:%s\n",errmsg);
    return;
  }

  if (spectra_init(&pr,&ba,&pt,&pm,&nl,&tr,&sp) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS spectra:%s\n",errmsg);
    return;
  }

  //CLASS calculations done - now allocate CCL splines
  double kmin = K_MIN;
  double kmax = K_MAX;
  int nk = N_K;
  double amin = A_SPLINE_MIN;
  double amax = A_SPLINE_MAX;
  int na = N_A;
  
  // The x array is initially k, but will later
  // be overwritten with log(k)
  double * x = ccl_log_spacing(kmin, kmax, nk);
  double * y = malloc(sizeof(double)*nk);
  double * z = ccl_linear_spacing(amin,amax, na);
  double * y2d = malloc(nk * na * sizeof(double));
  if (z==NULL||y==NULL|| x==NULL || y2d==0){
    cosmo->status = 4;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_class(): memory allocation error\n");
  }
  else{  
    // After this loop x will contain log(k), y will contain log(P_nl), z will contain log(P_lin)
    // all in Mpc, not Mpc/h units!
    double Z, ic;
    int s;
    for (int i=0; i<nk; i++){
      s =spectra_pk_at_k_and_z(&ba, &pm, &sp,x[i],0.0, &Z,&ic);
      y[i] = log(Z);
      x[i] = log(x[i]);
    }
  
    gsl_spline * log_power_lin = gsl_spline_alloc(K_SPLINE_TYPE, nk);
    int status = gsl_spline_init(log_power_lin, x, y, nk);
    if (status){
      gsl_spline_free(log_power_lin);
      cosmo->status = 4;
      strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_class(): Error creating log_power_lin spline\n");
      return;
    }
    else
      cosmo->data.p_lin = log_power_lin;
 

  
    gsl_spline2d * log_power_nl = gsl_spline2d_alloc(PNL_SPLINE_TYPE, nk,na);
    for (int j = 0; j < na; j++){
      for (int i=0; i<nk; i++){
      //The 2D interpolation routines access the function values y_{k_ia_j} with the following ordering:
      //y_ij = y2d[j*N_k + i]
      //with i = 0,...,N_k-1 and j = 0,...,N_a-1.
          s = spectra_pk_nl_at_k_and_z(&ba, &pm, &sp,exp(x[i]),1./z[j]-1., &Z);
          y2d[j*nk+i] = log(Z);
      }
    }

    status = gsl_spline2d_init(log_power_nl, x, z, y2d,nk,na);
    if (status){
      free(x);
      free(y);
      free(z);
      gsl_spline2d_free(log_power_nl);
      cosmo->status = 4;
      strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_class(): Error creating log_power_nl spline\n");
      return;
    }
    else
      cosmo->data.p_nl = log_power_nl;
  
    free(x);
    free(y);
    free(z);
  }
  if (spectra_free(&sp) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error freeing CLASS spectra:%s\n",sp.error_message);
    return;
  }
  
  if (transfer_free(&tr) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error freeing CLASS transfer:%s\n",tr.error_message);
    return;
  }
  if (nonlinear_free(&nl) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error freeing CLASS nonlinear:%s\n",nl.error_message);
    return;
  }
  
  if (primordial_free(&pm) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error freeing CLASS pm:%s\n",pm.error_message);
    return;
  }
  
  if (perturb_free(&pt) == _FAILURE_) {
      cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error freeing CLASS pt:%s\n",pt.error_message);
    return;
  }
  
  if (thermodynamics_free(&th) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error freeing CLASS thermo:%s\n",th.error_message);
    return;
  }

  if (background_free(&ba) == _FAILURE_) {
    cosmo->status = 12;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error freeing CLASS bg:%s\n",ba.error_message);
    return;
  }
  if (parser_free(&fc)== _FAILURE_) {
    cosmo->status = 12;
    strcpy(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error freeing CLASS parser\n");
    return;
  }
}

/*------ ROUTINE: tsqr_BBKS ----- 
INPUT: ccl_parameters and k wavenumber in 1/Mpc
TASK: provide the square of the BBKS transfer function with baryonic correction
*/

static double tsqr_BBKS(ccl_parameters * params, double k)
{
  double q = k/(params->Omega_m*params->h*params->h*exp(-params->Omega_b*(1.0+pow(2.*params->h,.5)/params->Omega_m)));
  return pow(log(1.+2.34*q)/(2.34*q),2.0)/pow(1.+3.89*q+pow(16.1*q,2.0)+pow(5.46*q,3.0)+pow(6.71*q,4.0),0.5);
}


/*------ ROUTINE: bbks_power ----- 
INPUT: ccl_parameters and k wavenumber in 1/Mpc
TASK: provide the BBKS power spectrum with baryonic correction at single k
*/

//Calculate Normalization see Cosmology Notes 8.105 (TODO: whose comment is this?)
static double bbks_power(ccl_parameters * params, double k){
  return pow(k,params->n_s)*tsqr_BBKS(params, k);
}


/*------ ROUTINE: ccl_cosmology_compute_bbks_power ----- 
INPUT: cosmology
TASK: provide spline for the BBKS power spectrum with baryonic correction
*/

void ccl_cosmology_compute_power_bbks(ccl_cosmology * cosmo){

  double kmin = K_MIN;
  double kmax = K_MAX;
  int nk = N_K;
  double amin = A_SPLINE_MIN;
  double amax = A_SPLINE_MAX;
  int na = N_A;
  
  // The x array is initially k, but will later
  // be overwritten with log(k)
  double * x = ccl_log_spacing(kmin, kmax, nk);
  double * y = malloc(sizeof(double)*nk);
  double * z = ccl_linear_spacing(amin,amax, na);
  double * y2d = malloc(nk * na * sizeof(double));
  if (z==NULL||y==NULL|| x==NULL || y2d==0){
    cosmo->status = 4;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_bbks(): memory allocation error\n");
    return;
  }

    // After this loop x will contain log(k)
    for (int i=0; i<nk; i++){
        y[i] = log(bbks_power(&cosmo->params, x[i]));
        x[i] = log(x[i]);
    }

    // now normalize to cosmo->params.sigma_8
    if (isnan(cosmo->params.sigma_8)){
        free(x);
        free(y);
        free(z);
        free(y2d);
        cosmo->status = 11;
        strcpy(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_bbks(): sigma_8 not set, required for BBKS\n");
        return;
    }

    gsl_spline * log_power_lin = gsl_spline_alloc(K_SPLINE_TYPE, nk);
    int status = gsl_spline_init(log_power_lin, x, y, nk);
    if (status){
      free(x);
      free(y);
      free(z);
      free(y2d);
      gsl_spline_free(log_power_lin);
      cosmo->status = 4;
      strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_bbks(): Error creating log_power_lin spline\n");
      return;
    }
    cosmo->data.p_lin=log_power_lin;

    cosmo->computed_power=true;
    double sigma_8 = ccl_sigma8(cosmo);
    cosmo->computed_power=false;
    if (status){
      free(x);
      free(y);
      free(z);
      free(y2d);
      gsl_spline_free(log_power_lin);
      cosmo->status = 10;
      strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_bbks(): error calling ccl_sigma8()\n");
      return;
    }
    double log_sigma_8 = 2*(log(cosmo->params.sigma_8) - log(sigma_8));
    for (int i=0; i<nk; i++){
        y[i] += log_sigma_8;
    }

    gsl_spline_free(log_power_lin);
    log_power_lin = gsl_spline_alloc(K_SPLINE_TYPE, nk);
    status = gsl_spline_init(log_power_lin, x, y, nk);    
    if (status){
      free(x);
      free(y);
      gsl_spline_free(log_power_lin);
      cosmo->status = 4;
      strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_bbks(): Error creating log_power_lin spline\n");
    }
    else
      cosmo->data.p_lin = log_power_lin;


    if (cosmo->config.matter_power_spectrum_method != ccl_linear){
      printf("WARNING: BBKS + config.matter_power_spectrum_method = %d not yet supported\n continuing with linear power spectrum\n",cosmo->config.matter_power_spectrum_method);
    }

    gsl_spline2d * log_power_nl = gsl_spline2d_alloc(PNL_SPLINE_TYPE, nk,na);
    for (int j = 0; j < na; j++){
      double g2 = 2.*log(ccl_growth_factor(cosmo,z[j]));
      for (int i=0; i<nk; i++){
          y2d[j*nk+i] = y[i]+g2;
      }
    }

    status = gsl_spline2d_init(log_power_nl, x, z, y2d,nk,na);

    if (status){
      gsl_spline2d_free(log_power_nl);
      cosmo->status = 4;
      strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_bbks(): Error creating log_power_nl spline\n");
    }
    else
      cosmo->data.p_nl = log_power_nl;

    free(x);
    free(y);
    free(z);
    free(y2d);
}



/*------ ROUTINE: ccl_cosmology_compute_power ----- 
INPUT: ccl_cosmology * cosmo
TASK: compute distances, compute growth, compute power spectrum
*/

void ccl_cosmology_compute_power(ccl_cosmology * cosmo){
    if (cosmo->computed_power) return;

    switch(cosmo->config.transfer_function_method){
        case ccl_bbks:
            ccl_cosmology_compute_power_bbks(cosmo);
            break;
        case ccl_boltzmann_class:
           ccl_cosmology_compute_power_class(cosmo);
           break;

        default:
        cosmo->status = 11;
        sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power(): Unknown or non-implemented transfer function method: %d \n",cosmo->config.transfer_function_method);
    }
    cosmo->computed_power = true;
    ccl_check_status(cosmo);
    return;

}


/*------ ROUTINE: ccl_linear_matter_power ----- 
INPUT: ccl_cosmology * cosmo, a, k [1/Mpc]
TASK: compute the linear power spectrum at a given redshift
      by rescaling using the growth function
*/

double ccl_linear_matter_power(ccl_cosmology * cosmo, double a, double k){
    ccl_cosmology_compute_power(cosmo);
    double log_p_1;
    int status = gsl_spline_eval_e(cosmo->data.p_lin, log(k),NULL,&log_p_1);
    if (status){
        cosmo->status = 13;
        sprintf(cosmo->status_message ,"ccl_power.c: ccl_linear_matter_power(): Spline evaluation error\n");
       return NAN;
    }

    double p_1 = exp(log_p_1);

    if (a==1){
        return p_1;
    }

    double D = ccl_growth_factor(cosmo, a);
    double p = D*D*p_1;
    return p;
}


/*------ ROUTINE: ccl_nonlin_matter_power ----- 
INPUT: ccl_cosmology * cosmo, a, k [1/Mpc]
TASK: compute the nonlinear power spectrum at a given redshift
*/

double ccl_nonlin_matter_power(ccl_cosmology * cosmo, double a, double k){
    ccl_cosmology_compute_power(cosmo);
    double log_p_1;
    // log power at a=1 (z=0)
    int status =  gsl_spline2d_eval_e(cosmo->data.p_nl, log(k),a,NULL ,NULL ,&log_p_1);
    if (status){
       cosmo->status = 13;
       sprintf(cosmo->status_message ,"ccl_power.c: ccl_nonlin_matter_power(): Spline evaluation error\n");
       return NAN;
    }

    double p_1 = exp(log_p_1);
    return p_1;
}

//Params for sigma(R) integrand
typedef struct {
  ccl_cosmology *cosmo;
  double R;
} SigmaR_pars;

static double sigmaR_integrand(double lk,void *params)
{
  SigmaR_pars *par=(SigmaR_pars *)params;
  double k=pow(10.,lk);
  double pk=ccl_linear_matter_power(par->cosmo,1.,k);
  double kR=k*par->R;
  double w;
  if(kR<0.1) {
    w =1.-0.1*kR*kR+0.003571429*kR*kR*kR*kR
      -6.61376E-5*kR*kR*kR*kR*kR*kR
      +7.51563E-7*kR*kR*kR*kR*kR*kR*kR*kR;
  }
  else
    w = 3.*(sin(kR) - kR*cos(kR))/(kR*kR*kR);
  return pk*k*k*k*w*w;
}

double ccl_sigmaR(ccl_cosmology *cosmo,double R)
{
  SigmaR_pars par;
  par.cosmo=cosmo;
  par.R=R;

  gsl_integration_cquad_workspace *workspace=gsl_integration_cquad_workspace_alloc(1000);
  gsl_function F;
  F.function=&sigmaR_integrand;
  F.params=&par;

  double sigma_R;
  gsl_integration_cquad(&F,log10(K_MIN_INT),log10(K_MAX_INT),0.0,1E-5,workspace,&sigma_R,NULL,NULL);
  //TODO: log10 could be taken already in the macros.
  //TODO: 1E-5 should be a macro
  //TODO: we should check for integration success
  gsl_integration_cquad_workspace_free(workspace);

  return sqrt(sigma_R*M_LN10/(2*M_PI*M_PI));
}

double ccl_sigma8(ccl_cosmology *cosmo)
{
  return ccl_sigmaR(cosmo,8/cosmo->params.h);
}
