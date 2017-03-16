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
*/
static void ccl_free_class_structs(
               ccl_cosmology *cosmo,               
               struct background *ba,
               struct thermo *th,
               struct perturbs *pt,
               struct transfers *tr,
               struct primordial *pm,
               struct spectra *sp,
               struct nonlinear *nl,
               struct lensing *le,
	       int * status){
  if (spectra_free(sp) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_free_class_structs(): Error freeing CLASS spectra:%s\n",sp->error_message);
    return;
  }
  
  if (transfer_free(tr) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_free_class_structs(): Error freeing CLASS transfer:%s\n",tr->error_message);
    return;
  }

  if (nonlinear_free(nl) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_free_class_structs(): Error freeing CLASS nonlinear:%s\n",nl->error_message);
    return;
  }
  
  if (primordial_free(pm) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_free_class_structs(): Error freeing CLASS pm:%s\n",pm->error_message);
    return;
  }
  
  if (perturb_free(pt) == _FAILURE_) {
      *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_free_class_structs(): Error freeing CLASS pt:%s\n",pt->error_message);
    return;
  }
  
  if (thermodynamics_free(th) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_free_class_structs(): Error freeing CLASS thermo:%s\n",th->error_message);
    return;
  }

  if (background_free(ba) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_free_class_structs(): Error freeing CLASS bg:%s\n",ba->error_message);
    return;
  }
}
static void ccl_class_preinit(
               struct background *ba,
               struct thermo *th,
               struct perturbs *pt,
               struct transfers *tr,
               struct primordial *pm,
               struct spectra *sp,
               struct nonlinear *nl,
               struct lensing *le){
//pre-initialize all fields that are freed by *_free() routine
//prevents crashes if *_init()failed and did not initialize all tables freed by *_free()

  //init for background_free
  ba->tau_table = NULL;
  ba->z_table = NULL;
  ba->d2tau_dz2_table = NULL;
  ba->background_table = NULL;
  ba->d2background_dtau2_table = NULL;

  //init for thermodynamics_free
  th->z_table = NULL;
  th->thermodynamics_table = NULL;
  th->d2thermodynamics_dz2_table = NULL;

  //init for perturb_free
  pt->tau_sampling = NULL;
  pt->tp_size = NULL;
  pt->ic_size = NULL;
  pt->k = NULL;
  pt->k_size_cmb = NULL;
  pt->k_size_cl = NULL;
  pt->k_size = NULL;
  pt->sources = NULL;

  //init for primordial_free
  pm->amplitude = NULL;
  pm->tilt = NULL;
  pm->running = NULL;
  pm->lnpk = NULL;
  pm->ddlnpk = NULL;
  pm->is_non_zero = NULL;
  pm->ic_size = NULL;
  pm->ic_ic_size = NULL;
  pm->lnk = NULL;

  //init for nonlinear_free
  nl->k = NULL;
  nl->tau = NULL;
  nl->nl_corr_density = NULL;
  nl->k_nl = NULL;

  //init for transfer_free
  tr->tt_size = NULL;
  tr->l_size_tt = NULL;
  tr->l_size = NULL;
  tr->l = NULL;
  tr->q = NULL;
  tr->k = NULL;
  tr->transfer = NULL;

  //init for spectra_free
  //spectra_free checks all other data fields before freeing
  sp->is_non_zero = NULL;
  sp->ic_size = NULL;
  sp->ic_ic_size = NULL;
}
static void ccl_run_class(
               ccl_cosmology *cosmo, 
               struct file_content *fc,
               struct precision* pr,
               struct background* ba,
               struct thermo* th,
               struct perturbs* pt,
               struct transfers* tr,
               struct primordial* pm,
               struct spectra* sp,
               struct nonlinear* nl,
               struct lensing* le,
               struct output* op, 
	       int * status){
  ErrorMsg errmsg;            // for error messages 
  ccl_class_preinit(ba,th,pt,tr,pm,sp,nl,le);
  
  if(input_init(fc,pr,ba,th,pt,tr,pm,sp,nl,le,op,errmsg) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS input:%s\n",errmsg);
    return;
  }
  if (background_init(pr,ba) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS background:%s\n",ba->error_message);
    return;
  }
  if (thermodynamics_init(pr,ba,th) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS thermodynamics:%s\n",th->error_message);
    return;
  }
  if (perturb_init(pr,ba,th,pt) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS pertubations:%s\n",pt->error_message);
    return;
  }
  if (primordial_init(pr,pt,pm) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS primordial:%s\n",pm->error_message);
    return;
  }
  if (nonlinear_init(pr,ba,th,pt,pm,nl) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS nonlinear:%s\n",nl->error_message);
    return;
  }
  if (transfer_init(pr,ba,th,pt,nl,tr) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS transfer:%s\n",tr->error_message);
    return;
  }
  if (spectra_init(pr,ba,pt,pm,nl,tr,sp) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error running CLASS spectra:%s\n",sp->error_message);
    return;
  }
}

static double ccl_get_class_As(ccl_cosmology *cosmo, struct file_content *fc, int position_As,double sigma8, int * status){
//structures for class test run
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

  //temporarily overwrite P_k_max_1/Mpc to speed up sigma_8 calculation
  double k_max_old = 0.;
  int position_kmax =2;
  double A_s_guess;
  if (strcmp(fc->name[position_kmax],"P_k_max_1/Mpc")){
    k_max_old = strtof(fc->value[position_kmax],NULL);
    sprintf(fc->value[position_kmax],"%e",10.);  
  }
  A_s_guess = 2.43e-9/0.87659*sigma8;
  sprintf(fc->value[position_As],"%e",A_s_guess);

  ccl_run_class(cosmo, fc,&pr,&ba,&th,&pt,&tr,&pm,&sp,&nl,&le,&op,status);
//  printf("ran shooting for sigma_8 method\n Target sigma_8 = %e;\nGuessed A_s = %e -> sigma_8 = %e\nuse A_s = %e",sigma8,A_s_guess,sp.sigma8,A_s_guess*sigma8/sp.sigma8);
  if (cosmo->status != CCL_ERROR_CLASS) A_s_guess*=pow(sigma8/sp.sigma8,2.);
  ccl_free_class_structs(cosmo, &ba,&th,&pt,&tr,&pm,&sp,&nl,&le, status);

  if (k_max_old >0){
    sprintf(fc->value[position_kmax],"%e",k_max_old);      
  }
  return A_s_guess;
}

static void ccl_fill_class_parameters(ccl_cosmology * cosmo, struct file_content * fc,int parser_length, int * status){
  strcpy(fc->name[0],"output");
  strcpy(fc->value[0],"mPk");

  strcpy(fc->name[1],"non linear");
  if (cosmo->config.matter_power_spectrum_method == ccl_halofit){ strcpy(fc->value[1],"Halofit"); }
  else {strcpy(fc->value[1],"none");}

  strcpy(fc->name[2],"P_k_max_1/Mpc");
  sprintf(fc->value[2],"%e",K_MAX_SPLINE); //in units of 1/Mpc, corroborated with ccl_constants.h

  strcpy(fc->name[3],"z_max_pk");
  sprintf(fc->value[3],"%e",1./A_SPLINE_MIN-1.);

  strcpy(fc->name[4],"modes");
  strcpy(fc->value[4],"s");

  strcpy(fc->name[5],"lensing");
  strcpy(fc->value[5],"no");

  // now, copy over cosmology parameters
  strcpy(fc->name[6],"h");
  sprintf(fc->value[6],"%e",cosmo->params.h);

  strcpy(fc->name[7],"Omega_cdm");
  sprintf(fc->value[7],"%e",cosmo->params.Omega_c);

  strcpy(fc->name[8],"Omega_b");
  sprintf(fc->value[8],"%e",cosmo->params.Omega_b);

  strcpy(fc->name[9],"Omega_k");
  sprintf(fc->value[9],"%e",cosmo->params.Omega_k);

  strcpy(fc->name[10],"n_s");
  sprintf(fc->value[10],"%e",cosmo->params.n_s);


//cosmological constant?
// set Omega_Lambda = 0.0 if w !=-1
  if ((cosmo->params.w0 !=-1.0) || (cosmo->params.wa !=0)){
    strcpy(fc->name[11],"Omega_Lambda");
    sprintf(fc->value[11],"%e",0.0);

    strcpy(fc->name[12],"w0_fld");
    sprintf(fc->value[12],"%e",cosmo->params.w0);

    strcpy(fc->name[13],"wa_fld");
    sprintf(fc->value[13],"%e",cosmo->params.wa);
  }
  //normalization comes last, so that all other parameters are filled in for determining A_s if sigma_8 is specified
  if (isfinite(cosmo->params.sigma_8) && isfinite(cosmo->params.A_s)){
      *status = CCL_ERROR_INCONSISTENT;
      strcpy(cosmo->status_message ,"ccl_power.c: class_parameters(): Error initialzing CLASS pararmeters: both sigma_8 and A_s defined\n");
    return;
  }
  if (isfinite(cosmo->params.sigma_8)){
    strcpy(fc->name[parser_length-1],"A_s");
    sprintf(fc->value[parser_length-1],"%e",ccl_get_class_As(cosmo,fc,parser_length-1,cosmo->params.sigma_8, status));
  }
  else if (isfinite(cosmo->params.A_s)){ 
    strcpy(fc->name[parser_length-1],"A_s");
    sprintf(fc->value[parser_length-1],"%e",cosmo->params.A_s);
  }
  else{
      *status = CCL_ERROR_INCONSISTENT;
      strcpy(cosmo->status_message ,"ccl_power.c: class_parameters(): Error initialzing CLASS pararmeters: neither sigma_8 nor A_s defined\n");
    return;
  }
}

static void ccl_cosmology_compute_power_class(ccl_cosmology * cosmo, int * status){

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
  struct file_content fc;

  ErrorMsg errmsg; // for error messages 
  // generate file_content structure 
  // CLASS configuration parameters will be passed through this structure,
  // to avoid writing and reading .ini files for every call
  int parser_length = 20;
  if (parser_init(&fc,parser_length,"none",errmsg) == _FAILURE_){
    *status = CCL_ERROR_CLASS;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): parser init error:%s\n",errmsg);
    return;
  }

  ccl_fill_class_parameters(cosmo,&fc,parser_length, status);

  if (*status != CCL_ERROR_CLASS) ccl_run_class(cosmo, &fc,&pr,&ba,&th,&pt,&tr,&pm,&sp,&nl,&le,&op, status);
  if (*status == CCL_ERROR_CLASS){
    //printed error message while running CLASS
    ccl_free_class_structs(cosmo, &ba,&th,&pt,&tr,&pm,&sp,&nl,&le, status);
    return;
  }
  if (parser_free(&fc)== _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    strcpy(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error freeing CLASS parser\n");
    ccl_free_class_structs(cosmo, &ba,&th,&pt,&tr,&pm,&sp,&nl,&le, status);
    return;
  }

  //CLASS calculations done - now allocate CCL splines
  double kmin = K_MIN;
  double kmax = K_MAX_SPLINE;
  int nk = N_K;
  double amin = A_SPLINE_MIN;
  double amax = A_SPLINE_MAX;
  int na = N_A;
  
  // The x array is initially k, but will later
  // be overwritten with log(k)
  double * x = ccl_log_spacing(kmin, kmax, nk);
  double * z = ccl_linear_spacing(amin,amax, na);
  double * y2d_lin = malloc(nk * na * sizeof(double));
  double * y2d_nl = malloc(nk * na * sizeof(double));
  if (z==NULL|| x==NULL || y2d_lin==NULL || y2d_nl==NULL){
    *status = CCL_ERROR_SPLINE;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_class(): memory allocation error\n");
  }
  else{  
    // After this loop x will contain log(k), y will contain log(P_nl), z will contain log(P_lin)
    // all in Mpc, not Mpc/h units!
    double psout_l,ic;
    int s;
  
    for (int i=0; i<nk; i++){
      for (int j = 0; j < na; j++){
	//The 2D interpolation routines access the function values y_{k_ia_j} with the following ordering:
	//y_ij = y2d[j*N_k + i]
	//with i = 0,...,N_k-1 and j = 0,...,N_a-1.
	s |= spectra_pk_at_k_and_z(&ba, &pm, &sp,x[i],1./z[j]-1., &psout_l,&ic);
	y2d_lin[j*nk+i] = log(psout_l);
      }
    }
    if(s){
      free(x); 
      free(z);
      free(y2d_nl);
      free(y2d_lin);
      *status = CCL_ERROR_CLASS;
      strcpy(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error computing CLASS power spectrum\n");
      ccl_free_class_structs(cosmo, &ba,&th,&pt,&tr,&pm,&sp,&nl,&le,status);
    }
    for (int i=0; i<nk; i++)
      x[i] = log(x[i]);

    
    gsl_spline2d * log_power = gsl_spline2d_alloc(PLIN_SPLINE_TYPE, nk,na);
    int pwstatus = gsl_spline2d_init(log_power, x, z, y2d_lin,nk,na);
    if (pwstatus){
      free(x); 
      free(z);
      free(y2d_nl);
      free(y2d_lin);
      gsl_spline2d_free(log_power);
      ccl_free_class_structs(cosmo, &ba,&th,&pt,&tr,&pm,&sp,&nl,&le,status);
      strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_class(): Error creating log_power spline\n");
      return;
    } else {
      cosmo->data.p_lin = log_power;
    }

    if(cosmo->config.matter_power_spectrum_method==ccl_halofit){
      
      double psout_nl;
  
      for (int i=0; i<nk; i++){
	for (int j = 0; j < na; j++){
	  s |= spectra_pk_nl_at_k_and_z(&ba, &pm, &sp,exp(x[i]),1./z[j]-1.,&psout_nl);
	  y2d_nl[j*nk+i] = log(psout_nl);
	}
      }
      if(s){
	free(x); 
	free(z);
	free(y2d_nl);
	free(y2d_lin);
	*status = CCL_ERROR_CLASS;
	strcpy(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_class(): Error computing CLASS power spectrum\n");
	ccl_free_class_structs(cosmo, &ba,&th,&pt,&tr,&pm,&sp,&nl,&le,status);
      }
      
      gsl_spline2d * log_power_nl = gsl_spline2d_alloc(PNL_SPLINE_TYPE, nk,na);
      pwstatus = gsl_spline2d_init(log_power_nl, x, z, y2d_nl,nk,na);

      if (pwstatus){
	free(x); 
	free(z);
	free(y2d_nl);
	free(y2d_lin);
	gsl_spline2d_free(log_power_nl);
	ccl_free_class_structs(cosmo, &ba,&th,&pt,&tr,&pm,&sp,&nl,&le,status);
	strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_class(): Error creating log_power_nl spline\n");
	return;
      } else {
	cosmo->data.p_nl = log_power_nl;
      }
      free(y2d_nl);
    } 

    ccl_free_class_structs(cosmo, &ba,&th,&pt,&tr,&pm,&sp,&nl,&le, status);
    free(x);
    free(y2d_lin);
    free(z);
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

static void ccl_cosmology_compute_power_bbks(ccl_cosmology * cosmo, int * status){

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
  if (z==NULL||y==NULL|| x==NULL || y2d==NULL){
    int bbksstatus = 4;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_bbks(): memory allocation error\n");
    return;
  }

  // After this loop x will contain log(k)
  for (int i=0; i<nk; i++){
    y[i] = log(bbks_power(&cosmo->params, x[i]));
    x[i] = log(x[i]);
    //printf("%.3le %.3le\n",x[i],y[i]);
  }

  // now normalize to cosmo->params.sigma_8
  if (isnan(cosmo->params.sigma_8)){
    free(x);
    free(y);
    free(z);
    free(y2d);
    *status = CCL_ERROR_INCONSISTENT;
    strcpy(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_bbks(): sigma_8 not set, required for BBKS\n");
    return;
  }
  
  gsl_spline2d * log_power_lin = gsl_spline2d_alloc(PLIN_SPLINE_TYPE, nk,na);
  for (int j = 0; j < na; j++){
    double g2 = 2.*log(ccl_growth_factor(cosmo,z[j], status));
    for (int i=0; i<nk; i++){
      y2d[j*nk+i] = y[i]+g2;
    }
  }
  
  int splinstatus = gsl_spline2d_init(log_power_lin, x, z, y2d,nk,na);
  
  if (splinstatus){
    free(x);
    free(y);
    free(z);
    free(y2d);
    gsl_spline2d_free(log_power_lin);
    *status = CCL_ERROR_SPLINE;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_bbks(): Error creating log_power_lin spline\n");
    return;
    }
  cosmo->data.p_lin=log_power_lin;
  
  cosmo->computed_power=true;
  double sigma_8 = ccl_sigma8(cosmo);
  cosmo->computed_power=false;
  
  double log_sigma_8 = 2*(log(cosmo->params.sigma_8) - log(sigma_8));
  for (int i=0; i<nk; i++){
    y[i] += log_sigma_8;
  }
  
  for (int j = 0; j < na; j++){
    double g2 = 2.*log(ccl_growth_factor(cosmo,z[j], status));
    for (int i=0; i<nk; i++){
      y2d[j*nk+i] = y[i]+g2;
    }
  }

  gsl_spline2d_free(log_power_lin);
  log_power_lin = gsl_spline2d_alloc(PLIN_SPLINE_TYPE, nk,na);
  splinstatus = gsl_spline2d_init(log_power_lin, x, z, y2d,nk,na);
  if (splinstatus){
    free(x);
    free(y);
    free(z);
    free(y2d);
    gsl_spline2d_free(log_power_lin);
    *status = CCL_ERROR_SPLINE;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_bbks(): Error creating log_power_lin spline\n");
    return;
  } else {
    cosmo->data.p_lin=log_power_lin;
  }
  

  gsl_spline2d * log_power_nl = gsl_spline2d_alloc(PNL_SPLINE_TYPE, nk,na);
  splinstatus = gsl_spline2d_init(log_power_nl, x, z, y2d,nk,na);
  
  if (splinstatus){
    free(x);
    free(y);
    free(z);
    free(y2d);
    gsl_spline2d_free(log_power_nl);
    gsl_spline2d_free(log_power_lin);
    *status = CCL_ERROR_SPLINE;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_bbks(): Error creating log_power_nl spline\n");
    return;
  } else {
    cosmo->data.p_nl = log_power_nl;
  }

  free(x);
  free(y);
  free(z);
  free(y2d);
}



/*------ ROUTINE: ccl_cosmology_compute_power ----- 
INPUT: ccl_cosmology * cosmo
TASK: compute power spectrum
*/
void ccl_cosmology_compute_power(ccl_cosmology * cosmo, int * status){

  
  if (cosmo->computed_power) return;
    switch(cosmo->config.transfer_function_method){
        case ccl_bbks:
	  ccl_cosmology_compute_power_bbks(cosmo,status);
	  break;
        case ccl_boltzmann_class:
	  ccl_cosmology_compute_power_class(cosmo,status);
	  break;
        default:
	  *status = CCL_ERROR_INCONSISTENT;
	  sprintf(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power(): Unknown or non-implemented transfer function method: %d \n",cosmo->config.transfer_function_method);
    }
    cosmo->computed_power = true;
    ccl_check_status(cosmo,status);
    return;

}


/*------ ROUTINE: ccl_power_extrapol_hxighk ----- 
INPUT: ccl_cosmology * cosmo, a, k [1/Mpc]
TASK: extrapolate power spectrum at high k
*/
static double ccl_power_extrapol_highk(ccl_cosmology * cosmo, double k, double a, gsl_spline2d * powerspl, int * status){

  double log_p_1;
  double deltak=1e-2; //step for numerical derivative;
  double deriv_pk_kmid,deriv2_pk_kmid;

  double lkmid=log(K_MAX_SPLINE)-2*deltak;
  double lpk_kmid;
  int pwstatus =  gsl_spline2d_eval_e(powerspl, lkmid,a,NULL ,NULL ,&lpk_kmid);
  if (pwstatus){
    *status = CCL_ERROR_SPLINE_EV;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_nonlin_matter_power(): Spline evaluation error\n");
    return NAN;
  }
  //GSL derivatives
  pwstatus = gsl_spline2d_eval_deriv_x_e (powerspl, lkmid, a, NULL,NULL,&deriv_pk_kmid);
  if (pwstatus){
    *status = CCL_ERROR_SPLINE_EV;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_nonlin_matter_power(): Spline evaluation error\n");
    return NAN;
  }
  pwstatus = gsl_spline2d_eval_deriv_xx_e (powerspl, lkmid, a, NULL,NULL,&deriv2_pk_kmid);
  if (pwstatus){
    *status = CCL_ERROR_SPLINE_EV;
    sprintf(cosmo->status_message ,"ccl_power.c: ccl_nonlin_matter_power(): Spline evaluation error\n");
    return NAN;
  }
  
  log_p_1=lpk_kmid+deriv_pk_kmid*(log(k)-lkmid)+deriv2_pk_kmid/2.*(log(k)-lkmid)*(log(k)-lkmid);

  return log_p_1;
    
}


/*------ ROUTINE: ccl_linear_matter_power ----- 
INPUT: ccl_cosmology * cosmo, k [1/Mpc],a
TASK: compute the linear power spectrum at a given redshift
      by rescaling using the growth function
*/

double ccl_linear_matter_power(ccl_cosmology * cosmo, double k, double a, int * status){
 
  if (!cosmo->computed_power) ccl_cosmology_compute_power(cosmo, status);
  double log_p_1;
  int pkstatus;
  
  switch(cosmo->config.transfer_function_method){

  case ccl_bbks :
    
    pkstatus = gsl_spline2d_eval_e(cosmo->data.p_lin, log(k), a,NULL,NULL,&log_p_1);
    if (pkstatus){
      *status = CCL_ERROR_SPLINE_EV;
      sprintf(cosmo->status_message ,"ccl_power.c: ccl_linear_matter_power(): Spline evaluation error\n");
      return NAN;
    } else {
      return exp(log_p_1);
    }
    break;
    
  default : 

    if(k<=K_MAX_SPLINE){
      pkstatus = gsl_spline2d_eval_e(cosmo->data.p_lin, log(k), a,NULL,NULL,&log_p_1);
      if (pkstatus){
	*status = CCL_ERROR_SPLINE_EV;
	sprintf(cosmo->status_message ,"ccl_power.c: ccl_linear_matter_power(): Spline evaluation error\n");
	return NAN;
      } else {
	return exp(log_p_1);
      }
    } else { //Extrapolate NL regime using log derivative
      log_p_1 = ccl_power_extrapol_highk(cosmo,k,a,cosmo->data.p_lin,status);
      return exp(log_p_1);
    }
    break;
    
  } 
  
}


/*------ ROUTINE: ccl_nonlin_matter_power ----- 
INPUT: ccl_cosmology * cosmo, a, k [1/Mpc]
TASK: compute the nonlinear power spectrum at a given redshift
*/

double ccl_nonlin_matter_power(ccl_cosmology * cosmo, double k, double a, int *status){

  switch(cosmo->config.matter_power_spectrum_method){
    //If the matter PS specified was linear, then do the linear compuation
  case ccl_linear:
    return ccl_linear_matter_power(cosmo,k,a,status);
    
  case ccl_halofit:
    
    if (!cosmo->computed_power) ccl_cosmology_compute_power(cosmo,status);
    
    double log_p_1;
    
    if(k<=K_MAX_SPLINE){
      
      int pwstatus =  gsl_spline2d_eval_e(cosmo->data.p_nl, log(k),a,NULL ,NULL ,&log_p_1);
      if (pwstatus){
	*status = CCL_ERROR_SPLINE_EV;
	sprintf(cosmo->status_message ,"ccl_power.c: ccl_nonlin_matter_power(): Spline evaluation error\n");
	return NAN;
      } else {
	return exp(log_p_1);
      }
    } else { //Extrapolate NL regime using log derivative
      log_p_1 = ccl_power_extrapol_highk(cosmo,k,a,cosmo->data.p_nl,status);
      return exp(log_p_1);
    }

    default:
      printf("WARNING:  config.matter_power_spectrum_method = %d not yet supported\n continuing with linear power spectrum\n",cosmo->config.matter_power_spectrum_method);
      cosmo->config.matter_power_spectrum_method=ccl_linear;
      return ccl_linear_matter_power(cosmo,k,a,status);
  }
}


//Params for sigma(R) integrand
typedef struct {
  ccl_cosmology *cosmo;
  double R;
  int* status;
} SigmaR_pars;

static double sigmaR_integrand(double lk,void *params)
{
  SigmaR_pars *par=(SigmaR_pars *)params;
  int stat = 0;
  par->status = &stat;
  
  double k=pow(10.,lk);
  double pk=ccl_linear_matter_power(par->cosmo,k, 1.,par->status);
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
  int stat = 0;
  par.status = &stat;
  
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
