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
  double * y = malloc(sizeof(double)*nk);
  double * z = ccl_linear_spacing(amin,amax, na);
  if (z==NULL||y==NULL|| x==NULL){
    *status = CCL_ERROR_SPLINE;
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
    int classstatus = gsl_spline_init(log_power_lin, x, y, nk);
    if (classstatus){
      gsl_spline_free(log_power_lin);
      ccl_free_class_structs(cosmo, &ba,&th,&pt,&tr,&pm,&sp,&nl,&le, status);
      *status = CCL_ERROR_SPLINE;
      strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_class(): Error creating log_power_lin spline\n");
      return;
    }
    else
      cosmo->data.p_lin = log_power_lin;

    if(cosmo->config.matter_power_spectrum_method==ccl_halofit){
      double * y2d = malloc(nk * na * sizeof(double));
      if (y2d==0){
        cosmo->status = CCL_ERROR_SPLINE;
        strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_class(): memory allocation error\n");
      }
      else{
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
       int pwstatus = gsl_spline2d_init(log_power_nl, x, z, y2d,nk,na);
       if (pwstatus){
        free(x);
        free(y);
        free(z);
        gsl_spline2d_free(log_power_nl);
        ccl_free_class_structs(cosmo, &ba,&th,&pt,&tr,&pm,&sp,&nl,&le,status);
        strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_class(): Error creating log_power_nl spline\n");
        return;
      }
      else
        cosmo->data.p_nl = log_power_nl;
    }

  }
  ccl_free_class_structs(cosmo, &ba,&th,&pt,&tr,&pm,&sp,&nl,&le, status);
  free(x);
  free(y);
  free(z);
  }
}

typedef struct {
  double rsound;
  double zeq;
  double keq;
  double zdrag;
  double kSilk;
  double rsound_approx;
  double th2p7;
  double alphac;
  double alphab;
  double betac;
  double betab;
  double bnode;
} eh_struct;

static eh_struct *eh_struct_new(ccl_parameters *params)
{
  //////
  // Computes Eisenstein & Hu parameters for
  // P_k and r_sound
  double OMh2,OBh2;
  double th2p7;
  eh_struct *eh=malloc(sizeof(eh_struct));
  if(eh==NULL)
    return NULL;

  OMh2=params->Omega_m*params->h*params->h;
  OBh2=params->Omega_b*params->h*params->h;
  th2p7=params->T_CMB/2.7;
  eh->th2p7=th2p7;
  eh->zeq=2.5E4*OMh2/pow(th2p7,4);
  eh->keq=0.0746*OMh2/(params->h*th2p7*th2p7);

  double b1,b2;
  b1=0.313*pow(OMh2,-0.419)*(1+0.607*pow(OMh2,0.674));
  b2=0.238*pow(OMh2,0.223);
  eh->zdrag=1291*pow(OMh2,0.251)*(1+b1*pow(OBh2,b2))/(1+0.659*pow(OMh2,0.828));

  double Req,Rd;
  Req=31.5*OBh2*1000./(eh->zeq*pow(th2p7,4));
  Rd=31.5*OBh2*1000./(eh->zdrag*pow(th2p7,4));
  eh->rsound=2/(3*eh->keq)*sqrt(6/Req)*
    log((sqrt(1+Rd)+sqrt(Rd+Req))/(1+sqrt(Req)));
  
  eh->kSilk=1.6*pow(OBh2,0.52)*pow(OMh2,0.73)*(1+pow(10.4*OMh2,-0.95))/params->h;

  double a1,a2,b_frac;
  a1=pow(46.9*OMh2,0.670)*(1+pow(32.1*OMh2,-0.532));
  a2=pow(12.0*OMh2,0.424)*(1+pow(45.0*OMh2,-0.582));
  b_frac=OBh2/OMh2;
  eh->alphac=pow(a1,-b_frac)*pow(a2,-b_frac*b_frac*b_frac);

  double bb1,bb2;
  bb1=0.944/(1+pow(458*OMh2,-0.708));
  bb2=pow(0.395*OMh2,-0.0266);
  eh->betac=1/(1+bb1*(pow(1-b_frac,bb2)-1));

  double y=(1+eh->zeq)/(1+eh->zdrag);
  double gy=y*(-6*sqrt(1+y)+(2+3*y)*log((sqrt(1+y)+1)/(sqrt(1+y)-1)));
  eh->alphab=2.07*eh->keq*eh->rsound*pow(1+Rd,-0.75)*gy;

  eh->betab=0.5+b_frac+(3-2*b_frac)*sqrt(pow(17.2*OMh2,2)+1);

  eh->bnode=8.41*pow(OMh2,0.435);

  eh->rsound_approx=params->h*44.5*log(9.83/OMh2)/
    sqrt(1+10*pow(OBh2,0.75));

  return eh;
}

static double tkEH_0(double keq,double k,double a,double b)
{
  //////
  // Eisentstein & Hu's Tk_0
  double q=k/(13.41*keq);
  double c=14.2/a+386./(1+69.9*pow(q,1.08));
  double l=log(M_E+1.8*b*q);
  return l/(l+c*q*q);
}

static double tkEH_c(eh_struct *eh,double k)
{
  //////
  // Eisenstein & Hu's Tk_c
  double f=1/(1+pow(k*eh->rsound/5.4,4));
  return f*tkEH_0(eh->keq,k,1,eh->betac)+
    (1-f)*tkEH_0(eh->keq,k,eh->alphac,eh->betac);
}

static double jbes0(double x)
{
  double jl;
  double ax=fabs(x);
  double ax2=ax*ax;

  if(ax<0.1) jl=1-ax2*(1-ax2/20.)/6.;
  else jl=sin(x)/x;

  return jl;
}

static double tkEH_b(eh_struct *eh,double k)
{
  //////
  // Eisenstein & Hu's Tk_b
  double x_bessel,part1,part2,jbes;
  double x=k*eh->rsound;

  if(k==0) x_bessel=0;
  else {
    x_bessel=x*pow(1+eh->bnode*eh->bnode*eh->bnode/(x*x*x),
		   -0.3333333333333333333);
  }

  part1=tkEH_0(eh->keq,k,1,1)/(1+pow(x/5.2,2));

  if(k==0) part2=0;
  else part2=eh->alphab/(1+pow(eh->betab/x,3))*
	 exp(-pow(k/eh->kSilk,1.4));
  
  return jbes0(x_bessel)*(part1+part2);
}

static double tsqr_EH(ccl_parameters *params,eh_struct * eh,double k,int wiggled)
{
  double tk;
  double b_frac=params->Omega_b/params->Omega_m;
  if(wiggled)
    tk=b_frac*tkEH_b(eh,k)+(1-b_frac)*tkEH_c(eh,k);
  else {
    double OMh2=params->Omega_m*params->h*params->h;
    double alpha_gamma=1-0.328*log(431*OMh2)*b_frac+0.38*log(22.3*OMh2)*b_frac*b_frac;
    double gamma_eff=params->Omega_m*params->h*(alpha_gamma+(1-alpha_gamma)/
						(1+pow(0.43*k*eh->rsound_approx,4)));
    double q=k*eh->th2p7*eh->th2p7/gamma_eff;
    double l0=log(2*M_E+1.8*q);
    double c0=14.2+731/(1+62.5*q);
    tk=l0/(l0+c0*q*q);
  }

  return tk*tk;
}

static double eh_power(ccl_parameters *params,eh_struct *eh,double k,int wiggled)
{
  //Wavenumber in units of Mpc^-1
  double kinvh=k/params->h;
  return pow(k,params->n_s)*tsqr_EH(params,eh,kinvh,wiggled);
}

static void ccl_cosmology_compute_power_eh(ccl_cosmology * cosmo, int * status)
{
  double kmin = K_MIN;
  double kmax = K_MAX;
  int nk = N_K;
  double amin = A_SPLINE_MIN;
  double amax = A_SPLINE_MAX;
  int na = N_A;
  eh_struct *eh=eh_struct_new(&(cosmo->params));
  if (eh==NULL) {
    *status=CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_eh(): memory allocation error\n");
    return;
  }

  // The x array is initially k, but will later
  // be overwritten with log(k)
  double * x = ccl_log_spacing(kmin, kmax, nk);
  double * y = malloc(sizeof(double)*nk);
  double * z = ccl_linear_spacing(amin,amax, na);
  double * y2d = malloc(nk * na * sizeof(double));
  if (z==NULL||y==NULL|| x==NULL || y2d==NULL){
    free(eh);
    *status=CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_eh(): memory allocation error\n");
    return;
  }

  // After this loop x will contain log(k)
  for (int i=0; i<nk; i++){
    y[i] = log(eh_power(&cosmo->params,eh,x[i],1));
    x[i] = log(x[i]);
  }

  // now normalize to cosmo->params.sigma_8
  if (isnan(cosmo->params.sigma_8)){
    free(eh);
    free(x);
    free(y);
    free(z);
    free(y2d);
    *status = CCL_ERROR_INCONSISTENT;
    strcpy(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_eh(): sigma_8 not set, required for E&H\n");
    return;
  }

  gsl_spline * log_power_lin = gsl_spline_alloc(K_SPLINE_TYPE, nk);
  int splinstatus = gsl_spline_init(log_power_lin, x, y, nk);

  if (splinstatus) {
    free(eh);
    free(x);
    free(y);
    free(z);
    free(y2d);
    gsl_spline_free(log_power_lin);
    *status = CCL_ERROR_SPLINE;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_eh(): Error creating log_power_lin spline\n");
    return;
  }

  cosmo->data.p_lin=log_power_lin;

  cosmo->computed_power=true;
  double sigma_8 = ccl_sigma8(cosmo);
  cosmo->computed_power=false;
  if (splinstatus){
    free(eh);
    free(x);
    free(y);
    free(z);
    free(y2d);
    gsl_spline_free(log_power_lin);
    *status = CCL_ERROR_INTEG;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_eh(): error calling ccl_sigma8()\n");
    return;
  }
  double log_sigma_8 = 2*(log(cosmo->params.sigma_8) - log(sigma_8));
  for (int i=0; i<nk; i++){
    y[i] += log_sigma_8;
  }

  gsl_spline_free(log_power_lin);
  log_power_lin = gsl_spline_alloc(K_SPLINE_TYPE, nk);

  splinstatus = gsl_spline_init(log_power_lin, x, y, nk);    
  if (splinstatus){
    free(eh);
    free(x);
    free(y);
    free(z);
    free(y2d);
    gsl_spline_free(log_power_lin);
    *status = CCL_ERROR_SPLINE;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_eh(): Error creating log_power_lin spline\n");
  }
  else
    cosmo->data.p_lin = log_power_lin;

  if (cosmo->config.matter_power_spectrum_method != ccl_linear){
    sprintf(cosmo->status_message,
	    "WARNING: E&H + config.matter_power_spectrum_method = %d not yet supported\n"
	    "continuing with linear power spectrum\n",cosmo->config.matter_power_spectrum_method);
  }

  gsl_spline2d * log_power_nl = gsl_spline2d_alloc(PNL_SPLINE_TYPE, nk,na);
  for (int j = 0; j < na; j++){
    double g2 = 2.*log(ccl_growth_factor(cosmo,z[j], status));
    for (int i=0; i<nk; i++){
      y2d[j*nk+i] = y[i]+g2;
    }
  }

  splinstatus = gsl_spline2d_init(log_power_nl, x, z, y2d,nk,na);

  if (splinstatus){
    free(eh);
    free(x);
    free(y);
    free(z);
    free(y2d);
    gsl_spline2d_free(log_power_nl);
    *status = CCL_ERROR_SPLINE;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_eh(): Error creating log_power_nl spline\n");
  }
  else
    cosmo->data.p_nl = log_power_nl;

  free(eh);
  free(x);
  free(y);
  free(z);
  free(y2d);
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
  if (z==NULL||y==NULL|| x==NULL || y2d==0){
    *status=CCL_ERROR_MEMORY;
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
        *status = CCL_ERROR_INCONSISTENT;
        strcpy(cosmo->status_message ,"ccl_power.c: ccl_cosmology_compute_power_bbks(): sigma_8 not set, required for BBKS\n");
        return;
    }

    gsl_spline * log_power_lin = gsl_spline_alloc(K_SPLINE_TYPE, nk);
    int splinstatus = gsl_spline_init(log_power_lin, x, y, nk);

    if (splinstatus){
      free(x);
      free(y);
      free(z);
      free(y2d);
      gsl_spline_free(log_power_lin);
      *status = CCL_ERROR_SPLINE;
      strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_bbks(): Error creating log_power_lin spline\n");
      return;
    }

    cosmo->data.p_lin=log_power_lin;

    cosmo->computed_power=true;
    double sigma_8 = ccl_sigma8(cosmo);
    cosmo->computed_power=false;
    if (splinstatus){
      free(x);
      free(y);
      free(z);
      free(y2d);
      gsl_spline_free(log_power_lin);
      *status = CCL_ERROR_INTEG;
      strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_bbks(): error calling ccl_sigma8()\n");
      return;
    }
    double log_sigma_8 = 2*(log(cosmo->params.sigma_8) - log(sigma_8));
    for (int i=0; i<nk; i++){
        y[i] += log_sigma_8;
    }

    gsl_spline_free(log_power_lin);
    log_power_lin = gsl_spline_alloc(K_SPLINE_TYPE, nk);

    splinstatus = gsl_spline_init(log_power_lin, x, y, nk);    
    if (splinstatus){
      free(x);
      free(y);
      gsl_spline_free(log_power_lin);
      *status = CCL_ERROR_SPLINE;
      strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_bbks(): Error creating log_power_lin spline\n");
    }
    else
      cosmo->data.p_lin = log_power_lin;


    if (cosmo->config.matter_power_spectrum_method != ccl_linear){
      printf("WARNING: BBKS + config.matter_power_spectrum_method = %d not yet supported\n continuing with linear power spectrum\n",cosmo->config.matter_power_spectrum_method);
    }

    gsl_spline2d * log_power_nl = gsl_spline2d_alloc(PNL_SPLINE_TYPE, nk,na);
    for (int j = 0; j < na; j++){
      double g2 = 2.*log(ccl_growth_factor(cosmo,z[j], status));
      for (int i=0; i<nk; i++){
          y2d[j*nk+i] = y[i]+g2;
      }
    }

    splinstatus = gsl_spline2d_init(log_power_nl, x, z, y2d,nk,na);

    if (splinstatus){
      gsl_spline2d_free(log_power_nl);
      *status = CCL_ERROR_SPLINE;
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
void ccl_cosmology_compute_power(ccl_cosmology * cosmo, int * status){

  if (cosmo->computed_power) return;
    switch(cosmo->config.transfer_function_method){
        case ccl_bbks:
	  ccl_cosmology_compute_power_bbks(cosmo,status);
	  break;
        case ccl_eisenstein_hu:
	  ccl_cosmology_compute_power_eh(cosmo,status);
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


/*------ ROUTINE: ccl_linear_matter_power ----- 
INPUT: ccl_cosmology * cosmo, a, k [1/Mpc]
TASK: compute the linear power spectrum at a given redshift
      by rescaling using the growth function
*/

double ccl_linear_matter_power(ccl_cosmology * cosmo, double a, double k, int * status){
  
    ccl_cosmology_compute_power(cosmo, status);
    double log_p_1;
    double deltak=1e-4;
    double deriv_plin_kmid,deriv2_plin_kmid;

    if(k<=K_MAX_SPLINE){
      int pkstatus = gsl_spline_eval_e(cosmo->data.p_lin, log(k), NULL,&log_p_1);
      if (pkstatus){
        *status = CCL_ERROR_SPLINE_EV;
        sprintf(cosmo->status_message ,"ccl_power.c: ccl_linear_matter_power(): Spline evaluation error\n");
	return NAN;
      }
    } else { //Extrapolate NL regime using log derivative
      
      double lkmid=log(K_MAX_SPLINE)-2*deltak;
      double lkmid_minus_2delta=lkmid-2*deltak;
      double lkmid_plus_2delta=log(K_MAX_SPLINE);
      double lkmid_minus_delta=lkmid-deltak;
      double lkmid_plus_delta=lkmid+deltak;
      double lplin_plus_2delta;
      int pkstatus =  gsl_spline_eval_e(cosmo->data.p_lin, lkmid_plus_2delta, NULL ,&lplin_plus_2delta);
      if (pkstatus){
	*status = CCL_ERROR_SPLINE_EV;
	sprintf(cosmo->status_message ,"ccl_power.c: ccl_linear_matter_power(): Spline evaluation error\n");
	return NAN;
      }
      double lplin_minus_2delta;
      pkstatus =  gsl_spline_eval_e(cosmo->data.p_lin, lkmid_minus_2delta, NULL ,&lplin_minus_2delta);
      if (pkstatus){
	*status = CCL_ERROR_SPLINE_EV;
	sprintf(cosmo->status_message ,"ccl_power.c: ccl_linear_matter_power(): Spline evaluation error\n");
	return NAN;
      }
      double lplin_plus_delta;
      pkstatus =  gsl_spline_eval_e(cosmo->data.p_lin, lkmid_plus_delta,NULL ,&lplin_plus_delta);
      if (pkstatus){
	*status = CCL_ERROR_SPLINE_EV;
	sprintf(cosmo->status_message ,"ccl_power.c: ccl_linear_matter_power(): Spline evaluation error\n");
	return NAN;
      }
      double lplin_minus_delta;
      pkstatus =  gsl_spline_eval_e(cosmo->data.p_lin, lkmid_minus_delta,NULL,&lplin_minus_delta);
      if (pkstatus){
	*status = CCL_ERROR_SPLINE_EV;
	sprintf(cosmo->status_message ,"ccl_power.c: ccl_linear_matter_power(): Spline evaluation error\n");
	return NAN;
      }
      double lplin_kmid;
      pkstatus =  gsl_spline_eval_e(cosmo->data.p_lin, lkmid,NULL,&lplin_kmid);
      if (pkstatus){
	*status = CCL_ERROR_SPLINE_EV;
	sprintf(cosmo->status_message ,"ccl_power.c: ccl_linear_matter_power(): Spline evaluation error\n");
	return NAN;
      }
      deriv_plin_kmid=(-lplin_plus_2delta+8.0*lplin_plus_delta-8.0*lplin_minus_delta+lplin_minus_2delta)/(12*deltak);
      deriv2_plin_kmid=(lplin_plus_delta-2*lplin_kmid+lplin_minus_delta)/deltak/deltak;
      log_p_1=lplin_kmid+deriv_plin_kmid*(log(k)-lkmid)+deriv2_plin_kmid/2.*(log(k)-lkmid)*(log(k)-lkmid);

    }

    double p_1 = exp(log_p_1);

    if (fabs(a-1.)<1.e-9){
      return p_1;
    } else {
      double D = ccl_growth_factor(cosmo, a, status);
      double p = D*D*p_1;
      return p;
    }

}


/*------ ROUTINE: ccl_nonlin_matter_power ----- 
INPUT: ccl_cosmology * cosmo, a, k [1/Mpc]
TASK: compute the nonlinear power spectrum at a given redshift
*/

double ccl_nonlin_matter_power(ccl_cosmology * cosmo, double a, double k, int *status){

	switch(cosmo->config.matter_power_spectrum_method){
	//If the matter PS specified was linear, then do the linear compuation
    case ccl_linear:
		return ccl_linear_matter_power(cosmo,a,k,status);
		
    case ccl_halofit:

    ccl_cosmology_compute_power(cosmo,status);
    
    double log_p_1;
    double deltak=1e-4;
    double deriv_pnl_kmid,deriv2_pnl_kmid;
    
    if(k<=K_MAX_SPLINE){
      
      int pwstatus =  gsl_spline2d_eval_e(cosmo->data.p_nl, log(k),a,NULL ,NULL ,&log_p_1);
      if (pwstatus){
	*status = CCL_ERROR_SPLINE_EV;
	sprintf(cosmo->status_message ,"ccl_power.c: ccl_nonlin_matter_power(): Spline evaluation error\n");
	return NAN;
      }
    } else { //Extrapolate NL regime using log derivative
      
      double lkmid=log(K_MAX_SPLINE)-2*deltak;
      double lkmid_minus_2delta=lkmid-2*deltak;
      double lkmid_plus_2delta=log(K_MAX_SPLINE);
      double lkmid_minus_delta=lkmid-deltak;
      double lkmid_plus_delta=lkmid+deltak;
      double lpnl_plus_2delta;
      int pwstatus =  gsl_spline2d_eval_e(cosmo->data.p_nl, lkmid_plus_2delta,a,NULL ,NULL ,&lpnl_plus_2delta);
      if (pwstatus){
	*status = CCL_ERROR_SPLINE_EV;
	sprintf(cosmo->status_message ,"ccl_power.c: ccl_nonlin_matter_power(): Spline evaluation error\n");
	return NAN;
      }
      double lpnl_minus_2delta;
      pwstatus =  gsl_spline2d_eval_e(cosmo->data.p_nl, lkmid_minus_2delta,a,NULL ,NULL ,&lpnl_minus_2delta);
      if (pwstatus){
	*status = CCL_ERROR_SPLINE_EV;
	sprintf(cosmo->status_message ,"ccl_power.c: ccl_nonlin_matter_power(): Spline evaluation error\n");
	return NAN;
      }
      double lpnl_plus_delta;
      pwstatus =  gsl_spline2d_eval_e(cosmo->data.p_nl, lkmid_plus_delta,a,NULL ,NULL ,&lpnl_plus_delta);
      if (pwstatus){
	*status = CCL_ERROR_SPLINE_EV;
	sprintf(cosmo->status_message ,"ccl_power.c: ccl_nonlin_matter_power(): Spline evaluation error\n");
	return NAN;
      }
      double lpnl_minus_delta;
      pwstatus =  gsl_spline2d_eval_e(cosmo->data.p_nl, lkmid_minus_delta,a,NULL ,NULL ,&lpnl_minus_delta);
      if (pwstatus){
	*status = CCL_ERROR_SPLINE_EV;
	sprintf(cosmo->status_message ,"ccl_power.c: ccl_nonlin_matter_power(): Spline evaluation error\n");
	return NAN;
      }
      double lpnl_kmid;
      pwstatus =  gsl_spline2d_eval_e(cosmo->data.p_nl, lkmid,a,NULL ,NULL ,&lpnl_kmid);
      if (pwstatus){
	*status = CCL_ERROR_SPLINE_EV;
	sprintf(cosmo->status_message ,"ccl_power.c: ccl_nonlin_matter_power(): Spline evaluation error\n");
	return NAN;
      }
      deriv_pnl_kmid=(-lpnl_plus_2delta+8.0*lpnl_plus_delta-8.0*lpnl_minus_delta+lpnl_minus_2delta)/(12*deltak);
      deriv2_pnl_kmid=(lpnl_plus_delta-2*lpnl_kmid+lpnl_minus_delta)/deltak/deltak;
      log_p_1=lpnl_kmid+deriv_pnl_kmid*(log(k)-lkmid)+deriv2_pnl_kmid/2.*(log(k)-lkmid)*(log(k)-lkmid);
    }
    
    double p_1 = exp(log_p_1);
    
    return p_1;

    default:
      printf("WARNING:  config.matter_power_spectrum_method = %d not yet supported\n continuing with linear power spectrum\n",cosmo->config.matter_power_spectrum_method);
      cosmo->config.matter_power_spectrum_method=ccl_linear;
      return ccl_linear_matter_power(cosmo,a,k,status);
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
  double pk=ccl_linear_matter_power(par->cosmo,1.,k, par->status);
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
