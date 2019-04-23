#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include <class.h> /* from extern/ */

#include "ccl.h"


/*------ ROUTINE: ccl_cosmology_compute_power_class -----
INPUT: ccl_cosmology * cosmo
*/
static void ccl_free_class_structs(ccl_cosmology *cosmo,
           struct background *ba,
           struct thermo *th,
           struct perturbs *pt,
           struct transfers *tr,
           struct primordial *pm,
           struct spectra *sp,
           struct nonlinear *nl,
           struct lensing *le,
           int *init_arr,
           int * status)
{
  int i_init=6;
  if(init_arr[i_init--]) {
    if (spectra_free(sp) == _FAILURE_) {
      *status = CCL_ERROR_CLASS;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_free_class_structs(): "
               "Error freeing CLASS spectra:%s\n", sp->error_message);
      return;
    }
  }

  if(init_arr[i_init--]) {
    if (transfer_free(tr) == _FAILURE_) {
      *status = CCL_ERROR_CLASS;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_free_class_structs(): "
               "Error freeing CLASS transfer:%s\n", tr->error_message);
      return;
    }
  }

  if(init_arr[i_init--]) {
    if (nonlinear_free(nl) == _FAILURE_) {
      *status = CCL_ERROR_CLASS;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_free_class_structs(): "
               "Error freeing CLASS nonlinear:%s\n", nl->error_message);
      return;
    }
  }

  if(init_arr[i_init--]) {
    if (primordial_free(pm) == _FAILURE_) {
      *status = CCL_ERROR_CLASS;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_free_class_structs(): "
               "Error freeing CLASS pm:%s\n", pm->error_message);
      return;
    }
  }

  if(init_arr[i_init--]) {
    if (perturb_free(pt) == _FAILURE_) {
      *status = CCL_ERROR_CLASS;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_free_class_structs(): "
               "Error freeing CLASS pt:%s\n", pt->error_message);
      return;
    }
  }

  if(init_arr[i_init--]) {
    if (thermodynamics_free(th) == _FAILURE_) {
      *status = CCL_ERROR_CLASS;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_free_class_structs(): "
               "Error freeing CLASS thermo:%s\n", th->error_message);
      return;
    }
  }

  if(init_arr[i_init--]) {
    if (background_free(ba) == _FAILURE_) {
      *status = CCL_ERROR_CLASS;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_free_class_structs(): "
               "Error freeing CLASS bg:%s\n", ba->error_message);
      return;
    }
  }

  return;
}

static void ccl_class_preinit(struct background *ba,
            struct thermo *th,
            struct perturbs *pt,
            struct transfers *tr,
            struct primordial *pm,
            struct spectra *sp,
            struct nonlinear *nl,
            struct lensing *le)
{
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

static void ccl_run_class(ccl_cosmology *cosmo,
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
        int *init_arr,
        int * status)
{
  ErrorMsg errmsg;            // for error messages
  int i_init=0;
  ccl_class_preinit(ba,th,pt,tr,pm,sp,nl,le);

  if(input_init(fc,pr,ba,th,pt,tr,pm,sp,nl,le,op,errmsg) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_class(): "
             "Error running CLASS input:%s\n", errmsg);
    return;
  }
  if (background_init(pr,ba) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_class(): "
             "Error running CLASS background:%s\n", ba->error_message);
    return;
  }
  init_arr[i_init++]=1;
  if (thermodynamics_init(pr,ba,th) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_class(): "
             "Error running CLASS thermodynamics:%s\n", th->error_message);
    return;
  }
  init_arr[i_init++]=1;
  if (perturb_init(pr,ba,th,pt) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_class(): "
             "Error running CLASS pertubations:%s\n", pt->error_message);
    return;
  }
  init_arr[i_init++]=1;
  if (primordial_init(pr,pt,pm) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_class(): "
             "Error running CLASS primordial:%s\n", pm->error_message);
    return;
  }
  init_arr[i_init++]=1;
  if (nonlinear_init(pr,ba,th,pt,pm,nl) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_class(): "
             "Error running CLASS nonlinear:%s\n", nl->error_message);
    return;
  }
  init_arr[i_init++]=1;
  if (transfer_init(pr,ba,th,pt,nl,tr) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_class(): "
             "Error running CLASS transfer:%s\n", tr->error_message);
    return;
  }
  init_arr[i_init++]=1;
  if (spectra_init(pr,ba,pt,pm,nl,tr,sp) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_class(): "
             "Error running CLASS spectra:%s\n", sp->error_message);
    return;
  }
  init_arr[i_init++]=1;
}

static double ccl_get_class_As(ccl_cosmology *cosmo, struct file_content *fc, int position_As,
             double sigma8, int * status)
{
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

  //temporarily overwrite P_k_max_1/Mpc to speed up sigma8 calculation
  double k_max_old = 0.;
  int position_kmax =2;
  double A_s_guess;
  int init_arr[7]={0,0,0,0,0,0,0};

  if (strcmp(fc->name[position_kmax],"P_k_max_1/Mpc")) {
    k_max_old = strtof(fc->value[position_kmax],NULL);
    sprintf(fc->value[position_kmax],"%.15e",10.);
  }
  A_s_guess = 2.43e-9/0.87659*sigma8;
  sprintf(fc->value[position_As],"%.15e",A_s_guess);

  ccl_run_class(cosmo, fc,&pr,&ba,&th,&pt,&tr,&pm,&sp,&nl,&le,&op,init_arr,status);
  if (cosmo->status != CCL_ERROR_CLASS) A_s_guess*=pow(sigma8/sp.sigma8,2.);
  ccl_free_class_structs(cosmo, &ba,&th,&pt,&tr,&pm,&sp,&nl,&le,init_arr,status);

  if (k_max_old >0) {
    sprintf(fc->value[position_kmax],"%.15e",k_max_old);
  }
  return A_s_guess;
}

static void ccl_fill_class_parameters(ccl_cosmology * cosmo, struct file_content * fc,
              int parser_length, int * status)

{

  // initialize fc fields
  //silences Valgrind's "Conditional jump or move depends on uninitialised value" warning
  for (int i = 0; i< parser_length; i++){
    strcpy(fc->name[i]," ");
    strcpy(fc->value[i]," ");
  }

  strcpy(fc->name[0],"output");
  strcpy(fc->value[0],"mPk");

  strcpy(fc->name[1],"non linear");
  if (cosmo->config.matter_power_spectrum_method == ccl_halofit)
    strcpy(fc->value[1],"Halofit");
  else
    strcpy(fc->value[1],"none");

  strcpy(fc->name[2],"P_k_max_1/Mpc");
  sprintf(fc->value[2],"%.15e",cosmo->spline_params.K_MAX_SPLINE); //in units of 1/Mpc, corroborated with ccl_constants.h

  strcpy(fc->name[3],"z_max_pk");
  sprintf(fc->value[3],"%.15e",1./cosmo->spline_params.A_SPLINE_MINLOG_PK-1.);

  strcpy(fc->name[4],"modes");
  strcpy(fc->value[4],"s");

  strcpy(fc->name[5],"lensing");
  strcpy(fc->value[5],"no");

  // now, copy over cosmology parameters
  strcpy(fc->name[6],"h");
  sprintf(fc->value[6],"%.15e",cosmo->params.h);

  strcpy(fc->name[7],"Omega_cdm");
  sprintf(fc->value[7],"%.15e",cosmo->params.Omega_c);

  strcpy(fc->name[8],"Omega_b");
  sprintf(fc->value[8],"%.15e",cosmo->params.Omega_b);

  strcpy(fc->name[9],"Omega_k");
  sprintf(fc->value[9],"%.15e",cosmo->params.Omega_k);

  strcpy(fc->name[10],"n_s");
  sprintf(fc->value[10],"%.15e",cosmo->params.n_s);


  //cosmological constant?
  // set Omega_Lambda = 0.0 if w !=-1
  if ((cosmo->params.w0 !=-1.0) || (cosmo->params.wa !=0)) {
    strcpy(fc->name[11],"Omega_Lambda");
    sprintf(fc->value[11],"%.15e",0.0);

    strcpy(fc->name[12],"w0_fld");
    sprintf(fc->value[12],"%.15e",cosmo->params.w0);

    strcpy(fc->name[13],"wa_fld");
    sprintf(fc->value[13],"%.15e",cosmo->params.wa);
  }
  //neutrino parameters
  //massless neutrinos
  if (cosmo->params.N_nu_rel > 1.e-4) {
    strcpy(fc->name[14],"N_ur");
    sprintf(fc->value[14],"%.15e",cosmo->params.N_nu_rel);
  }
  else {
    strcpy(fc->name[14],"N_ur");
    sprintf(fc->value[14],"%.15e", 0.);
  }
  if (cosmo->params.N_nu_mass > 0) {
    strcpy(fc->name[15],"N_ncdm");
    sprintf(fc->value[15],"%d",cosmo->params.N_nu_mass);
    strcpy(fc->name[16],"m_ncdm");
    sprintf(fc->value[16],"%f", (cosmo->params.mnu)[0]);
    if (cosmo->params.N_nu_mass >=1){
    for (int i = 1; i < cosmo->params.N_nu_mass; i++) {
      char tmp[20];
      sprintf(tmp,", %f",(cosmo->params.mnu)[i]);
      strcat(fc->value[16],tmp);
    }
  }

  }

  strcpy(fc->name[17],"T_cmb");
  sprintf(fc->value[17],"%.15e",cosmo->params.T_CMB);

  //normalization comes last, so that all other parameters are filled in for determining A_s if sigma8 is specified
  if (isfinite(cosmo->params.sigma8) && isfinite(cosmo->params.A_s)){
      *status = CCL_ERROR_INCONSISTENT;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: class_parameters(): "
               "Error initializing CLASS parameters: both sigma8 and A_s defined\n");
    return;
  }
  if (isfinite(cosmo->params.sigma8)) {
    strcpy(fc->name[parser_length-1],"A_s");
    sprintf(fc->value[parser_length-1],"%.15e",ccl_get_class_As(cosmo,fc,parser_length-1,cosmo->params.sigma8, status));
  }
  else if (isfinite(cosmo->params.A_s)) {
    strcpy(fc->name[parser_length-1],"A_s");
    sprintf(fc->value[parser_length-1],"%.15e",cosmo->params.A_s);
  }
  else {
    *status = CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(cosmo, "ccl_power.c: class_parameters(): "
             "Error initializing CLASS pararmeters: "
             "neither sigma8 nor A_s defined\n");
    return;
  }

}

/*
 * Compute the power spectrum using CLASS
 * @param cosmo Cosmological parameters
 * @param status, integer indicating the status
 */
void ccl_cosmology_compute_linpower_class(ccl_cosmology* cosmo, int* status) {
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
  int init_arr[7]={0,0,0,0,0,0,0};
  int init_parser=0;
  if (parser_init(&fc,parser_length,"none",errmsg) == _FAILURE_) {
    *status = CCL_ERROR_CLASS;
    ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_class(): "
             "parser init error:%s\n", errmsg);
  }

  if(*status==0) {
    init_parser=1;
    ccl_fill_class_parameters(cosmo,&fc,parser_length, status);
  }

  if (*status==0)
    ccl_run_class(cosmo, &fc,&pr,&ba,&th,&pt,&tr,&pm,&sp,&nl,&le,&op,init_arr,status);

  if(init_parser)
    parser_free(&fc);

  double kmin,kmax,ndecades,amin,amax,ic;
  int nk,na,s;
  double *lk=NULL, *aa=NULL, *lpk_ln=NULL, *lpk_nl=NULL;
  if (*status == 0) {
    //CLASS calculations done - now allocate CCL splines
    kmin = 2*exp(sp.ln_k[0]);
    kmax = cosmo->spline_params.K_MAX_SPLINE;
    //Compute nk from number of decades and N_K = # k per decade
    ndecades = log10(kmax) - log10(kmin);
    nk = (int)ceil(ndecades*cosmo->spline_params.N_K);
    amin = cosmo->spline_params.A_SPLINE_MINLOG_PK;
    amax = cosmo->spline_params.A_SPLINE_MAX;
    na = cosmo->spline_params.A_SPLINE_NA_PK+cosmo->spline_params.A_SPLINE_NLOG_PK-1;

    // The lk array is initially k, but will later
    // be overwritten with log(k)
    lk=ccl_log_spacing(kmin, kmax, nk);
    if(lk==NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,"ccl_power.c: ccl_cosmology_compute_power_class(): "
               "memory allocation\n");
    }
  }

  if (*status == 0) {
    aa=ccl_linlog_spacing(amin, cosmo->spline_params.A_SPLINE_MIN_PK,
       amax, cosmo->spline_params.A_SPLINE_NLOG_PK,
       cosmo->spline_params.A_SPLINE_NA_PK);
    if(aa==NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,"ccl_power.c: ccl_cosmology_compute_power_class(): "
               "memory allocation\n");
    }
  }

  if(*status==0) {
    lpk_ln = malloc(nk * na * sizeof(double));
    if(lpk_ln==NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,"ccl_power.c: ccl_cosmology_compute_power_class(): "
               "memory allocation\n");
    }
  }

  if(*status==0) {
    lpk_nl = malloc(nk * na * sizeof(double));
    if(lpk_nl==NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,"ccl_power.c: ccl_cosmology_compute_power_class(): "
               "memory allocation\n");
    }
  }

  if(*status==0) {
    // After this loop lk will contain log(k), lpk_ln will contain log(P_lin), all in Mpc, not Mpc/h units!
    double psout_l;
    s=0;
    for (int i=0; i<nk; i++) {
      for (int j = 0; j < na; j++) {
  //The 2D interpolation routines access the function values pk_{k_ia_j} with the following ordering:
  //pk_ij = pk[j*N_k + i]
  //with i = 0,...,N_k-1 and j = 0,...,N_a-1.
  s |= spectra_pk_at_k_and_z(&ba, &pm, &sp,lk[i],1./aa[j]-1.+1e-10, &psout_l,&ic);
  lpk_ln[j*nk+i] = log(psout_l);
      }
      lk[i] = log(lk[i]);
    }
    if(s) {
      *status = CCL_ERROR_CLASS;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_class(): "
               "Error computing CLASS power spectrum\n");
    }
  }

  if(*status==0)
    cosmo->data.p_lin=ccl_f2d_t_new(na,aa,nk,lk,lpk_ln,1,2,ccl_f2d_cclgrowth,1,NULL,0,2,ccl_f2d_3,status);

  ccl_free_class_structs(cosmo,&ba,&th,&pt,&tr,&pm,&sp,&nl,&le,init_arr,status);
  free(lk);
  free(aa);
  free(lpk_nl);
  free(lpk_ln);
}
