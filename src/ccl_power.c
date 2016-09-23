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
#include "../class/include/class.h"


void ccl_cosmology_compute_power_class(ccl_cosmology * cosmo, int *status){

    if (*status){
        return;
    }
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
  // CLASS configuration parameters will be passed through this strcuture,
  // to avoid writing and reading .ini files for every call
  if (parser_init(&fc,15,"none",errmsg) == _FAILURE_){
    fprintf(stderr,"\n\nparser_init\n=>%s\n",errmsg);
    *status = 1;
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
  sprintf(fc.value[2],"%e",K_MAX);

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
      fprintf(stderr,"\n\nError initialzing pararmeters: both sigma_8 and A_s defined\n\n");
    *status = 1;
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
       fprintf(stderr,"\n\nError initialzing pararmeters: neither sigma_8 nor A_sdefined\n\n");
    *status = 1;
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
    fprintf(stderr,"\n\nError running input_init\n=>%s\n",errmsg);
    *status = 1;
    return;
  }

  if (background_init(&pr,&ba) == _FAILURE_) {
    fprintf(stderr,"\n\nError running background_init \n=>%s\n",ba.error_message);
    *status = 1;
    return;
  }

  if (thermodynamics_init(&pr,&ba,&th) == _FAILURE_) {
    fprintf(stderr,"\n\nError in thermodynamics_init \n=>%s\n",th.error_message);
    *status = 1;
    return;
  }

  if (perturb_init(&pr,&ba,&th,&pt) == _FAILURE_) {
    fprintf(stderr,"\n\nError in perturb_init \n=>%s\n",pt.error_message);
    *status = 1;
    return;
  }

  if (primordial_init(&pr,&pt,&pm) == _FAILURE_) {
    fprintf(stderr,"\n\nError in primordial_init \n=>%s\n",pm.error_message);
    *status = 1;
    return;
  }

  if (nonlinear_init(&pr,&ba,&th,&pt,&pm,&nl) == _FAILURE_) {
    fprintf(stderr,"\n\nError in nonlinear_init \n=>%s\n",nl.error_message);
    *status = 1;
    return;
  }
  if (transfer_init(&pr,&ba,&th,&pt,&nl,&tr) == _FAILURE_) {
     fprintf(stderr,"\n\nError in transfer_init \n=>%s\n",tr.error_message);
    *status = 1;
     return;
  }

 if (spectra_init(&pr,&ba,&pt,&pm,&nl,&tr,&sp) == _FAILURE_) {
     fprintf(stderr,"\n\nError in spectra_init \n=>%s\n",sp.error_message);
    *status = 1;
     return;
  }

    //CLASS calculations done - now allocate CCL splines
    double kmin = K_MIN;
    double kmax = K_MAX;
    int nk = N_K;
    double amin = A_SPLINE_MIN;
    double amax = A_SPLINE_MAX;
    int ak = N_A;

    // The x array is initially k, but will later
    // be overwritten with log(k)
    double * x = ccl_log_spacing(kmin, kmax, nk);
    double * z = malloc(sizeof(double)*nk);
    //The 2D interpolation routines access the function values y_{a_ik_j} with the following ordering:
    //y_ij = ya[j*N_a + i]
    //with i = 0,...,N_a-1 and j = 0,...,N_k-1.
    double * y = malloc(sizeof(double)*nk);

    if (z==NULL||y==NULL|| x==NULL){
        fprintf(stderr, "Could not allocate memory for power spectra\n");
        free(x);
        free(y);
        free(z);
        *status = 1;
        return;
    }

    // After this loop x will contain log(k), y will contain log(P_nl), z will contain log(P_lin)
    // all in Mpc, not Mpc/h units!
    double Z, ic;
    int s;
    for (int i=0; i<nk; i++){
        s =spectra_pk_at_k_and_z(&ba, &pm, &sp,x[i],0.0, &Z,&ic);
        z[i] = log(Z);
        //TODO: add loop over a for P_nl once 2D interpolation works!
        s = spectra_pk_nl_at_k_and_z(&ba, &pm, &sp,x[i],0.0, &Z);
        y[i] = log(Z);
        x[i] = log(x[i]);
    }

    gsl_spline * log_power_lin = gsl_spline_alloc(K_SPLINE_TYPE, nk);
    *status = gsl_spline_init(log_power_lin, x, z, nk);


    gsl_spline * log_power_nl = gsl_spline_alloc(K_SPLINE_TYPE, nk);
    *status = gsl_spline_init(log_power_nl, x, y, nk);

    free(x);
    free(y);
    free(z);

    cosmo->data.p_lin = log_power_lin;
    cosmo->data.p_nl = log_power_nl;
  if (spectra_free(&sp) == _FAILURE_) {
     fprintf(stderr,"\n\nError in spectra_free \n=>%s\n",sp.error_message);
     *status = 1;
     return;
  }

  if (transfer_free(&tr) == _FAILURE_) {
     fprintf(stderr,"\n\nError in transfer_free \n=>%s\n",tr.error_message);
     *status = 1;
     return;
  }
  if (nonlinear_free(&nl) == _FAILURE_) {
    fprintf(stderr,"\n\nError in nonlinear_free \n=>%s\n",nl.error_message);
     *status = 1;
     return;
  }

  if (primordial_free(&pm) == _FAILURE_) {
    fprintf(stderr,"\n\nError in primordial_free \n=>%s\n",pm.error_message);
     *status = 1;
     return;
  }

  if (perturb_free(&pt) == _FAILURE_) {
    fprintf(stderr,"\n\nError in perturb_free \n=>%s\n",pt.error_message);
     *status = 1;
     return;
  }

  if (thermodynamics_free(&th) == _FAILURE_) {
    fprintf(stderr,"\n\nError in thermodynamics_free \n=>%s\n",th.error_message);
     *status = 1;
     return;
  }

  if (background_free(&ba) == _FAILURE_) {
    fprintf(stderr,"\n\nError in background_free \n=>%s\n",ba.error_message);
     *status = 1;
     return;
  }
  if (parser_free(&fc)== _FAILURE_) {
    fprintf(stderr,"\n\nError in background_free \n=>%s\n",ba.error_message);
     *status = 1;
     return;
  }
}


void ccl_cosmology_compute_power_bbks(ccl_cosmology * cosmo, int *status){

    if (*status){
        return;
    }


    double kmin = K_MIN;
    double kmax = K_MAX;
    int nk = N_K;

    // The x array is initially k, but will later
    // be overwritten with log(k)
    double * x = ccl_log_spacing(kmin, kmax, nk);
    double * y = malloc(sizeof(double)*nk);

    if (y==NULL|| x==NULL){
        fprintf(stderr, "Could not allocate memory for power\n");
        free(x);
        free(y);
        *status = 1;
        return;
    }

    // After this loop k will contain 
    for (int i=0; i<nk; i++){
        y[i] = log(ccl_bbks_power(&cosmo->params, x[i]));
        x[i] = log(x[i]);
    }

    // now normalize to cosmo->params.sigma_8
    printf("test %e\n",cosmo->params.sigma_8);
    if (isnan(cosmo->params.sigma_8)){
        fprintf(stderr, "\nsigma_8 not set; required for BBKS power spectra\n");
        free(x);
        free(y);
        *status = 1;
        return;

    }
    gsl_spline * log_power_lin = gsl_spline_alloc(K_SPLINE_TYPE, nk);
    *status = gsl_spline_init(log_power_lin, x, y, nk);
    double sigma_8 = ccl_sigma8(log_power_lin, cosmo->params.h, status);
    double log_sigma_8 = log(cosmo->params.sigma_8) - log(sigma_8);
    for (int i=0; i<nk; i++){
        y[i] += log_sigma_8;
    }

    gsl_spline_free(log_power_lin);
    log_power_lin = gsl_spline_alloc(K_SPLINE_TYPE, nk);
    *status = gsl_spline_init(log_power_lin, x, y, nk);    

    if (cosmo->config.matter_power_spectrum_method != ccl_linear){
      printf("WARNING: BBKS + config.matter_power_spectrum_method = %d not yet supported\n continuing with linear power spectrum\n",cosmo->config.matter_power_spectrum_method);
    }

    gsl_spline * log_power_nl = gsl_spline_alloc(K_SPLINE_TYPE, nk);
    *status = gsl_spline_init(log_power_nl, x, y, nk);

    free(x);
    free(y);

    cosmo->data.p_lin = log_power_lin;
    cosmo->data.p_nl = log_power_nl;


}




void ccl_cosmology_compute_power(ccl_cosmology * cosmo, int *status){
    if (cosmo->computed_power) return;

    ccl_cosmology_compute_distances(cosmo, status);
    ccl_cosmology_compute_growth(cosmo, status);

    if (*status){
        return;
    }

    switch(cosmo->config.transfer_function_method){
        case ccl_bbks:
            ccl_cosmology_compute_power_bbks(cosmo, status);
            break;
        case ccl_boltzmann_class:
           ccl_cosmology_compute_power_class(cosmo, status);
           break;

        default:
            fprintf(stderr, "Unknown or non-implemented transfer function method\n");
            *status =1;
            return;
    }
    cosmo->computed_power = true;
    return;

}


double ccl_linear_matter_power(ccl_cosmology * cosmo, double a, double k, int * status){
    ccl_cosmology_compute_power(cosmo, status);
    if (*status) return NAN;
    double log_p_1;
    // log power at a=1 (z=0)
    *status = gsl_spline_eval_e(cosmo->data.p_lin, log(k), NULL,&log_p_1);
    if (*status){
       return NAN;
    // TODO: GSL spline error message
    }

    double p_1 = exp(log_p_1);

    if (a==1){
        return p_1;
    }

    double D = ccl_growth_factor(cosmo, a, status);
    double p = D*D*p_1;
    if (*status){
        p = NAN;
    }
    return p;
}

double ccl_nonlin_matter_power(ccl_cosmology * cosmo, double a, double k, int * status){
    ccl_cosmology_compute_power(cosmo, status);
    if (*status) return NAN;
    double log_p_1;
    // log power at a=1 (z=0)
    *status = gsl_spline_eval_e(cosmo->data.p_nl, log(k), NULL,&log_p_1);
    if (*status){
       return NAN;
    // TODO: GSL spline error message
    }

    double p_1 = exp(log_p_1);

    if (a==1){
        return p_1;
    }
    // WARNING: NOT CORRECT, but 2d interpolation in gsl is still flaky!
    double D = ccl_growth_factor(cosmo, a, status);
    double p = D*D*p_1;
    if (*status){
        p = NAN;
    }
    return p;
}
