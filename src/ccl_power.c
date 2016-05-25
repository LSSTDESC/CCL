#include "ccl_core.h"
#include "ccl_utils.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"
#include "ccl_placeholder.h"
#include "ccl_background.h"

void ccl_cosmology_compute_power_class(ccl_cosmology * cosmo, int *status){

    if (*status){
        return;
    }
  struct precision pr;        /* for precision parameters */
  struct background ba;       /* for cosmological background */
  struct thermo th;           /* for thermodynamics */
  struct perturbs pt;         /* for source functions */
  struct transfers tr;        /* for transfer functions */
  struct primordial pm;       /* for primordial spectra */
  struct nonlinear nl;        /* for non-linear spectra */
  ErrorMsg errmsg;            /* for error messages */
  struct file_content fc;   
  // generate file_content structure 
  // CLASS configuration parameters will be passed through this strcuture,
  // to avoid writing and reading .ini files for every call
  if (parser_init(&fc,15,"none",errmsg) == _FAILURE_){
    printf("\n\nparser_init\n=>%s\n",errmsg);
    return _FAILURE_;   
  }
  // basic CLASS configuration parameters
  // these need to be decided once, and they unchanged for (most) CLASS calls from CCL
  strcpy(fc.name[0],"output");
  strcpy(fc.value[0],"mPk");

  strcpy(fc.name[1],"non linear");
  strcpy(fc.value[1],"Halofit");

  strcpy(fc.name[2],"P_k_max_1/Mpc");
  strcpy(fc.value[2],"100.");

  strcpy(fc.name[3],"z_max_pk");
  strcpy(fc.value[3],"3.");

  strcpy(fc.name[4],"modes");
  strcpy(fc.value[4],"s");

  strcpy(fc.name[5],"lensing");
  strcpy(fc.value[5],"no");

  // now, copy over cosmology parameters
  strcpy(fc.name[6],"h");
  sprintf(fc.value[6],"%e",cosmo->params.h);

  strcpy(fc.name[7],"Omega_cdm");
  sprintf(fc.value[7],"%e",cosmo->Omega_cdm);

  strcpy(fc.name[8],"Omega_b");
  sprintf(fc.value[8],"%e",cosmo->Omega_b);

  strcpy(fc.name[9],"Omega_k");
  sprintf(fc.value[9],"%e",cosmo->Omega_k);

  strcpy(fc.name[10],"n_s");
  sprintf(fc.value[10],"%e",cosmo->n_s);

  strcpy(fc.name[11],"A_s");
  sprintf(fc.value[11],"%e",cosmo->A_s);

  // set Omega_Lambda = 0.0 if w !=-1
  if ((cosmo->w0 ==-1.0) && (cosmo->wa ==0)){
    strcpy(fc.name[12],"Omega_Lambda");
    sprintf(fc.value[12],"%e",1.0-cosmo->Omega_m-cosmo->Omega_n-cosmo->Omega_k);
  }
  else{
    strcpy(fc.name[12],"Omega_Lambda");
    sprintf(fc.value[12],"%e",0.0);

    strcpy(fc.name[13],"w0_fld");
    sprintf(fc.value[13],"%e",cosmo->w0);

    strcpy(fc.name[14],"wa_fld");
    sprintf(fc.value[14],"%e",cosmo->wa);
  }




  if (input_init(&fc,&pr,&ba,&th,&pt,&tr,&pm,&sp,&nl,&le,&op,errmsg) == _FAILURE_) {
    printf("\n\nError running input_init\n=>%s\n",errmsg);
    return _FAILURE_;
  }

  if (background_init(&pr,&ba) == _FAILURE_) {
    printf("\n\nError running background_init \n=>%s\n",ba.error_message);
    return _FAILURE_;
  }

  if (thermodynamics_init(&pr,&ba,&th) == _FAILURE_) {
    printf("\n\nError in thermodynamics_init \n=>%s\n",th.error_message);
    return _FAILURE_;
  }

  if (perturb_init(&pr,&ba,&th,&pt) == _FAILURE_) {
    printf("\n\nError in perturb_init \n=>%s\n",pt.error_message);
    return _FAILURE_;
  }

  if (primordial_init(&pr,&pt,&pm) == _FAILURE_) {
    printf("\n\nError in primordial_init \n=>%s\n",pm.error_message);
    return _FAILURE_;
  }

  if (nonlinear_init(&pr,&ba,&th,&pt,&pm,&nl) == _FAILURE_) {
    printf("\n\nError in nonlinear_init \n=>%s\n",nl.error_message);
    return _FAILURE_;
  }
    //CLASS calculations done - now allocate CCL splines
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
        y[i] = 0.;//TODO: evalute CLASS power spectra, getting units right...
        x[i] = log(x[i]);
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



    gsl_spline * log_power_nl = gsl_spline_alloc(K_SPLINE_TYPE, nk);
    *status = gsl_spline_init(log_power_nl, x, y, nk);

    free(x);
    free(y);

    cosmo->data.p_lin = log_power_lin;
    cosmo->data.p_nl = log_power_nl;
 if (nonlinear_free(&nl) == _FAILURE_) {
    printf("\n\nError in nonlinear_free \n=>%s\n",nl.error_message);
    return _FAILURE_;
  }

  if (primordial_free(&pm) == _FAILURE_) {
    printf("\n\nError in primordial_free \n=>%s\n",pm.error_message);
    return _FAILURE_;
  }

  if (perturb_free(&pt) == _FAILURE_) {
    printf("\n\nError in perturb_free \n=>%s\n",pt.error_message);
    return _FAILURE_;
  }

  if (thermodynamics_free(&th) == _FAILURE_) {
    printf("\n\nError in thermodynamics_free \n=>%s\n",th.error_message);
    return _FAILURE_;
  }

  if (background_free(&ba) == _FAILURE_) {
    printf("\n\nError in background_free \n=>%s\n",ba.error_message);
    return _FAILURE_;
  }
  if (parser_free(&fc)== _FAILURE_) {
    printf("\n\nError in background_free \n=>%s\n",ba.error_message);
    return _FAILURE_;
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

    if (*status){
        return;
    }

    switch(cosmo->config.transfer_function_method){
        case bbks:
            ccl_cosmology_compute_power_bbks(cosmo, status);
            break;
        case class:
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

    // log power at a=1 (z=0)
    double log_p_1 = gsl_spline_eval(cosmo->data.p_lin, log(k), NULL);
    // TODO: GSL spline error handling

    double p_1 = exp(log_p_1);

    if (a==1){
        return p_1;
    }

    //TODO: this is valid in the linear regime.
    double D = ccl_growth_factor(cosmo, a, status);
    double p = D*D*p_1;
    if (*status){
        p = NAN;
    }
    return p;
}

