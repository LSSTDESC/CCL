#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include "ccl.h"
#include "ccl_f2d.h"
#include "ccl_emu17.h"
#include "ccl_emu17_params.h"

// helper function for BCM corrections
static void correct_bcm(ccl_cosmology *cosmo, int na, double *a_arr, int nk,
                        double *lk_arr,double *pk2d, int *status) {
  for (int ii=0; ii < na; ii++) {
    double a = a_arr[ii];
    for(int jj=0; jj < nk; jj++) {
      double k = exp(lk_arr[jj]);
      double fbcm = ccl_bcm_model_fka(cosmo, k, a, status);
      pk2d[jj+nk*ii] += log(fbcm);
    }
  }
}

// helper functions for BBKS and EH98
static double bbks_power(ccl_parameters *params, void *p, double k) {
  return ccl_bbks_power(params, k);
}

static double eh_power(ccl_parameters *params, void *p, double k) {
  return ccl_eh_power(params, (eh_struct*)p, k);
}

// helper functions for non-linear power tabulation
static double linear_power(ccl_cosmology* cosmo, double k, double a, void *p, int* status) {
  return ccl_linear_matter_power(cosmo, k, a, status);
}

static double halofit_power(ccl_cosmology* cosmo, double k, double a, void *p, int* status) {
  return ccl_halofit_power(cosmo, k, a, (halofit_struct*)p, status);
}

/*------ ROUTINE: ccl_cosmology_compute_power_analytic -----
INPUT: cosmology
TASK: provide spline for an analytic power spectrum with baryonic correction
*/

static void ccl_cosmology_compute_linpower_analytic(
    ccl_cosmology* cosmo, void* par,
    double (*pk)(ccl_parameters* params, void* p, double k),
    int* status) {
  double sigma8,log_sigma8;
  //These are the limits of the splining range
  double kmin = cosmo->spline_params.K_MIN;
  double kmax = cosmo->spline_params.K_MAX;
  //Compute nk from number of decades and N_K = # k per decade
  double ndecades = log10(kmax) - log10(kmin);
  int nk = (int)ceil(ndecades*cosmo->spline_params.N_K);
  // Compute na using predefined spline spacing
  double amin = cosmo->spline_params.A_SPLINE_MINLOG_PK;
  double amax = cosmo->spline_params.A_SPLINE_MAX;
  int na = cosmo->spline_params.A_SPLINE_NA_PK+cosmo->spline_params.A_SPLINE_NLOG_PK-1;

  // Exit if sigma8 wasn't specified
  if (isnan(cosmo->params.sigma8)) {
    *status = CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_analytic(): "
             "sigma8 not set, required for analytic power spectra\n");
    return;
  }

  // The x array is initially k, but will later
  // be overwritten with log(k)
  double *x=NULL, *y=NULL, *z=NULL, *y2d=NULL;
  x=ccl_log_spacing(kmin, kmax, nk);
  if(x==NULL) {
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,"ccl_power.c: ccl_cosmology_compute_power_analytic(): "
             "memory allocation\n");
  }
  if(*status==0) {
    y=malloc(sizeof(double)*nk);
    if(y==NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,"ccl_power.c: ccl_cosmology_compute_power_analytic(): "
               "memory allocation\n");
    }
  }
  if(*status==0) {
    z=ccl_linlog_spacing(amin, cosmo->spline_params.A_SPLINE_MIN_PK,
       amax, cosmo->spline_params.A_SPLINE_NLOG_PK,
       cosmo->spline_params.A_SPLINE_NA_PK);
    if(z==NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,"ccl_power.c: ccl_cosmology_compute_power_analytic(): "
               "memory allocation\n");
    }
  }

  if(*status==0) {
    y2d = malloc(nk * na * sizeof(double));
    if(y2d==NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,"ccl_power.c: ccl_cosmology_compute_power_analytic(): "
               "memory allocation\n");
    }
  }

  if(*status==0) {
    // Calculate P(k) on k grid. After this loop, x will contain log(k) and y
    // will contain log(pk) [which has not yet been normalized]
    // After this loop x will contain log(k)
    for (int i=0; i<nk; i++) {
      y[i] = log((*pk)(&cosmo->params, par, x[i]));
      x[i] = log(x[i]);
    }
  }

  if(*status==0) {
    for (int j = 0; j < na; j++) {
      double gfac = ccl_growth_factor(cosmo,z[j], status);
      double g2 = 2.*log(gfac);
      for (int i=0; i<nk; i++) {
        y2d[j*nk+i] = y[i]+g2;
      }
    }
  }

  if(*status==0) {
    cosmo->data.p_lin=ccl_f2d_t_new(na,z,nk,x,y2d,NULL,NULL,0,
      1,2,ccl_f2d_cclgrowth,1,NULL,0,2,
      ccl_f2d_3,status);
  }
  if(*status==0) {
    cosmo->computed_linear_power=true;
    sigma8 = ccl_sigma8(cosmo,status);
    cosmo->computed_linear_power=false;
  }

  if(*status==0) {
    // Calculate normalization factor using computed value of sigma8, then
    // recompute P(k, a) using this normalization
    log_sigma8 = 2*(log(cosmo->params.sigma8) - log(sigma8));
    for(int j=0;j<na*nk;j++)
      y2d[j] += log_sigma8;
  }

  if(*status==0) {
    // Free the previous P(k,a) spline, and allocate a new one to store the
    // properly-normalized P(k,a)
    ccl_f2d_t_free(cosmo->data.p_lin);
    cosmo->data.p_lin = ccl_f2d_t_new(
      na,z,nk,x,y2d,NULL,NULL,0,
      1,2,ccl_f2d_cclgrowth,1,NULL,0,2,
      ccl_f2d_3,status);
  }

  free(x);
  free(y);
  free(z);
  free(y2d);
}


/*------ ROUTINE: ccl_cosmology_compute_power_emu -----
INPUT: cosmology
TASK: provide spline for the emulated power spectrum from Cosmic EMU
*/

static void ccl_cosmology_compute_power_emu(ccl_cosmology * cosmo, int * status)
{
  double Omeganuh2_eq;

  // Check ranges to see if the cosmology is valid
  if(*status==0) {
    if((cosmo->params.h<0.55) || (cosmo->params.h>0.85)){
      *status=CCL_ERROR_INCONSISTENT;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_emu(): "
               "h is outside allowed range\n");
    }
  }

  if(*status==0) {
   // Check if the cosmology has been set up with equal neutrino masses for the emulator
    // If not, check if the user has forced redistribution of masses and if so do this.
    if(cosmo->params.N_nu_mass>0) {
      if (cosmo->config.emulator_neutrinos_method == ccl_emu_strict){
  if (cosmo->params.N_nu_mass==3){
    if (cosmo->params.m_nu[0] != cosmo->params.m_nu[1] || cosmo->params.m_nu[0] != cosmo->params.m_nu[2] || cosmo->params.m_nu[1] != cosmo->params.m_nu[2]){
      *status = CCL_ERROR_INCONSISTENT;
      ccl_cosmology_set_status_message(cosmo,
               "ccl_power.c: ccl_cosmology_compute_power_emu(): In the default configuration, you must pass a list of 3 "
               "equal neutrino masses or pass a sum and set m_nu_type = 'equal'. If you wish to over-ride this, "
               "set config->emulator_neutrinos_method = 'ccl_emu_equalize'. This will force the neutrinos to be of equal "
               "mass but will result in internal inconsistencies.\n");
    }
  }else if (cosmo->params.N_nu_mass!=3){
    *status = CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(cosmo,
             "ccl_power.c: ccl_cosmology_compute_power_emu(): In the default configuration, you must pass a list of 3 "
             "equal neutrino masses or pass a sum and set m_nu_type = 'equal'. If you wish to over-ride this, "
             "set config->emulator_neutrinos_method = 'ccl_emu_equalize'. This will force the neutrinos to be of equal "
             "mass but will result in internal inconsistencies.\n");
  }
      }else if (cosmo->config.emulator_neutrinos_method == ccl_emu_equalize){
  // Reset the masses to equal
  double mnu_eq[3] = {cosmo->params.sum_nu_masses / 3., cosmo->params.sum_nu_masses / 3., cosmo->params.sum_nu_masses / 3.};
  Omeganuh2_eq = ccl_Omeganuh2(1.0, 3, mnu_eq, cosmo->params.T_CMB, status);
      }
    } else {
      if(fabs(cosmo->params.N_nu_rel - 3.04)>1.e-6){
  *status=CCL_ERROR_INCONSISTENT;
  ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_emu(): "
           "Set Neff = 3.04 for cosmic emulator predictions in "
           "absence of massive neutrinos.\n");
      }
    }
  }

  if(*status==0) {
    double w0wacomb = -cosmo->params.w0 - cosmo->params.wa;
    if(w0wacomb<8.1e-3){ //0.3^4
      *status=CCL_ERROR_INCONSISTENT;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_emu(): "
               "w0 and wa do not satisfy the emulator bound\n");
    }
  }

  if(*status==0) {
    if(cosmo->params.Omega_nu_mass*cosmo->params.h*cosmo->params.h>0.01){
      *status=CCL_ERROR_INCONSISTENT;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_emu(): "
               "Omega_nu does not satisfy the emulator bound\n");
    }
  }

  if(*status==0) {
    // Check to see if sigma8 was defined
    if(isnan(cosmo->params.sigma8)){
      *status=CCL_ERROR_INCONSISTENT;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_emu(): "
               "sigma8 is not defined; specify sigma8 instead of A_s\n");
    }
  }

  int na=cosmo->spline_params.A_SPLINE_NA_PK;
  double *lpk_1a=NULL,*lk=NULL,*aemu=NULL,*lpk_nl=NULL;
  if (*status == 0) {
    //Now start the NL computation with the emulator
    //These are the limits of the splining range
    aemu = ccl_linear_spacing(A_MIN_EMU,cosmo->spline_params.A_SPLINE_MAX, na);
    if(aemu==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_emu(): "
               "memory allocation error\n");
    }
  }
  if (*status == 0) {
    lk=malloc(NK_EMU*sizeof(double));
    if(lk==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_emu(): "
               "memory allocation error\n");
    }
  }
  if (*status == 0) { //The emulator only computes power spectra at fixed nodes in k, given by the global variable "mode"
    for (int i=0; i<NK_EMU; i++)
      lk[i] = log(mode[i]);
  }
  if (*status == 0) {
    lpk_nl = malloc(NK_EMU * na * sizeof(double));
    if(lpk_nl==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_emu(): "
               "memory allocation error\n");
    }
  }
  if (*status == 0) {
    lpk_1a=malloc(NK_EMU*sizeof(double));
    if(lpk_1a==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_emu(): "
               "memory allocation error\n");
    }
  }

  if (*status == 0) {
    double emu_par[9];
    //For each redshift:
    for (int j = 0; j < na; j++){
      //Turn cosmology into emu_par:
      emu_par[0] = (cosmo->params.Omega_c+cosmo->params.Omega_b)*cosmo->params.h*cosmo->params.h;
      emu_par[1] = cosmo->params.Omega_b*cosmo->params.h*cosmo->params.h;
      emu_par[2] = cosmo->params.sigma8;
      emu_par[3] = cosmo->params.h;
      emu_par[4] = cosmo->params.n_s;
      emu_par[5] = cosmo->params.w0;
      emu_par[6] = cosmo->params.wa;
      if ((cosmo->params.N_nu_mass>0) && (cosmo->config.emulator_neutrinos_method == ccl_emu_equalize)){
  emu_par[7] = Omeganuh2_eq;
      }else{
  emu_par[7] = cosmo->params.Omega_nu_mass*cosmo->params.h*cosmo->params.h;
      }
      emu_par[8] = 1./aemu[j]-1;
      //Need to have this here because otherwise overwritten by emu in each loop

      //Call emulator at this redshift
      ccl_pkemu(emu_par,NK_EMU,lpk_1a, status, cosmo);
      if (*status) {
  *status=CCL_ERROR_MEMORY;
  ccl_cosmology_set_status_message(cosmo, "ccl_power.c: ccl_cosmology_compute_power_emu(): "
           "memory allocation error\n");
  break;
      }
      for (int i=0; i<NK_EMU; i++)
  lpk_nl[j*NK_EMU+i] = log(lpk_1a[i]);
    }
  }

  if (*status == 0) {
    if(cosmo->config.baryons_power_spectrum_method==ccl_bcm)
      correct_bcm(cosmo,na,aemu,NK_EMU,lk,lpk_nl,status);
  }

  if(*status==0) {
    cosmo->data.p_nl=ccl_f2d_t_new(na,aemu,NK_EMU,lk,lpk_nl,NULL,NULL,0,
     1,2,ccl_f2d_no_extrapol,
     1,NULL,0,2,ccl_f2d_3,status);
  }

  free(lpk_1a);
  free(lk);
  free(aemu);
  free(lpk_nl);
}


static void ccl_cosmology_spline_nonlinpower(
    ccl_cosmology* cosmo,
    double (*pk)(ccl_cosmology* cosmo, double k, double a, void *p, int* status),
    void *data,
    int* status) {

  double sigma8,log_sigma8;

  //These are the limits of the splining range
  double kmin = exp(cosmo->data.p_lin->lkmin);
  double kmax = exp(cosmo->data.p_lin->lkmax);

  //Compute nk from number of decades and N_K = # k per decade
  double ndecades = log10(kmax) - log10(kmin);
  int nk = (int)ceil(ndecades*cosmo->spline_params.N_K);

  // Compute na using predefined spline spacing
  double amin = cosmo->data.p_lin->amin;
  double amax = cosmo->data.p_lin->amax;
  int na = cosmo->spline_params.A_SPLINE_NA_PK + cosmo->spline_params.A_SPLINE_NLOG_PK - 1;

  // The x array is initially k, but will later
  // be overwritten with log(k)
  double *x=NULL, *z=NULL, *y2d=NULL;
  x = ccl_log_spacing(kmin, kmax, nk);

  if (x == NULL) {
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_power.c: ccl_cosmology_spline_nonlinpower(): memory allocation\n");
  }

  if (*status == 0) {
    z = ccl_linlog_spacing(amin, cosmo->spline_params.A_SPLINE_MIN_PK,
                           amax, cosmo->spline_params.A_SPLINE_NLOG_PK,
                           cosmo->spline_params.A_SPLINE_NA_PK);
    if (z == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_power.c: ccl_cosmology_spline_nonlinpower(): memory allocation\n");
    }
  }

  if (*status == 0) {
    y2d = malloc(nk * na * sizeof(double));
    if (y2d == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_power.c: ccl_cosmology_spline_nonlinpower(): memory allocation\n");
    }
  }

  if (*status == 0) {
    // Calculate P(k) on a, k grid. After this loop, x will contain log(k) and y
    // will contain log(pk) [which has not yet been normalized]
    for (int i=0; i<nk; i++) {
      for (int j = 0; j<na; j++) {
        if (*status == 0)
          y2d[j*nk + i] = log((*pk)(cosmo, x[i], z[j], data, status));
      }
    }

    // need log(k) for BCM and interp
    for (int i=0; i<nk; i++)
      x[i] = log(x[i]);
  }

  if (*status == 0) {
    if(cosmo->config.baryons_power_spectrum_method == ccl_bcm)
      correct_bcm(cosmo, na, z, nk, x, y2d, status);
  }

  if(*status == 0)
    cosmo->data.p_nl = ccl_f2d_t_new(na, z, nk, x, y2d, NULL, NULL, 0,
       1, 2, ccl_f2d_cclgrowth, 1, NULL,
       0, 2, ccl_f2d_3, status);

  free(x);
  free(z);
  free(y2d);
}

/*------ ROUTINE: ccl_cosmology_compute_power -----
INPUT: ccl_cosmology * cosmo
TASK: compute linear power spectrum
*/
void ccl_cosmology_compute_linear_power(ccl_cosmology* cosmo, ccl_f2d_t *psp, int* status) {
  if (cosmo->computed_linear_power) return;

  if (*status == 0) {
    // get linear P(k)
    switch (cosmo->config.transfer_function_method) {
      case ccl_transfer_none:
        break;

      case ccl_bbks:
        ccl_cosmology_compute_linpower_analytic(cosmo, NULL, bbks_power, status);
        break;

      case ccl_eisenstein_hu: {
          eh_struct *eh = NULL;
          eh = ccl_eh_struct_new(&(cosmo->params),1);
          if (eh != NULL) {
            ccl_cosmology_compute_linpower_analytic(cosmo, eh, eh_power, status);
          }
          free(eh);}
        break;

      case ccl_boltzmann_class:
        ccl_cosmology_spline_linpower_musigma(cosmo, psp, status);
        break;

      case ccl_boltzmann_camb:
        ccl_cosmology_spline_linpower_musigma(cosmo, psp, status);
        break;

      default: {
        *status = CCL_ERROR_INCONSISTENT;
        ccl_cosmology_set_status_message(
          cosmo,
          "ccl_power.c: ccl_cosmology_compute_power(): "
          "Unknown or non-implemented transfer function method: %d \n",
          cosmo->config.transfer_function_method);
        }
    }
  }

  if (*status == 0)
    cosmo->computed_linear_power = true;
}

void ccl_cosmology_compute_nonlin_power_from_f2d(ccl_cosmology *cosmo,
                                                 ccl_f2d_t *psp, int *status)
{
  cosmo->data.p_nl = ccl_f2d_t_copy(psp, status);
}

/*------ ROUTINE: ccl_cosmology_compute_power -----
INPUT: ccl_cosmology * cosmo
TASK: compute linear power spectrum
*/
void ccl_cosmology_compute_nonlin_power(ccl_cosmology* cosmo, ccl_f2d_t *psp_o,
                                        int* status) {
  if ((fabs(cosmo->params.mu_0)>1e-14 || fabs(cosmo->params.sigma_0)>1e-14) &&
      cosmo->config.matter_power_spectrum_method != ccl_linear) {
    *status = CCL_ERROR_NOT_IMPLEMENTED;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_power.c: ccl_cosmology_compute_power(): The power spectrum in the "
      "mu / Sigma modified gravity parameterisation is only implemented with "
      "the linear power spectrum.\n");
    return;
  }

  if (cosmo->computed_nonlin_power) return;

  // if everything is OK, get the non-linear P(K)
  if (*status == 0) {

    switch (cosmo->config.matter_power_spectrum_method) {

      case ccl_linear: {
        ccl_cosmology_spline_nonlinpower(cosmo, linear_power, NULL, status);}
        break;

      case ccl_halofit: {
        halofit_struct *hf = NULL;
        hf = ccl_halofit_struct_new(cosmo, status);
        if (*status == 0 && hf != NULL)
          ccl_cosmology_spline_nonlinpower(cosmo, halofit_power, (void*)hf, status);
        ccl_halofit_struct_free(hf);}
        break;

      case ccl_halo_model: {
        ccl_cosmology_compute_nonlin_power_from_f2d(cosmo, psp_o, status);}
        break;

      case ccl_emu: {
        ccl_cosmology_compute_power_emu(cosmo, status);}
        break;

    default: {
      *status = CCL_ERROR_INCONSISTENT;
      ccl_cosmology_set_status_message(
        cosmo,
        "ccl_power.c: ccl_cosmology_compute_power(): "
        "Unknown or non-implemented matter power spectrum method: %d \n",
        cosmo->config.matter_power_spectrum_method);
      }
    }
  }

  if (*status == 0)
    cosmo->computed_nonlin_power = true;
}

/*------ ROUTINE: ccl_linear_matter_power -----
INPUT: ccl_cosmology * cosmo, k [1/Mpc],a
TASK: compute the linear power spectrum at a given redshift
      by rescaling using the growth function
*/
double ccl_linear_matter_power(ccl_cosmology* cosmo, double k, double a, int* status) {
  if (!cosmo->computed_linear_power) {
    *status = CCL_ERROR_LINEAR_POWER_INIT;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_power.c: ccl_linear_matter_power(): linear power spctrum has not been computed!");
    return NAN;
  }

  if (cosmo->config.transfer_function_method != ccl_transfer_none)
    return ccl_f2d_t_eval(cosmo->data.p_lin,log(k),a,cosmo,status);
  else {
    *status = CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_power.c: ccl_linear_matter_power(): linear power spctrum is None!");
    return NAN;
  }
}

/*------ ROUTINE: ccl_nonlin_matter_power -----
INPUT: ccl_cosmology * cosmo, a, k [1/Mpc]
TASK: compute the nonlinear power spectrum at a given redshift
*/
double ccl_nonlin_matter_power(ccl_cosmology* cosmo, double k, double a, int* status) {
  if (!cosmo->computed_nonlin_power) {
    *status = CCL_ERROR_NONLIN_POWER_INIT;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_power.c: ccl_nonlin_matter_power(): non-linear power spctrum has not been computed!");
    return NAN;
  }

  return ccl_f2d_t_eval(cosmo->data.p_nl,log(k),a,cosmo,status);
}

// Params for sigma(R) integrand
typedef struct {
  ccl_cosmology *cosmo;
  double R;
  int* status;
} SigmaR_pars;


typedef struct {
  ccl_cosmology *cosmo;
  double R;
  int* status;
} SigmaV_pars;

/* --------- ROUTINE: w_tophat ---------
INPUT: kR, ususally a wavenumber multiplied by a smoothing radius
TASK: Output W(x)=[sin(x)-x*cos(x)]*(3/x)^3
*/
static double w_tophat(double kR) {
  double w;
  double kR2 = kR*kR;

  // This is the Maclaurin expansion of W(x)=[sin(x)-xcos(x)]*3/x**3 to O(x^10), with x=kR.
  // Necessary numerically because at low x W(x) relies on the fine cancellation of two terms
  if(kR<0.1) {
    w= 1. + kR2*(-1.0/10.0 + kR2*(1.0/280.0 +
      kR2*(-1.0/15120.0 + kR2*(1.0/1330560.0 +
      kR2* (-1.0/172972800.0)))));
  }
  else
    w = 3.*(sin(kR) - kR*cos(kR))/(kR2*kR);
  return w;
}

// Integrand for sigmaR integral
static double sigmaR_integrand(double lk,void *params) {
  SigmaR_pars *par=(SigmaR_pars *)params;

  double k=pow(10.,lk);
  double pk=ccl_linear_matter_power(par->cosmo,k, 1.,par->status);
  double kR=k*par->R;
  double w = w_tophat(kR);

  return pk*k*k*k*w*w;
}

// Integrand for sigmaV integral
static double sigmaV_integrand(double lk,void *params) {
  SigmaV_pars *par=(SigmaV_pars *)params;

  double k=pow(10.,lk);
  double pk=ccl_linear_matter_power(par->cosmo,k, 1.,par->status);
  double kR=k*par->R;
  double w = w_tophat(kR);

  return pk*k*w*w/3.0;
}

/* --------- ROUTINE: ccl_sigmaR ---------
INPUT: cosmology, comoving smoothing radius, scale factor
TASK: compute sigmaR, the variance in the *linear* density field
smoothed with a tophat filter of comoving size R
*/
double ccl_sigmaR(ccl_cosmology *cosmo,double R,double a,int *status) {
  if (!cosmo->computed_linear_power) {
    *status = CCL_ERROR_LINEAR_POWER_INIT;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_power.c: ccl_sigmaR(): linear power spctrum has not been computed!");
    return NAN;
  }
  if (!cosmo->computed_growth){
    *status = CCL_ERROR_GROWTH_INIT;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_power.c: ccl_sigmaR(): growth factor splines have not been precomputed!");
    return NAN;
  }

  SigmaR_pars par;
  par.status = status;

  par.cosmo=cosmo;
  par.R=R;
  gsl_integration_cquad_workspace *workspace =  NULL;
  gsl_function F;
  F.function=&sigmaR_integrand;
  F.params=&par;
  double sigma_R;

  workspace = gsl_integration_cquad_workspace_alloc(cosmo->gsl_params.N_ITERATION);
  if (workspace == NULL) {
    *status = CCL_ERROR_MEMORY;
  }
  if (*status == 0) {
    int gslstatus = gsl_integration_cquad(&F, log10(cosmo->spline_params.K_MIN), log10(cosmo->spline_params.K_MAX),
                                          0.0, cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL,
                                          workspace,&sigma_R,NULL,NULL);
    if(gslstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_power.c: ccl_sigmaR():");
      *status |= gslstatus;
    }
  }
  gsl_integration_cquad_workspace_free(workspace);

  return sqrt(sigma_R*M_LN10/(2*M_PI*M_PI))*ccl_growth_factor(cosmo, a, status);
}

/* --------- ROUTINE: ccl_sigmaV ---------
INPUT: cosmology, comoving smoothing radius, scale factor
TASK: compute sigmaV, the variance in the *linear* displacement field
smoothed with a tophat filter of comoving size R
The linear displacement field is the gradient of the linear density field
*/
double ccl_sigmaV(ccl_cosmology *cosmo,double R,double a,int *status) {
  if (!cosmo->computed_linear_power) {
    *status = CCL_ERROR_LINEAR_POWER_INIT;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_power.c: ccl_sigmaV(): linear power spctrum has not been computed!");
    return NAN;
  }
  if (!cosmo->computed_growth){
    *status = CCL_ERROR_GROWTH_INIT;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_power.c: ccl_sigmaV(): growth factor splines have not been precomputed!");
    return NAN;
  }

  SigmaV_pars par;
  par.status = status;

  par.cosmo=cosmo;
  par.R=R;
  gsl_integration_cquad_workspace *workspace = NULL;


  gsl_function F;
  F.function=&sigmaV_integrand;
  F.params=&par;
  double sigma_V;

  workspace = gsl_integration_cquad_workspace_alloc(cosmo->gsl_params.N_ITERATION);

  if (workspace == NULL) {
    *status = CCL_ERROR_MEMORY;
  }

  if (*status == 0) {
    int gslstatus = gsl_integration_cquad(&F, log10(cosmo->spline_params.K_MIN), log10(cosmo->spline_params.K_MAX),
            0.0, cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL,
            workspace,&sigma_V,NULL,NULL);

    if(gslstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_power.c: ccl_sigmaV():");
      *status |= gslstatus;
    }
  }

  gsl_integration_cquad_workspace_free(workspace);

  return sqrt(sigma_V*M_LN10/(2*M_PI*M_PI))*ccl_growth_factor(cosmo, a, status);
}

/* --------- ROUTINE: ccl_sigma8 ---------
INPUT: cosmology
TASK: compute sigma8, the variance in the *linear* density field at a=1
smoothed with a tophat filter of comoving size 8 Mpc/h
*/

double ccl_sigma8(ccl_cosmology *cosmo, int *status) {
  return ccl_sigmaR(cosmo, 8/cosmo->params.h, 1., status);
}
