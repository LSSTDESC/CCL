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

// helper functions for BBKS and EH98
static double bbks_power(ccl_parameters *params, void *p, double k) {
  return ccl_bbks_power(params, k);
}

static double eh_power(ccl_parameters *params, void *p, double k) {
  return ccl_eh_power(params, (eh_struct*)p, k);
}

/*------ ROUTINE: ccl_cosmology_compute_power_analytic -----
INPUT: cosmology
TASK: provide spline for an analytic power spectrum with baryonic correction
*/

static ccl_f2d_t *ccl_compute_linpower_analytic(ccl_cosmology* cosmo, void* par,
                                                double (*pk)(ccl_parameters* params,
                                                             void* p, double k),
                                                int* status) {
  ccl_f2d_t *psp_out = NULL;
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
    ccl_cosmology_set_status_message(cosmo,
             "ccl_power.c: ccl_compute_linpower_analytic(): "
             "sigma8 not set, required for analytic power spectra\n");
    return NULL;
  }

  // The x array is initially k, but will later
  // be overwritten with log(k)
  double *x=NULL, *y=NULL, *z=NULL, *y2d=NULL;
  x=ccl_log_spacing(kmin, kmax, nk);
  if(x==NULL) {
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
             "ccl_power.c: ccl_compute_linpower_analytic(): "
             "memory allocation\n");
  }
  if(*status==0) {
    y=malloc(sizeof(double)*nk);
    if(y==NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
               "ccl_power.c: ccl_compute_linpower_analytic(): "
               "memory allocation\n");
    }
  }
  if(*status==0) {
    z=ccl_linlog_spacing(amin, cosmo->spline_params.A_SPLINE_MIN_PK,
       amax, cosmo->spline_params.A_SPLINE_NLOG_PK,
       cosmo->spline_params.A_SPLINE_NA_PK);
    if(z==NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
               "ccl_power.c: ccl_compute_linpower_analytic(): "
               "memory allocation\n");
    }
  }

  if(*status==0) {
    y2d = malloc(nk * na * sizeof(double));
    if(y2d==NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
               "ccl_power.c: ccl_compute_linpower_analytic(): "
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
    psp_out=ccl_f2d_t_new(na,z,nk,x,y2d,NULL,NULL,0,
                          1,2,ccl_f2d_cclgrowth,1,0,2,
                          ccl_f2d_3,status);
  }
  if(*status==0) {
    sigma8 = ccl_sigma8(cosmo, psp_out, status);
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
    ccl_f2d_t_free(psp_out);
    psp_out = ccl_f2d_t_new(na,z,nk,x,y2d,NULL,NULL,0,
                            1,2,ccl_f2d_cclgrowth,1,0,2,
                            ccl_f2d_3,status);
  }

  free(x);
  free(y);
  free(z);
  free(y2d);
  return psp_out;
}

ccl_f2d_t *ccl_compute_linpower_bbks(ccl_cosmology *cosmo, int *status)
{
  ccl_f2d_t *psp=ccl_compute_linpower_analytic(cosmo, NULL, bbks_power, status);
  return psp;
}

ccl_f2d_t *ccl_compute_linpower_eh(ccl_cosmology *cosmo, int *status)
{
  ccl_f2d_t *psp = NULL;
  eh_struct *eh = NULL;
  eh = ccl_eh_struct_new(&(cosmo->params),1);
  if (eh != NULL) {
    psp=ccl_compute_linpower_analytic(cosmo, eh,
                                      eh_power,
                                      status);
  }
  else
    *status = CCL_ERROR_MEMORY;
  free(eh);
  return psp;
}

/*------ ROUTINE: ccl_compute_power_emu -----
INPUT: cosmology
TASK: provide spline for the emulated power spectrum from Cosmic EMU
*/

ccl_f2d_t *ccl_compute_power_emu(ccl_cosmology * cosmo, int * status)
{
  double Omeganuh2_eq;
  ccl_f2d_t *psp_out=NULL;

  // Check ranges to see if the cosmology is valid
  if(*status==0) {
    if((cosmo->params.h<0.55) || (cosmo->params.h>0.85)){
      *status=CCL_ERROR_INCONSISTENT;
      ccl_cosmology_set_status_message(cosmo,
               "ccl_power.c: ccl_compute_power_emu(): "
               "h is outside allowed range\n");
    }
  }

  if(*status==0) {
   // Check if the cosmology has been set up with equal neutrino masses for the emulator
    // If not, check if the user has forced redistribution of masses and if so do this.
    if(cosmo->params.N_nu_mass>0) {
      if (cosmo->config.emulator_neutrinos_method == ccl_emu_strict){
        if (cosmo->params.N_nu_mass==3){
          if (cosmo->params.m_nu[0] != cosmo->params.m_nu[1] ||
              cosmo->params.m_nu[0] != cosmo->params.m_nu[2] ||
              cosmo->params.m_nu[1] != cosmo->params.m_nu[2]){
            *status = CCL_ERROR_INCONSISTENT;
            ccl_cosmology_set_status_message(cosmo,
                                             "ccl_power.c: ccl_compute_power_emu(): "
                                             "In the default configuration, you must pass a list of 3 "
                                             "equal neutrino masses or pass a sum and set "
                                             "m_nu_type = 'equal'. If you wish to over-ride this, "
                                             "set config->emulator_neutrinos_method = "
                                             "'ccl_emu_equalize'. This will force the neutrinos to "
                                             "be of equal mass but will result in "
                                             "internal inconsistencies.\n");
          }
        }else if (cosmo->params.N_nu_mass!=3){
          *status = CCL_ERROR_INCONSISTENT;
          ccl_cosmology_set_status_message(cosmo,
                                           "ccl_power.c: ccl_compute_power_emu(): "
                                           "In the default configuration, you must pass a list of 3 "
                                           "equal neutrino masses or pass a sum and set "
                                           "m_nu_type = 'equal'. If you wish to over-ride this, "
                                           "set config->emulator_neutrinos_method = "
                                           "'ccl_emu_equalize'. This will force the neutrinos to "
                                           "be of equal mass but will result in "
                                           "internal inconsistencies.\n");
        }
      }else if (cosmo->config.emulator_neutrinos_method == ccl_emu_equalize){
        // Reset the masses to equal
        double mnu_eq[3] = {cosmo->params.sum_nu_masses / 3.,
                            cosmo->params.sum_nu_masses / 3.,
                            cosmo->params.sum_nu_masses / 3.};
        Omeganuh2_eq = ccl_Omeganuh2(1.0, 3, mnu_eq, cosmo->params.T_CMB, status);
      }
    } else {
      if(fabs(cosmo->params.N_nu_rel - 3.04)>1.e-6){
        *status=CCL_ERROR_INCONSISTENT;
        ccl_cosmology_set_status_message(cosmo,
                                         "ccl_power.c: ccl_compute_power_emu(): "
                                         "Set Neff = 3.04 for cosmic emulator predictions in "
                                         "absence of massive neutrinos.\n");
      }
    }
  }

  if(*status==0) {
    double w0wacomb = -cosmo->params.w0 - cosmo->params.wa;
    if(w0wacomb<8.1e-3){ //0.3^4
      *status=CCL_ERROR_INCONSISTENT;
      ccl_cosmology_set_status_message(cosmo,
                                       "ccl_power.c: ccl_compute_power_emu(): "
                                       "w0 and wa do not satisfy the emulator bound\n");
    }
  }

  if(*status==0) {
    if(cosmo->params.Omega_nu_mass*cosmo->params.h*cosmo->params.h>0.01){
      *status=CCL_ERROR_INCONSISTENT;
      ccl_cosmology_set_status_message(cosmo,
                                       "ccl_power.c: ccl_compute_power_emu(): "
                                       "Omega_nu does not satisfy the emulator bound\n");
    }
  }

  if(*status==0) {
    // Check to see if sigma8 was defined
    if(isnan(cosmo->params.sigma8)){
      *status=CCL_ERROR_INCONSISTENT;
      ccl_cosmology_set_status_message(cosmo,
                                       "ccl_power.c: ccl_compute_power_emu(): "
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
      ccl_cosmology_set_status_message(cosmo,
                                       "ccl_power.c: ccl_compute_power_emu(): "
                                       "memory allocation error\n");
    }
  }
  if (*status == 0) {
    lk=malloc(NK_EMU*sizeof(double));
    if(lk==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
                                       "ccl_power.c: ccl_compute_power_emu(): "
                                       "memory allocation error\n");
    }
  }
  if (*status == 0) {
    //The emulator only computes power spectra at fixed nodes in k,
    //given by the global variable "mode"
    for (int i=0; i<NK_EMU; i++)
      lk[i] = log(mode[i]);
  }
  if (*status == 0) {
    lpk_nl = malloc(NK_EMU * na * sizeof(double));
    if(lpk_nl==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
                                       "ccl_power.c: ccl_compute_power_emu(): "
                                       "memory allocation error\n");
    }
  }
  if (*status == 0) {
    lpk_1a=malloc(NK_EMU*sizeof(double));
    if(lpk_1a==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
                                       "ccl_power.c: ccl_compute_power_emu(): "
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
      if ((cosmo->params.N_nu_mass>0) &&
          (cosmo->config.emulator_neutrinos_method == ccl_emu_equalize)){
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
        ccl_cosmology_set_status_message(cosmo,
                                         "ccl_power.c: ccl_compute_power_emu(): "
                                         "memory allocation error\n");
        break;
      }
      for (int i=0; i<NK_EMU; i++)
        lpk_nl[j*NK_EMU+i] = log(lpk_1a[i]);
    }
  }

  if(*status==0) {
    psp_out=ccl_f2d_t_new(na,aemu,NK_EMU,lk,lpk_nl,NULL,NULL,0,
                          1,2,ccl_f2d_no_extrapol,
                          1,0,2,ccl_f2d_3,status);
  }

  free(lpk_1a);
  free(lk);
  free(aemu);
  free(lpk_nl);
  return psp_out;
}


ccl_f2d_t *ccl_apply_halofit(ccl_cosmology* cosmo, ccl_f2d_t *plin, int *status)
{
  ccl_f2d_t *psp_out=NULL;
  size_t nk, na;
  double *x, *z, *y2d=NULL;

  //Halofit structure
  halofit_struct *hf=NULL;
  hf = ccl_halofit_struct_new(cosmo, plin, status);

  if(*status == 0) {
    //Find lk array
    if(plin->fk != NULL) {
      nk = plin->fk->size;
      x = plin->fk->x;
    }
    else {
      nk = plin->fka->interp_object.xsize;
      x = plin->fka->xarr;
    }

    //Find a array
    if(plin->fa != NULL) {
      na = plin->fa->size;
      z = plin->fa->x;
    }
    else {
      na = plin->fka->interp_object.ysize;
      z = plin->fka->yarr;
    }

    //Allocate pka array
    y2d = malloc(nk * na * sizeof(double));
    if (y2d == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
        "ccl_power.c: ccl_apply_halofit(): memory allocation\n");
    }
  }

  if (*status == 0) {
    // Calculate P(k) on a, k grid. After this loop, x will contain log(k) and y
    // will contain log(pk) [which has not yet been normalized]
    for (int j = 0; j<na; j++) {
      for (int i=0; i<nk; i++) {
        if (*status == 0) {
          double pk = ccl_halofit_power(cosmo, plin, x[i], z[j], hf, status);
          y2d[j*nk + i] = log(pk);
        }
      }
    }
  }

  if(*status == 0)
    psp_out = ccl_f2d_t_new(na, z, nk, x, y2d, NULL, NULL, 0,
                            1, 2, ccl_f2d_cclgrowth, 1,
                            0, 2, ccl_f2d_3, status);

  free(y2d);
  ccl_halofit_struct_free(hf);
  return psp_out;
}

void ccl_rescale_linpower(ccl_cosmology* cosmo, ccl_f2d_t *psp,
                          int rescale_mg, int rescale_norm,
                          int *status)
{
  if(rescale_mg || rescale_norm)
    ccl_rescale_musigma_s8(cosmo, psp, rescale_mg, status);
}

// Params for sigma(R) integrand
typedef struct {
  ccl_cosmology *cosmo;
  double R;
  double a;
  ccl_f2d_t *psp;
  int* status;
} SigmaR_pars;


typedef struct {
  ccl_cosmology *cosmo;
  double R;
  double a;
  ccl_f2d_t *psp;
  int* status;
} SigmaV_pars;

// Params for k_NL integrand
typedef struct {
  ccl_cosmology *cosmo;
  double a;
  ccl_f2d_t *psp;
  int* status;
} KNL_pars;

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
  double pk=ccl_f2d_t_eval(par->psp, lk * M_LN10, par->a,
                           par->cosmo, par->status);
  double kR=k*par->R;
  double w = w_tophat(kR);

  return pk*k*k*k*w*w;
}

// Integrand for sigmaV integral
static double sigmaV_integrand(double lk,void *params) {
  SigmaV_pars *par=(SigmaV_pars *)params;

  double k=pow(10.,lk);
  double pk=ccl_f2d_t_eval(par->psp, lk * M_LN10, par->a,
                           par->cosmo, par->status);
  double kR=k*par->R;
  double w = w_tophat(kR);

  return pk*k*w*w/3.0;
}

/* --------- ROUTINE: ccl_sigmaR ---------
INPUT: cosmology, comoving smoothing radius, scale factor
TASK: compute sigmaR, the variance in the *linear* density field
smoothed with a tophat filter of comoving size R
*/
double ccl_sigmaR(ccl_cosmology *cosmo,double R,double a,ccl_f2d_t *psp, int *status) {

  SigmaR_pars par;
  par.status = status;

  par.cosmo=cosmo;
  par.R=R;
  par.a=a;
  par.psp=psp;

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
    int gslstatus = gsl_integration_cquad(&F,
                                          log10(cosmo->spline_params.K_MIN),
                                          log10(cosmo->spline_params.K_MAX),
                                          0.0, cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL,
                                          workspace,&sigma_R,NULL,NULL);
    if(gslstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_power.c: ccl_sigmaR():");
      *status |= gslstatus;
    }
  }
  gsl_integration_cquad_workspace_free(workspace);

  return sqrt(sigma_R*M_LN10/(2*M_PI*M_PI));
}

/* --------- ROUTINE: ccl_sigmaV ---------
INPUT: cosmology, comoving smoothing radius, scale factor
TASK: compute sigmaV, the variance in the *linear* displacement field
smoothed with a tophat filter of comoving size R
The linear displacement field is the gradient of the linear density field
*/
double ccl_sigmaV(ccl_cosmology *cosmo,double R,double a,ccl_f2d_t *psp, int *status) {

  SigmaV_pars par;
  par.status = status;

  par.cosmo=cosmo;
  par.R=R;
  par.a=a;
  par.psp=psp;

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
    int gslstatus = gsl_integration_cquad(&F,
                                          log10(cosmo->spline_params.K_MIN),
                                          log10(cosmo->spline_params.K_MAX),
                                          0.0, cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL,
                                          workspace,&sigma_V,NULL,NULL);

    if(gslstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_power.c: ccl_sigmaV():");
      *status |= gslstatus;
    }
  }

  gsl_integration_cquad_workspace_free(workspace);

  return sqrt(sigma_V*M_LN10/(2*M_PI*M_PI));
}

/* --------- ROUTINE: ccl_sigma8 ---------
INPUT: cosmology
TASK: compute sigma8, the variance in the *linear* density field at a=1
smoothed with a tophat filter of comoving size 8 Mpc/h
*/

double ccl_sigma8(ccl_cosmology *cosmo, ccl_f2d_t *psp, int *status) {
  return ccl_sigmaR(cosmo, 8/cosmo->params.h, 1., psp, status);
}

// Integrand for kNL integral
static double kNL_integrand(double k,void *params) {
  KNL_pars *par=(KNL_pars *)params;

  double pk=ccl_f2d_t_eval(par->psp, log(k), par->a,
                           par->cosmo, par->status);

  return pk;
}

/* --------- ROUTINE: ccl_kNL ---------
INPUT: cosmology, scale factor
TASK: compute kNL, the scale for the non-linear cut
*/
double ccl_kNL(ccl_cosmology *cosmo,double a,ccl_f2d_t *psp, int *status) {

  KNL_pars par;
  par.status = status;
  par.a = a;
  par.psp=psp;

  par.cosmo=cosmo;
  gsl_integration_cquad_workspace *workspace =  NULL;
  gsl_function F;
  F.function=&kNL_integrand;
  F.params=&par;
  double PL_integral;

  workspace = gsl_integration_cquad_workspace_alloc(cosmo->gsl_params.N_ITERATION);
  if (workspace == NULL) {
    *status = CCL_ERROR_MEMORY;
  }
  if (*status == 0) {
    int gslstatus = gsl_integration_cquad(&F, cosmo->spline_params.K_MIN, cosmo->spline_params.K_MAX,
                                          0.0, cosmo->gsl_params.INTEGRATION_KNL_EPSREL,
                                          workspace,&PL_integral,NULL,NULL);
    if(gslstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_power.c: ccl_kNL():");
      *status |= gslstatus;
    }
  }
  gsl_integration_cquad_workspace_free(workspace);
  double sigma_eta = sqrt(PL_integral/(6*M_PI*M_PI));
  return pow(sigma_eta, -1);
}
