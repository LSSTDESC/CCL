#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>

#include "ccl.h"
#include "ccl_f2d.h"

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

ccl_f2d_t *ccl_compute_linpower_eh(ccl_cosmology *cosmo, int wiggled, int *status)
{
  ccl_f2d_t *psp = NULL;
  eh_struct *eh = NULL;
  eh = ccl_eh_struct_new(&(cosmo->params),wiggled);
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
    ccl_rescale_musigma_s8(cosmo, psp, rescale_mg, rescale_norm, status);
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
static double w_tophat_2d(double kR) {
  double w;
  double kR2 = kR*kR;

  // This is the Maclaurin expansion of W(x)=2 J1(x)/x to O(x^10), with x=kR.
  // Necessary numerically because at low x W(x) relies on the fine cancellation of two terms
  if(kR<0.1) {
    w= 1. + kR2*(-1.0/8.0 + kR2*(1.0/192.0 +
      kR2*(-1.0/9216.0 + kR2*(1.0/737280.0 +
      kR2* (-1.0/88473600.0)))));
  }
  else
    w = 2 * gsl_sf_bessel_J1(kR) / kR;
  return w;
}

// Integrand for sigmaB integral (used for the SSC covariance calculation)
static double sigma2B_integrand(double lk,void *params) {
  SigmaR_pars *par=(SigmaR_pars *)params;

  double k=pow(10.,lk);
  double pk=ccl_f2d_t_eval(par->psp, lk * M_LN10, par->a,
                           par->cosmo, par->status);
  double kR=k*par->R;
  double w = w_tophat_2d(kR);

  return pk*k*k*w*w;
}

/* --------- ROUTINE: ccl_sigmaB ---------
INPUT: cosmology, comoving smoothing radius, scale factor
TASK: compute sigmaB, the variance in the projected *linear* density field
smoothed with a 2D tophat filter of comoving size R
*/
double ccl_sigma2B(ccl_cosmology *cosmo,double R,double a,ccl_f2d_t *psp, int *status)
{
  SigmaR_pars par;
  par.status = status;

  par.cosmo=cosmo;
  par.R=R;
  par.a=a;
  par.psp=psp;

  gsl_integration_cquad_workspace *workspace =  NULL;
  gsl_function F;
  F.function=&sigma2B_integrand;
  F.params=&par;
  double sigma_B;

  workspace = gsl_integration_cquad_workspace_alloc(cosmo->gsl_params.N_ITERATION);
  if (workspace == NULL) {
    *status = CCL_ERROR_MEMORY;
  }
  if (*status == 0) {
    int gslstatus = gsl_integration_cquad(&F,
                                          log10(cosmo->spline_params.K_MIN),
                                          log10(cosmo->spline_params.K_MAX),
                                          0.0, cosmo->gsl_params.INTEGRATION_SIGMAR_EPSREL,
                                          workspace,&sigma_B,NULL,NULL);
    if(gslstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_power.c: ccl_sigma2B():");
      *status |= gslstatus;
    }
  }
  gsl_integration_cquad_workspace_free(workspace);

  return sigma_B*M_LN10/(2*M_PI);
}

void ccl_sigma2Bs(ccl_cosmology *cosmo,int na, double *a, double *R,
                  double *sigma2B_out, ccl_f2d_t *psp, int *status) {
#pragma omp parallel default(none)                      \
  shared(cosmo, na, a, R, psp, sigma2B_out, status)
  {
    int ia;
    int local_status=*status;

    #pragma omp for
    for(ia=0; ia<na; ia++) {
      if(local_status==0)
        sigma2B_out[ia]=ccl_sigma2B(cosmo,R[ia],a[ia],psp,&local_status);
    } //end omp for
    if(local_status) {
      #pragma omp atomic write
      *status=local_status;
    }
  } //end omp parallel

  if(*status) {
    ccl_cosmology_set_status_message(cosmo,
             "ccl_power.c: ccl_sigma2Bs(): "
             "integration error\n");
  }
}

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
  double res = ccl_sigmaR(cosmo, 8/cosmo->params.h, 1., psp, status);
  if (isnan(cosmo->params.sigma8)) {
    cosmo->params.sigma8 = res;
  }
  return res;
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
