#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

#include "ccl.h"

typedef struct{
  double l;
  ccl_cosmology *cosmo;
  ccl_cl_tracer_collection_t *trc1;
  ccl_cl_tracer_collection_t *trc2;
  ccl_f2d_t *psp;
  int *status;
} integ_cl_par;

static void get_k_interval(ccl_cosmology *cosmo,
			   ccl_cl_tracer_collection_t *trc1,
			   ccl_cl_tracer_collection_t *trc2,
			   double l,double *lkmin,double *lkmax)
{
  int itr;

  //Loop through all tracers and find distance bounds
  double chi_min1=1E15;
  double chi_max1=-1E15;
  for(itr=0;itr<trc1->n_tracers;itr++) {
    if(trc1->ts[itr]->chi_min<chi_min1)
      chi_min1=trc1->ts[itr]->chi_min;
    if(trc1->ts[itr]->chi_max>chi_max1)
      chi_max1=trc1->ts[itr]->chi_max;
  }
  double chi_min2=1E15;
  double chi_max2=-1E15;
  for(itr=0;itr<trc2->n_tracers;itr++) {
    if(trc2->ts[itr]->chi_min<chi_min2)
      chi_min2=trc2->ts[itr]->chi_min;
    if(trc2->ts[itr]->chi_max>chi_max2)
      chi_max2=trc2->ts[itr]->chi_max;
  }

  //Find maximum of minima and minimum of maxima
  //(i.e. edges where the product of both kernels will have support).
  double chi_min=fmax(chi_min1,chi_min2);
  double chi_max=fmin(chi_max1,chi_max2);

  if(chi_min<=0)
    chi_min=0.5*(l+0.5)/cosmo->spline_params.K_MAX;

  //Don't go beyond kmax
  *lkmax=log(fmin(cosmo->spline_params.K_MAX,2  *(l+0.5)/chi_min));
  *lkmin=log(fmax(cosmo->spline_params.K_MIN,0.5*(l+0.5)/chi_max));
}

static double transfer_limber_single(ccl_cl_tracer_t *tr,
				     double l,double lk,double k,double chi_l,double a_l,
				     ccl_cosmology *cosmo,ccl_f2d_t *psp,
				     int *status)
{
  double dd=0;
  // Kernel and transfer evaluated at chi_l
  double w=ccl_cl_tracer_t_get_kernel(tr,chi_l,status);
  double t=ccl_cl_tracer_t_get_transfer(tr,lk,a_l,status);
  double fl=ccl_cl_tracer_t_get_f_ell(tr,l,status);
  
  if(tr->der_bessel<1) { //We don't need l+1
    dd=w*t;
    if(tr->der_bessel==-1) { //If we divide by (chi*k)^2
      double lp1h=l+0.5;
      dd/=lp1h*lp1h;
    }
  }
  else { //We will need l+1
    //Compute chi_{l+1} and a_{l+1}
    double lp1h=l+0.5;
    double lp3h=l+1.5;
    double chi_lp=lp3h/k;
    double a_lp=ccl_scale_factor_of_chi(cosmo,chi_lp,status);
    
    //Compute power spectrum ratio there
    double pk_ratio=fabs(ccl_f2d_t_eval(psp,lk,a_lp,cosmo,status)/
			 ccl_f2d_t_eval(psp,lk,a_l,cosmo,status));
    //Compute kernel and trasfer at chi_{l+1}
    double w_p=ccl_cl_tracer_t_get_kernel(tr,chi_lp,status);
    double t_p=ccl_cl_tracer_t_get_transfer(tr,lk,a_lp,status);
    //sqrt(2l+1/2l+3)
    double sqell=sqrt(lp1h*pk_ratio/lp3h);
    if(tr->der_bessel==1)
      dd=l*w*t/lp1h-sqell*w_p*t_p;
    else //we assume der_bessel=2 here to avoid extra if clause
      dd=sqell*2*w_p*t_p/lp3h-(0.25+2*l)*w*t/(lp1h*lp1h);
  }
  return dd*fl;
}

static double transfer_limber_wrap(double l,double lk,double k,double chi,double a,
				   ccl_cl_tracer_collection_t *trc,
				   ccl_cosmology *cosmo,ccl_f2d_t *psp,
				   int *status)
{
  int itr;
  double transfer=0;

  for(itr=0;itr<trc->n_tracers;itr++) {
    transfer+=transfer_limber_single(trc->ts[itr],l,lk,k,chi,a,cosmo,psp,status);
    if(*status==0)
      return -1;
  }
  return transfer;
}

static double cl_integrand(double lk,void *params)
{
  double d1,d2;
  integ_cl_par *p=(integ_cl_par *)params;
  double k=exp(lk);
  double chi=(p->l+0.5)/k;
  double a=ccl_scale_factor_of_chi(p->cosmo,chi,p->status);
  d1=transfer_limber_wrap(p->l,lk,k,chi,a,p->trc1,
			  p->cosmo,p->psp,p->status);
  if(d1==0)
    return 0;
  d2=transfer_limber_wrap(p->l,lk,k,chi,a,p->trc2,
			  p->cosmo,p->psp,p->status);
  if(d2==0)
    return 0;

  double pk=ccl_f2d_t_eval(p->psp,lk,a,p->cosmo,p->status);

  return k*pk*d1*d2;
}

double ccl_angular_cl_limber(ccl_cosmology *cosmo,
			     ccl_cl_tracer_collection_t **trc1,
			     ccl_cl_tracer_collection_t **trc2,
			     ccl_f2d_t *psp,double l,int *status)
{
  //Figure out which power spectrum to use
  ccl_f2d_t *psp_use;
  if(psp==NULL) {
    if (!cosmo->computed_power) ccl_cosmology_compute_power(cosmo, status);
    // Return if computation failed
    if (!cosmo->computed_power) return NAN;
    psp_use=cosmo->data.p_nl;
  }
  else
    psp_use=psp;

  int clastatus=0;
  //Set up integrating function
  integ_cl_par ipar;
  gsl_function F;
  ipar.l=l;
  ipar.cosmo=cosmo;
  ipar.trc1=trc1;
  ipar.trc2=trc2;
  ipar.psp=psp_use;
  ipar.status=&clastatus;
  F.function=&cl_integrand;
  F.params=&ipar;

  //Get integration limits
  double lkmin,lkmax;
  get_k_interval(cosmo,trc1,trc2,l,&lkmin,&lkmax);

  //Integrate
  double result,eresult;
  int gslstatus;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(cosmo->gsl_params.N_ITERATION);
  gslstatus=gsl_integration_qag(&F, lkmin, lkmax, 0,
                                cosmo->gsl_params.INTEGRATION_LIMBER_EPSREL,
				cosmo->gsl_params.N_ITERATION,
                                cosmo->gsl_params.INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS,
                                w, &result, &eresult);
  gsl_integration_workspace_free(w);

  // Test if a round-off error occured in the evaluation of the integral
  // If so, try another integration function, more robust but potentially slower
  if(gslstatus == GSL_EROUND) {
    ccl_raise_gsl_warning(gslstatus, "ccl_cls.c: ccl_angular_cl_limber(): Default GSL integration failure, attempting backup method.");
    gsl_integration_cquad_workspace *w_cquad=gsl_integration_cquad_workspace_alloc(cosmo->gsl_params.N_ITERATION);
    size_t nevals=0;
    gslstatus=gsl_integration_cquad(&F, lkmin, lkmax, 0,
				    cosmo->gsl_params.INTEGRATION_LIMBER_EPSREL,
				    w_cquad, &result, &eresult, &nevals);
    gsl_integration_cquad_workspace_free(w_cquad);
  }
  if(gslstatus!=GSL_SUCCESS || *ipar.status) {
    ccl_raise_gsl_warning(gslstatus, "ccl_cls.c: ccl_angular_cl_limber():");
    // If an error status was already set, don't overwrite it.
    if(*status == 0){
        *status=CCL_ERROR_INTEG;
        ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_angular_cl_limber(): error integrating over k\n");
    }
    return -1;
  }
  ccl_check_status(cosmo,status);

  //Return
  return result/(l+0.5);
}

double ccl_angular_cls_nonlimber(ccl_cosmology *cosmo,
				 double l_logstep,int l_linstep,
				 ccl_cl_tracer_collection_t **trc1,
				 ccl_cl_tracer_collection_t **trc2,
				 int nl_out,int *l_out,double *cl_out,
				 int *status)
{
  *status=CCL_ERROR_INCONSISTENT;
  ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_angular_cls_nonlimber(); non-Limber integrator not implemented yet\n");
}
