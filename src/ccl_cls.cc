#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "gsl/gsl_errno.h"
#include "gsl/gsl_integration.h"
#ifdef __cplusplus
extern "C"{
#endif //__cplusplus
#include "ccl_cls.h"
#include "ccl_power.h"
#include "ccl_background.h"
#include "ccl_error.h"
#include "ccl_utils.h"
#include "ccl_params.h"
#ifdef __cplusplus
}
#endif //__cplusplus
#include "Angpow/angpow_tools.h"
#include "Angpow/angpow_parameters.h"
#include "Angpow/angpow_pk2cl.h"
#include "Angpow/angpow_powspec_base.h"
#include "Angpow/angpow_cosmo_base.h"
#include "Angpow/angpow_radial.h"
#include "Angpow/angpow_radial_base.h"
#include "Angpow/angpow_clbase.h"
#include "Angpow/angpow_ctheta.h"
#include "Angpow/angpow_exceptions.h"  //exceptions
#include "Angpow/angpow_integrand_base.h"

#define CCL_FRAC_RELEVANT 5E-4
//Gets the x-interval where the values of y are relevant
//(meaning, that the values of y for those x are at least above a fraction frac of its maximum)
static void get_support_interval(int n,double *x,double *y,double frac,
				 double *xmin_out,double *xmax_out)
{
  int ix;
  double ythr=-1000;

  //Initialize as the original edges in case we don't find an interval
  *xmin_out=x[0];
  *xmax_out=x[n-1];

  //Find threshold
  for(ix=0;ix<n;ix++) {
    if(y[ix]>ythr) ythr=y[ix];
  }
  ythr*=frac;

  //Find minimum
  for(ix=0;ix<n;ix++) {
    if(y[ix]>=ythr) {
      *xmin_out=x[ix];
      break;
    }
  }

  //Find maximum
  for(ix=n-1;ix>=0;ix--) {
    if(y[ix]>=ythr) {
      *xmax_out=x[ix];
      break;
    }
  }
}

//Spline creator
//n     -> number of points
//x     -> x-axis
//y     -> f(x)-axis
//y0,yf -> values of f(x) to use beyond the interpolation range
static SplPar *spline_init(int n,double *x,double *y,double y0,double yf)
{
  SplPar *spl=(SplPar *)malloc(sizeof(SplPar));
  if(spl==NULL)
    return NULL;
  
  spl->intacc=gsl_interp_accel_alloc();
  spl->spline=gsl_spline_alloc(gsl_interp_cspline,n);
  int parstatus=gsl_spline_init(spl->spline,x,y,n);
  if(parstatus) {
    gsl_interp_accel_free(spl->intacc);
    gsl_spline_free(spl->spline);
    return NULL;
  }

  spl->x0=x[0];
  spl->xf=x[n-1];
  spl->y0=y0;
  spl->yf=yf;

  return spl;
}

//Evaluates spline at x checking for bound errors
static double spline_eval(double x,SplPar *spl)
{
  if(x<=spl->x0)
    return spl->y0;
  else if(x>=spl->xf) 
    return spl->yf;
  else
    return gsl_spline_eval(spl->spline,x,spl->intacc);
}

//Wrapper around spline_eval with GSL function syntax
static double speval_bis(double x,void *params)
{
  return spline_eval(x,(SplPar *)params);
}

//Spline destructor
static void spline_free(SplPar *spl)
{
  gsl_spline_free(spl->spline);
  gsl_interp_accel_free(spl->intacc);
  free(spl);
}


void ccl_cl_workspace_free(CCL_ClWorkspace *w)
{
  free(w->l_arr);
  free(w);
}

CCL_ClWorkspace *ccl_cl_workspace_new(int lmax,int l_limber,int non_limber_method,
				      double l_logstep,int l_linstep,double dchi,int *status)
{
  CCL_ClWorkspace *w=(CCL_ClWorkspace *)malloc(sizeof(CCL_ClWorkspace));
  if(w==NULL) {
    *status=CCL_ERROR_MEMORY;
    //Can't access cosmology object
    //    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_workspace_new(); memory allocation\n");
    return NULL;
  }

  //Set params
  w->dchi=dchi;
  w->lmax=lmax;
  if((non_limber_method!=CCL_NONLIMBER_METHOD_NATIVE) && (non_limber_method!=CCL_NONLIMBER_METHOD_ANGPOW)) {
    free(w);
    *status=CCL_ERROR_INCONSISTENT;
    //Can't access cosmology object
    //    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_workspace_new(); unknown non-limber method\n");
    return NULL;
  }
  w->nlimb_method=non_limber_method;
  w->l_limber=l_limber;
  w->l_logstep=l_logstep;
  w->l_linstep=l_linstep;

  //Compute number of multipoles
  int i_l=0,l0=0;
  int increment=CCL_MAX(((int)(l0*(w->l_logstep-1.))),1);
  while((l0 < w->lmax) && (increment < w->l_linstep)) {
    i_l++;
    l0+=increment;
    increment=CCL_MAX(((int)(l0*(w->l_logstep-1))),1);
  }
  increment=w->l_linstep;
  while(l0 < w->lmax) {
    i_l++;
    l0+=increment;
  }

  //Allocate array of multipoles
  w->n_ls=i_l+1;
  w->l_arr=(int *)malloc(w->n_ls*sizeof(int));
  if(w->l_arr==NULL) {
    free(w);
    *status=CCL_ERROR_MEMORY;
    //Can't access cosmology object
    //    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_workspace_new(); memory allocation\n");
    return NULL;
  }

  //Redo the computation above and store values of ell
  i_l=0; l0=0;
  increment=CCL_MAX(((int)(l0*(w->l_logstep-1.))),1);
  while((l0 < w->lmax) && (increment < w->l_linstep)) {
    w->l_arr[i_l]=l0;
    i_l++;
    l0+=increment;
    increment=CCL_MAX(((int)(l0*(w->l_logstep-1))),1);
  }
  increment=w->l_linstep;
  while(l0 < w->lmax) {
    w->l_arr[i_l]=l0;
    i_l++;
    l0+=increment;
  }
  w->l_arr[i_l]=l0;

  return w;
}

CCL_ClWorkspace *ccl_cl_workspace_new_default(int lmax,int l_limber,int *status)
{
  //Default parameters: 1.05 logarithmic sampling, 20 linear sampling, native non-limber method
  return ccl_cl_workspace_new(lmax,l_limber,CCL_NONLIMBER_METHOD_NATIVE,1.05,20.,3.,status);
}

//Params for lensing kernel integrand
typedef struct {
  double chi;
  SplPar *spl_pz;
  ccl_cosmology *cosmo;
  int *status;
} IntLensPar;

//Integrand for lensing kernel
static double integrand_wl(double chip,void *params)
{
  IntLensPar *p=(IntLensPar *)params;
  double chi=p->chi;
  double a=ccl_scale_factor_of_chi(p->cosmo,chip, p->status);
  double z=1./a-1;
  double pz=spline_eval(z,p->spl_pz);
  double h=p->cosmo->params.h*ccl_h_over_h0(p->cosmo,a, p->status)/CLIGHT_HMPC;

  if(chi==0)
    return h*pz;
  else
    return h*pz*ccl_sinn(p->cosmo,chip-chi,p->status)/ccl_sinn(p->cosmo,chip,p->status);
}

//Integral to compute lensing window function
//chi     -> comoving distance
//cosmo   -> ccl_cosmology object
//spl_pz  -> normalized N(z) spline
//chi_max -> maximum comoving distance to which the integral is computed
//win     -> result is stored here
static int window_lensing(double chi,ccl_cosmology *cosmo,SplPar *spl_pz,double chi_max,double *win)
{
  int gslstatus =0, status =0;
  double result,eresult;
  IntLensPar ip;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);

  ip.chi=chi;
  ip.cosmo=cosmo;
  ip.spl_pz=spl_pz;
  ip.status = &status;
  F.function=&integrand_wl;
  F.params=&ip;
  gslstatus=gsl_integration_qag(&F,chi,chi_max,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
  *win=result;
  gsl_integration_workspace_free(w);
  if(gslstatus!=GSL_SUCCESS || *ip.status)
    return 1;
  //TODO: chi_max should be changed to chi_horizon
  //we should precompute this quantity and store it in cosmo by default

  return 0;
}

//Params for lensing kernel integrand
typedef struct {
  double chi;
  SplPar *spl_pz;
  SplPar *spl_sz;
  ccl_cosmology *cosmo;
  int *status;
} IntMagPar;

//Integrand for magnification kernel
static double integrand_mag(double chip,void *params)
{
  IntMagPar *p=(IntMagPar *)params;
//EK: added local status here as the status testing is done in routines called from this function
  int status;
  double chi=p->chi;
  double a=ccl_scale_factor_of_chi(p->cosmo,chip, p->status);
  double z=1./a-1;
  double pz=spline_eval(z,p->spl_pz);
  double sz=spline_eval(z,p->spl_sz);
  double h=p->cosmo->params.h*ccl_h_over_h0(p->cosmo,a, p->status)/CLIGHT_HMPC;

  if(chi==0)
    return h*pz*(1-2.5*sz);
  else
    return h*pz*(1-2.5*sz)*ccl_sinn(p->cosmo,chip-chi,p->status)/ccl_sinn(p->cosmo,chip,p->status);
}

//Integral to compute magnification window function
//chi     -> comoving distance
//cosmo   -> ccl_cosmology object
//spl_pz  -> normalized N(z) spline
//spl_pz  -> magnification bias s(z)
//chi_max -> maximum comoving distance to which the integral is computed
//win     -> result is stored here
static int window_magnification(double chi,ccl_cosmology *cosmo,SplPar *spl_pz,SplPar *spl_sz,
				double chi_max,double *win)
{
  int gslstatus =0, status =0;
  double result,eresult;
  IntMagPar ip;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);

  ip.chi=chi;
  ip.cosmo=cosmo;
  ip.spl_pz=spl_pz;
  ip.spl_sz=spl_sz;
  ip.status = &status;
  F.function=&integrand_mag;
  F.params=&ip;
  gslstatus=gsl_integration_qag(&F,chi,chi_max,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
  *win=result;
  gsl_integration_workspace_free(w);
  if(gslstatus!=GSL_SUCCESS || *ip.status)
    return 1;
  //TODO: chi_max should be changed to chi_horizon
  //we should precompute this quantity and store it in cosmo by default

  return 0;
}

//CCL_ClTracer creator
//cosmo   -> ccl_cosmology object
//tracer_type -> type of tracer. Supported: CL_TRACER_NC, CL_TRACER_WL
//nz_n -> number of points for N(z)
//z_n  -> array of z-values for N(z)
//n    -> corresponding N(z)-values. Normalization is irrelevant
//        N(z) will be set to zero outside the range covered by z_n
//nz_b -> number of points for b(z)
//z_b  -> array of z-values for b(z)
//b    -> corresponding b(z)-values.
//        b(z) will be assumed constant outside the range covered by z_n
static CCL_ClTracer *cl_tracer_new(ccl_cosmology *cosmo,int tracer_type,
				   int has_rsd,int has_magnification,int has_intrinsic_alignment,
				   int nz_n,double *z_n,double *n,
				   int nz_b,double *z_b,double *b,
				   int nz_s,double *z_s,double *s,
				   int nz_ba,double *z_ba,double *ba,
				   int nz_rf,double *z_rf,double *rf, int * status)
{
  int clstatus=0;
  CCL_ClTracer *clt=(CCL_ClTracer *)malloc(sizeof(CCL_ClTracer));
  if(clt==NULL) {
    *status=CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): memory allocation\n");
    return NULL;
  }
  clt->tracer_type=tracer_type;

  double hub=cosmo->params.h*ccl_h_over_h0(cosmo,1.,status)/CLIGHT_HMPC;
  clt->prefac_lensing=1.5*hub*hub*cosmo->params.Omega_m;

  if((tracer_type==CL_TRACER_NC)||(tracer_type==CL_TRACER_WL)) {
    get_support_interval(nz_n,z_n,n,CCL_FRAC_RELEVANT,&(clt->zmin),&(clt->zmax));
    clt->chimax=ccl_comoving_radial_distance(cosmo,1./(1+clt->zmax),status);
    clt->chimin=ccl_comoving_radial_distance(cosmo,1./(1+clt->zmin),status);
    clt->spl_nz=spline_init(nz_n,z_n,n,0,0);
    if(clt->spl_nz==NULL) {
      free(clt);
      *status=CCL_ERROR_SPLINE;
      strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error initializing spline for N(z)\n");
      return NULL;
    }

    //Normalize n(z)
    gsl_function F;
    double nz_norm,nz_enorm;
    double *nz_normalized=(double *)malloc(nz_n*sizeof(double));
    if(nz_normalized==NULL) {
      spline_free(clt->spl_nz);
      free(clt);
      *status=CCL_ERROR_MEMORY;
      strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): memory allocation\n");
      return NULL;
    }

    gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);
    F.function=&speval_bis;
    F.params=clt->spl_nz;
    clstatus=gsl_integration_qag(&F,z_n[0],z_n[nz_n-1],0,1E-4,1000,GSL_INTEG_GAUSS41,w,&nz_norm,&nz_enorm);
    gsl_integration_workspace_free(w); //TODO:check for integration errors
    if(clstatus!=GSL_SUCCESS) {
      spline_free(clt->spl_nz);
      free(clt);
      *status=CCL_ERROR_INTEG;
      strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): integration error when normalizing N(z)\n");
      return NULL;
    }
    for(int ii=0;ii<nz_n;ii++)
      nz_normalized[ii]=n[ii]/nz_norm;
    spline_free(clt->spl_nz);
    clt->spl_nz=spline_init(nz_n,z_n,nz_normalized,0,0);
    free(nz_normalized);
    if(clt->spl_nz==NULL) {
      free(clt);
      *status=CCL_ERROR_SPLINE;
      strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error initializing normalized spline for N(z)\n");
      return NULL;
    }

    if(tracer_type==CL_TRACER_NC) {
      //Initialize bias spline
      clt->spl_bz=spline_init(nz_b,z_b,b,b[0],b[nz_b-1]);
      if(clt->spl_bz==NULL) {
	spline_free(clt->spl_nz);
	free(clt);
	*status=CCL_ERROR_SPLINE;
	strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error initializing spline for b(z)\n");
	return NULL;
      }
      clt->has_rsd=has_rsd;
      clt->has_magnification=has_magnification;
      if(clt->has_magnification) {
	//Compute weak lensing kernel
	int nchi;
	double *x,*y;
	double dchi=5.;
	double zmax=clt->spl_nz->xf;
	double chimax=ccl_comoving_radial_distance(cosmo,1./(1+zmax),status);
	//TODO: The interval in chi (5. Mpc) should be made a macro

	//In this case we need to integrate all the way to z=0. Reset zmin and chimin
	clt->zmin=0;
	clt->chimin=0;
	clt->spl_sz=spline_init(nz_s,z_s,s,s[0],s[nz_s-1]);
	if(clt->spl_sz==NULL) {
	  spline_free(clt->spl_nz);
	  spline_free(clt->spl_bz);
	  free(clt);
	  *status=CCL_ERROR_SPLINE;
	  strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error initializing spline for s(z)\n");
	  return NULL;
	}

	nchi=(int)(chimax/dchi)+1;
	x=ccl_linear_spacing(0.,chimax,nchi);
	dchi=chimax/nchi;
	if(x==NULL || (fabs(x[0]-0)>1E-5) || (fabs(x[nchi-1]-chimax)>1e-5)) {
	  spline_free(clt->spl_nz);
	  spline_free(clt->spl_bz);
	  spline_free(clt->spl_sz);
	  free(clt);
	  *status=CCL_ERROR_LINSPACE;
	  strcpy(cosmo->status_message,
		 "ccl_cls.c: ccl_cl_tracer_new(): Error creating linear spacing in chi\n");
	  return NULL;
	}
	y=(double *)malloc(nchi*sizeof(double));
	if(y==NULL) {
	  free(x);
	  spline_free(clt->spl_nz);
	  spline_free(clt->spl_bz);
	  spline_free(clt->spl_sz);
	  free(clt);
	  *status=CCL_ERROR_MEMORY;
	  strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): memory allocation\n");
	  return NULL;
	}
      
	for(int j=0;j<nchi;j++)
	  clstatus|=window_magnification(x[j],cosmo,clt->spl_nz,clt->spl_sz,chimax,&(y[j]));
	if(clstatus) {
	  free(y);
	  free(x);
	  spline_free(clt->spl_nz);
	  spline_free(clt->spl_bz);
	  spline_free(clt->spl_sz);
	  free(clt);
	  *status=CCL_ERROR_INTEG;
	  strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error computing lensing window\n");
	  return NULL;
	}

	clt->spl_wM=spline_init(nchi,x,y,y[0],0);
	if(clt->spl_wM==NULL) {
	  free(y);
	  free(x);
	  spline_free(clt->spl_nz);
	  spline_free(clt->spl_bz);
	  spline_free(clt->spl_sz);
	  free(clt);
	  *status=CCL_ERROR_SPLINE;
	  strcpy(cosmo->status_message,
		 "ccl_cls.c: ccl_cl_tracer_new(): error initializing spline for lensing window\n");
	  return NULL;
	}
	free(x); free(y);
      }
    }
    else if(tracer_type==CL_TRACER_WL) {
      //Compute weak lensing kernel
      int nchi;
      double *x,*y;
      double dchi=5.;
      double zmax=clt->spl_nz->xf;
      double chimax=ccl_comoving_radial_distance(cosmo,1./(1+zmax),status);
      //TODO: The interval in chi (5. Mpc) should be made a macro

      //In this case we need to integrate all the way to z=0. Reset zmin and chimin
      clt->zmin=0;
      clt->chimin=0;
      nchi=(int)(chimax/dchi)+1;
      x=ccl_linear_spacing(0.,chimax,nchi);
      dchi=chimax/nchi;
      if(x==NULL || (fabs(x[0]-0)>1E-5) || (fabs(x[nchi-1]-chimax)>1e-5)) {
	spline_free(clt->spl_nz);
	free(clt);
	*status=CCL_ERROR_LINSPACE;
	strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): Error creating linear spacing in chi\n");
	return NULL;
      }
      y=(double *)malloc(nchi*sizeof(double));
      if(y==NULL) {
	free(x);
	spline_free(clt->spl_nz);
	free(clt);
	*status=CCL_ERROR_MEMORY;
	strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): memory allocation\n");
	return NULL;
      }
      
      for(int j=0;j<nchi;j++)
	clstatus|=window_lensing(x[j],cosmo,clt->spl_nz,chimax,&(y[j]));
      if(clstatus) {
	free(y);
	free(x);
	spline_free(clt->spl_nz);
	free(clt);
	*status=CCL_ERROR_INTEG;
	strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error computing lensing window\n");
	return NULL;
      }

      clt->spl_wL=spline_init(nchi,x,y,y[0],0);
      if(clt->spl_wL==NULL) {
	free(y);
	free(x);
	spline_free(clt->spl_nz);
	free(clt);
	*status=CCL_ERROR_SPLINE;
	strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error initializing spline for lensing window\n");
	return NULL;
      }
      free(x); free(y);
      
      clt->has_intrinsic_alignment=has_intrinsic_alignment;
      if(clt->has_intrinsic_alignment) {
	clt->spl_rf=spline_init(nz_rf,z_rf,rf,rf[0],rf[nz_rf-1]);
	if(clt->spl_rf==NULL) {
	  spline_free(clt->spl_nz);
	  free(clt);
	  *status=CCL_ERROR_SPLINE;
	  strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error initializing spline for rf(z)\n");
	  return NULL;
	}
	clt->spl_ba=spline_init(nz_ba,z_ba,ba,ba[0],ba[nz_ba-1]);
	if(clt->spl_ba==NULL) {
	  spline_free(clt->spl_rf);
	  spline_free(clt->spl_nz);
	  free(clt);
	  *status=CCL_ERROR_SPLINE;
	  strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error initializing spline for ba(z)\n");
	  return NULL;
	}
      }
    }
  }
  else {
    *status=CCL_ERROR_INCONSISTENT;
    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): unknown tracer type\n");
    return NULL;
  }

  return clt;
}

//CCL_ClTracer constructor with error checking
//cosmo   -> ccl_cosmology object
//tracer_type -> type of tracer. Supported: CL_TRACER_NC, CL_TRACER_WL
//nz_n -> number of points for N(z)
//z_n  -> array of z-values for N(z)
//n    -> corresponding N(z)-values. Normalization is irrelevant
//        N(z) will be set to zero outside the range covered by z_n
//nz_b -> number of points for b(z)
//z_b  -> array of z-values for b(z)
//b    -> corresponding b(z)-values.
//        b(z) will be assumed constant outside the range covered by z_n
CCL_ClTracer *ccl_cl_tracer_new(ccl_cosmology *cosmo,int tracer_type,
				int has_rsd,int has_magnification,int has_intrinsic_alignment,
				int nz_n,double *z_n,double *n,
				int nz_b,double *z_b,double *b,
				int nz_s,double *z_s,double *s,
				int nz_ba,double *z_ba,double *ba,
				int nz_rf,double *z_rf,double *rf, int * status)
{
  CCL_ClTracer *clt=cl_tracer_new(cosmo,tracer_type,has_rsd,has_magnification,has_intrinsic_alignment,
				  nz_n,z_n,n,nz_b,z_b,b,nz_s,z_s,s,nz_ba,z_ba,ba,nz_rf,z_rf,rf, status);
  ccl_check_status(cosmo,status);
  return clt;
}

//CCL_ClTracer destructor
void ccl_cl_tracer_free(CCL_ClTracer *clt)
{
  spline_free(clt->spl_nz);
  if(clt->tracer_type==CL_TRACER_NC) {
    spline_free(clt->spl_bz);
    if(clt->has_magnification) {
      spline_free(clt->spl_sz);
      spline_free(clt->spl_wM);
    }
  }
  else if(clt->tracer_type==CL_TRACER_WL) {
    spline_free(clt->spl_wL);
    if(clt->has_intrinsic_alignment) {
      spline_free(clt->spl_ba);
      spline_free(clt->spl_rf);
    }
  }
  free(clt);
}

CCL_ClTracer *ccl_cl_tracer_number_counts_new(ccl_cosmology *cosmo,
					      int has_rsd,int has_magnification,
					      int nz_n,double *z_n,double *n,
					      int nz_b,double *z_b,double *b,
					      int nz_s,double *z_s,double *s, int * status)
{
  return ccl_cl_tracer_new(cosmo,CL_TRACER_NC,has_rsd,has_magnification,0,
			   nz_n,z_n,n,nz_b,z_b,b,nz_s,z_s,s,
			   -1,NULL,NULL,-1,NULL,NULL, status);
}

CCL_ClTracer *ccl_cl_tracer_number_counts_simple_new(ccl_cosmology *cosmo,
						     int nz_n,double *z_n,double *n,
						     int nz_b,double *z_b,double *b, int * status)
{
  return ccl_cl_tracer_new(cosmo,CL_TRACER_NC,0,0,0,
			   nz_n,z_n,n,nz_b,z_b,b,-1,NULL,NULL,
			   -1,NULL,NULL,-1,NULL,NULL, status);
}

CCL_ClTracer *ccl_cl_tracer_lensing_new(ccl_cosmology *cosmo,
					int has_alignment,
					int nz_n,double *z_n,double *n,
					int nz_ba,double *z_ba,double *ba,
					int nz_rf,double *z_rf,double *rf, int * status)
{
  return ccl_cl_tracer_new(cosmo,CL_TRACER_WL,0,0,has_alignment,
			   nz_n,z_n,n,-1,NULL,NULL,-1,NULL,NULL,
			   nz_ba,z_ba,ba,nz_rf,z_rf,rf, status);
}

CCL_ClTracer *ccl_cl_tracer_lensing_simple_new(ccl_cosmology *cosmo,
					       int nz_n,double *z_n,double *n, int * status)
{
  return ccl_cl_tracer_new(cosmo,CL_TRACER_WL,0,0,0,
			   nz_n,z_n,n,-1,NULL,NULL,-1,NULL,NULL,
			   -1,NULL,NULL,-1,NULL,NULL, status);
}

static double j_bessel_limber(int l,double k)
{
  return sqrt(M_PI/(2*l+1.))/k;
}

static double f_dens(double a,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double z=1./a-1;
  double pz=spline_eval(z,clt->spl_nz);
  double bz=spline_eval(z,clt->spl_bz);
  double h=cosmo->params.h*ccl_h_over_h0(cosmo,a,status)/CLIGHT_HMPC;

  return pz*bz*h;
}

static double f_rsd(double a,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double z=1./a-1;
  double pz=spline_eval(z,clt->spl_nz);
  double fg=ccl_growth_rate(cosmo,a,status);
  double h=cosmo->params.h*ccl_h_over_h0(cosmo,a,status)/CLIGHT_HMPC;

  return pz*fg*h;
}

static double f_mag(double a,double chi,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double wM=spline_eval(chi,clt->spl_wM);
  
  if(wM<=0)
    return 0;
  else
    return wM/(a*chi);
}

//Transfer function for number counts
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//w -> CCL_ClWorskpace object
//clt -> CCL_ClTracer object (must be of the CL_TRACER_NC type)
static double transfer_nc(int l,double k,
			  ccl_cosmology *cosmo,CCL_ClWorkspace *w,CCL_ClTracer *clt, int * status)
{
  double ret=0;
  if(l>w->l_limber) {
    double x0=(l+0.5);
    double chi0=x0/k;
    if(chi0<=clt->chimax) {
      double a0=ccl_scale_factor_of_chi(cosmo,chi0,status);
      double pk0=ccl_nonlin_matter_power(cosmo,k,a0,status);
      double jl0=j_bessel_limber(l,k);
      double f_all=f_dens(a0,cosmo,clt,status)*jl0;
      if(clt->has_rsd) {
	double x1=(l+1.5);
	double chi1=x1/k;
	if(chi1<=clt->chimax) {
	  double a1=ccl_scale_factor_of_chi(cosmo,chi1,status);
	  double pk1=ccl_nonlin_matter_power(cosmo,k,a1,status);
	  double fg0=f_rsd(a0,cosmo,clt,status);
	  double fg1=f_rsd(a1,cosmo,clt,status);
	  double jl1=j_bessel_limber(l+1,k);
	  f_all+=fg0*(1.-l*(l-1.)/(x0*x0))*jl0-fg1*2.*jl1*sqrt(pk1/pk0)/x1;
	}
      }
      if(clt->has_magnification)
      	f_all+=-2*clt->prefac_lensing*l*(l+1)*f_mag(a0,chi0,cosmo,clt,status)*jl0/(k*k);
      ret=f_all*sqrt(pk0);
    }
  }
  else {
    int i,nchi=(int)((clt->chimax-clt->chimin)/w->dchi)+1;
    for(i=0;i<nchi;i++) {
      double chi=clt->chimin+w->dchi*(i+0.5);
      if(chi<=clt->chimax) {
	double a=ccl_scale_factor_of_chi(cosmo,chi,status);
	double pk=ccl_nonlin_matter_power(cosmo,k,a,status);
	double jl=ccl_j_bessel(l,k*chi);
	double f_all=f_dens(a,cosmo,clt,status)*jl;
	if(clt->has_rsd) {
	  double ddjl,x=k*chi;
	  if(x<1E-10) {
	    if(l==0) ddjl=0.3333-0.1*x*x;
	    else if(l==2) ddjl=-0.13333333333+0.05714285714285714*x*x;
	    else ddjl=0;
          }
	  else {
	    double jlp1=ccl_j_bessel(l+1,x);
	    ddjl=((x*x-l*(l-1))*jl-2*x*jlp1)/(x*x);
	  }
	  f_all+=f_rsd(a,cosmo,clt,status)*ddjl;
	}
	if(clt->has_magnification)
	  f_all+=-2*clt->prefac_lensing*l*(l+1)*f_mag(a,chi,cosmo,clt,status)*jl/(k*k);
	
	ret+=f_all*sqrt(pk); //TODO: is it worth splining this sqrt?
      }
    }
    ret*=w->dchi;
  }

  return ret;
}

static double f_lensing(double a,double chi,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double wL=spline_eval(chi,clt->spl_wL);
  
  if(wL<=0)
    return 0;
  else
    return clt->prefac_lensing*wL/(a*chi);
}

static double f_IA_NLA(double a,double chi,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  if(chi<=1E-10)
    return 0;
  else {
    double a=ccl_scale_factor_of_chi(cosmo,chi, status);
    double z=1./a-1;
    double pz=spline_eval(z,clt->spl_nz);
    double ba=spline_eval(z,clt->spl_ba);
    double rf=spline_eval(z,clt->spl_rf);
    double h=cosmo->params.h*ccl_h_over_h0(cosmo,a,status)/CLIGHT_HMPC;
    
    return pz*ba*rf*h/(chi*chi);
  }
}


//Transfer function for shear
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//w -> CCL_ClWorskpace object
//clt -> CCL_ClTracer object (must be of the CL_TRACER_WL type)
static double transfer_wl(int l,double k,
			  ccl_cosmology *cosmo,CCL_ClWorkspace *w,CCL_ClTracer *clt, int * status)
{
  double ret=0;
  if(l>w->l_limber) {
    double chi=(l+0.5)/k;
    if(chi<=clt->chimax) {
      double a=ccl_scale_factor_of_chi(cosmo,chi,status);
      double pk=ccl_nonlin_matter_power(cosmo,k,a,status);
      double jl=j_bessel_limber(l,k);
      double f_all=f_lensing(a,chi,cosmo,clt,status)*jl;
      if(clt->has_intrinsic_alignment)
	f_all+=f_IA_NLA(a,chi,cosmo,clt,status)*jl;

      ret=f_all*sqrt(pk);
    }
  }
  else {
    int i,nchi=(int)((clt->chimax-clt->chimin)/w->dchi)+1;
    for(i=0;i<nchi;i++) {
      double chi=clt->chimin+w->dchi*(i+0.5);
      if(chi<=clt->chimax) {
	double a=ccl_scale_factor_of_chi(cosmo,chi,status);
	double pk=ccl_nonlin_matter_power(cosmo,k,a,status);
	double jl=ccl_j_bessel(l,k*chi);
	double f_all=f_lensing(a,chi,cosmo,clt,status)*jl;
	if(clt->has_intrinsic_alignment)
	  f_all+=f_IA_NLA(a,chi,cosmo,clt,status)*jl;
	
	ret+=f_all*sqrt(pk); //TODO: is it worth splining this sqrt?
      }
    }
    ret*=w->dchi;
  }

  //  return sqrt((l+2.)*(l+1.)*l*(l-1.))*ret/(k*k);
  return (l+1.)*l*ret/(k*k);
}

//Wrapper for transfer function
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//clt -> CCL_ClTracer object
static double transfer_wrap(int l,double k,ccl_cosmology *cosmo,
			    CCL_ClWorkspace *w,CCL_ClTracer *clt, int * status)
{
  double transfer_out=0;

  if(clt->tracer_type==CL_TRACER_NC)
    transfer_out=transfer_nc(l,k,cosmo,w,clt,status);
  else if(clt->tracer_type==CL_TRACER_WL)
    transfer_out=transfer_wl(l,k,cosmo,w,clt,status);
  else
    transfer_out=-1;
  return transfer_out;
}

//Params for power spectrum integrand
typedef struct {
  int l;
  ccl_cosmology *cosmo;
  CCL_ClWorkspace *w;
  CCL_ClTracer *clt1;
  CCL_ClTracer *clt2;
  int *status;
} IntClPar;

//Integrand for integral power spectrum
static double cl_integrand(double lk,void *params)
{
  double d1,d2;
  IntClPar *p=(IntClPar *)params;
  double k=pow(10.,lk);
  d1=transfer_wrap(p->l,k,p->cosmo,p->w,p->clt1,p->status);
  d2=transfer_wrap(p->l,k,p->cosmo,p->w,p->clt2,p->status);

  return k*k*k*d1*d2;
}

//Figure out k intervals where the Limber kernel has support
//clt1 -> tracer #1
//clt2 -> tracer #2
//l    -> angular multipole
//lkmin, lkmax -> log10 of the range of scales where the transfer functions have support
static void get_k_interval(ccl_cosmology *cosmo,CCL_ClWorkspace *w,
			   CCL_ClTracer *clt1,CCL_ClTracer *clt2,int l,
			   double *lkmin,double *lkmax)
{
  double chimin,chimax;
  int cut_low_1=0,cut_low_2=0;

  //Define a minimum distance only if no lensing is needed
  if((clt1->tracer_type==CL_TRACER_NC) && (clt1->has_magnification==0)) cut_low_1=1;
  if((clt2->tracer_type==CL_TRACER_NC) && (clt2->has_magnification==0)) cut_low_2=1;

  if(l<w->l_limber) {
    chimin=2*(l+0.5)/ccl_splines->K_MAX;
    chimax=0.5*(l+0.5)/ccl_splines->K_MIN_DEFAULT;
  }
  else {
    if(cut_low_1) {
      if(cut_low_2) {
	chimin=fmax(clt1->chimin,clt2->chimin);
	chimax=fmin(clt1->chimax,clt2->chimax);
      }
      else {
	chimin=clt1->chimin;
	chimax=clt1->chimax;
      }
    }
    else if(cut_low_2) {
      chimin=clt2->chimin;
      chimax=clt2->chimax;
    }
    else {
      chimin=0.5*(l+0.5)/ccl_splines->K_MAX;
      chimax=2*(l+0.5)/ccl_splines->K_MIN_DEFAULT;
    }
  }

  if(chimin<=0)
    chimin=0.5*(l+0.5)/ccl_splines->K_MAX;

  *lkmax=fmin( 2,log10(2  *(l+0.5)/chimin));
  *lkmin=fmax(-4,log10(0.5*(l+0.5)/chimax));
}

//Compute angular power spectrum between two bins
//cosmo -> ccl_cosmology object
//l -> angular multipole
//clt1 -> tracer #1
//clt2 -> tracer #2
static double ccl_angular_cl_native(ccl_cosmology *cosmo,CCL_ClWorkspace *cw,int l,
				    CCL_ClTracer *clt1,CCL_ClTracer *clt2,int * status)
{
  int clastatus=0, qagstatus;
  IntClPar ipar;
  double result=0,eresult;
  double lkmin,lkmax;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);

  ipar.l=l;
  ipar.cosmo=cosmo;
  ipar.w=cw;
  ipar.clt1=clt1;
  ipar.clt2=clt2;
  ipar.status = &clastatus;
  F.function=&cl_integrand;
  F.params=&ipar;
  get_k_interval(cosmo,cw,clt1,clt2,l,&lkmin,&lkmax);
  qagstatus=gsl_integration_qag(&F,lkmin,lkmax,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
  gsl_integration_workspace_free(w);
  if(qagstatus!=GSL_SUCCESS || *ipar.status) {
    *status=CCL_ERROR_INTEG;
    strcpy(cosmo->status_message,"ccl_cls.c: ccl_angular_cls(): error integrating over k\n");
    return -1;
  }
  ccl_check_status(cosmo,status);

  return result*M_LN10*2./M_PI;
}

//! Here CCL is passed to Angpow base classes
namespace Angpow {

  //! Selection window W(z) with spline input from CCL 
  class RadSplineSelect : public RadSelectBase {
  public:
    RadSplineSelect(SplPar* spl,double x0,double xf): 
    RadSelectBase(x0, xf), spl_(spl) {}
    
    
    virtual r_8 operator()(r_8 z) const {
      return spline_eval(z,spl_);
    }
    
  private:
    SplPar* spl_;
  };//RadSplineSelect
  

  //! This class import the cosmology from CCL to make the conversion z <-> r(z)
  //! and then to compute the cut-off in the integrals
  class CosmoCoordCCL : public CosmoCoordBase {
  public:
    //! Ctor
    CosmoCoordCCL(ccl_cosmology * cosmo)//, double zmin=0., double zmax=9., size_t npts=1000)
    : ccl_cosmo_(cosmo) //, zmin_(zmin), zmax_(zmax), npts_(npts) 
    {
    }//Ctor
    
    //! Dtor
    virtual ~CosmoCoordCCL() {}
    
    //! r(z): radial comoving distance Mpc
    inline virtual double r(double z) const {
      int status=0;
      return ccl_comoving_radial_distance(ccl_cosmo_,1./(1+z),&status);
    }
    //! z(r): the inverse of radial comoving distance (Mpc)
    inline virtual double z(double r) const {
      int status=0;
      return 1./ccl_scale_factor_of_chi(ccl_cosmo_,r,&status)-1.;
    }
    
  protected:
    ccl_cosmology* ccl_cosmo_;  //!< access to CCL cosmology
  };// CosmoCoordCCL

  //! Base class of half the integrand function
  //! Basically Angpow does three integrals of the form
  //!    C_ell = int dz1 int dz2 int dk f1(ell,k,z1)*f2(ell,k,z2)
  //! This class set the f1 and f2 functions from CCL_ClTracer class
  //! for galaxy counts :
  //!    f(ell,k,z) = k*sqrt(P(k,z))*[ b(z)*j_ell(r(z)*k) + f(z)*j"_ell(r(z)*k) ]
  class IntegrandCCL : public IntegrandBase {
  public:
    IntegrandCCL(CCL_ClTracer* clt, ccl_cosmology* cosmo, int ell=0, r_8 z=0.):
      clt_(clt), cosmo_(cosmo), ell_(ell), z_(z) {
      Init(ell,z);
    }
    //! Initialize the function f(ell,k,z) at a given ell and z
    //! to be integrated over k
    void Init(int ell, r_8 z){
      int status=0;
      ell_=ell; z_=z;
      R_=ccl_comoving_radial_distance(cosmo_,1.0/(1+z),&status);
      jlR_ = new JBess1(ell_,R_);
      if(clt_->has_rsd) {
	jlp1R_ = new JBess1(ell_+1,R_);
	// WARNING: here we want to store dlnD/dln(+1z) = - dlnD/dlna
	fz_= - ccl_growth_rate(cosmo_,1.0/(1+z), &status);
      }
      bz_ = spline_eval(z,clt_->spl_bz);
    }
    virtual ~IntegrandCCL() {}
    //! Return f(ell,k,z) for a given k
    //! (ell and z must be initialized before) 
    virtual r_8 operator()(r_8 k) const {
      int status=0;
      r_8 Pk=ccl_nonlin_matter_power(cosmo_,k,1./(1+z_),&status);
      r_8 x = k*R_;
      r_8 jlRk = (*jlR_)(k);
      r_8 delta = bz_*jlRk; // density term with bias
      if(clt_->has_rsd){ // RSD term
	r_8 jlRksecond = 0.;
	if(x<1e-40) { // compute second derivative j"_ell(r(z)*k)
	  if(ell_==0) {
	    jlRksecond = -1./3. + x*x/10.;
	  } else if(ell_==2) {
	    jlRksecond = 2./15. - 2*x*x/35.;
	  } else {
	    jlRksecond = 0.;
	  }
	} else {
	  jlRksecond = 2.*(*jlp1R_)(k)/x + (ell_*(ell_-1.)/(x*x) - 1.)*jlRk;
	}
	delta += fz_*jlRksecond;
      }
      return(k*sqrt(fabs(Pk))*delta);
    }
    //! Clone function for OpenMP integration
    virtual IntegrandCCL* clone() const {
      return new IntegrandCCL(static_cast<const IntegrandCCL&>(*this));
    }
    virtual void ExplicitDestroy() {
      if(jlR_) delete jlR_;
      if(jlp1R_) delete jlp1R_;
    }
  private:
    CCL_ClTracer* clt_;  //no ownership
    ccl_cosmology* cosmo_;  //no ownership
    int ell_;  // multipole ell
    r_8 z_;   // redshift z
    r_8 R_;   // radial comoving distance r(z)
    r_8 fz_;  // growth rate f(z)
    r_8 bz_;  // bias b(z)
    JBess1* jlR_;  // j_ell(k*R)
    JBess1* jlp1R_;   // j_(ell+1)(k*R)
    
    //Minimal copy to allow Main operator(int, r_8, r_8) to work
    //JEC 22/4/17 use cloning of PowerSpectrum
    IntegrandCCL(const IntegrandCCL& copy) : clt_(copy.clt_),
					     cosmo_(copy.cosmo_), ell_(0), z_(0), jlR_(0), jlp1R_(0){} 

  };//IntegrandCCL
}//end namespace

//Compute angular power spectrum given two ClTracers from ell=0 to lmax
//ccl_cosmo -> CCL cosmology (for P(k) and distances)
//lmax -> maximum angular multipole
//clt1 -> tracer #1
//clt2 -> tracer #2
//status -> status
static void ccl_angular_cls_angpow(ccl_cosmology *ccl_cosmo,CCL_ClWorkspace *w,
				   CCL_ClTracer *clt1,CCL_ClTracer *clt2,
				   double *cl_out,int * status)
{
  // Initialize the Angpow parameters
  int chebyshev_order_1=9; //TODO ANGPOW: we should figure out if this is good enough
  int chebyshev_order_2=9;
  double cl_kmax=1.; //TODO ANGPOW: this shouldn't be hard-coded 
  int nsamp_z_1=(int)((clt1->chimax-clt1->chimin)/w->dchi)+1; //TODO ANGPOW: we need to figure out if this is good enough
  int nsamp_z_2=(int)((clt2->chimax-clt2->chimin)/w->dchi)+1;
  int l_max_use=CCL_MIN(w->l_limber,w->lmax);

  // Initialize the radial selection windows W(z)
  Angpow::RadSplineSelect Z1win(clt1->spl_nz,clt1->zmin,clt1->zmax);
  Angpow::RadSplineSelect Z2win(clt2->spl_nz,clt2->zmin,clt2->zmax);

  // The cosmological distance tool to make the conversion z <-> r(z)
  Angpow::CosmoCoordCCL cosmo(ccl_cosmo);

  // Initilaie the two integrand functions f(ell,k,z)
  Angpow::IntegrandCCL int1(clt1,ccl_cosmo);
  Angpow::IntegrandCCL int2(clt2,ccl_cosmo);

  // Initialize the Cl with parameters to select the ell set which is interpolated after the processing
  Angpow::Clbase clout(l_max_use+1,w->l_linstep,w->l_logstep);
  // Check Angpow's ells match those of CCL (maybe we could to pass those ells explicitly to Angpow)
  for(int index_l=0; index_l<clout.Size(); index_l++) {
    if(clout[index_l].first!=w->l_arr[index_l]) {
      *status=CCL_ERROR_ANGPOW;
      strcpy(ccl_cosmo->status_message,"ccl_cls.c: ccl_angular_cls_angpow(); "
	     "ell-bins defined in angpow don't match those of CCL\n");
      return;
    }
  }

  // Main class to compute Cl with Angpow
  Angpow::Pk2Cl pk2cl; //Default: the user parameters are used in the Constructor 
  pk2cl.SetOrdFunc(chebyshev_order_1,chebyshev_order_2);
  pk2cl.SetRadOrder(nsamp_z_1/2,nsamp_z_2/2);
  pk2cl.SetKmax(cl_kmax);
  pk2cl.Compute(int1,int2,cosmo,&Z1win,&Z2win,clout[clout.Size()-1].first+1,clout);

  // Pass the Clbase class values (ell and C_ell) to the output spline
  for(int index_l=0; index_l<clout.Size(); index_l++)
    cl_out[index_l]=clout[index_l].second;
}

void ccl_angular_cls(ccl_cosmology *cosmo,CCL_ClWorkspace *w,
		     CCL_ClTracer *clt1,CCL_ClTracer *clt2,
		     int nl_out,int *l_out,double *cl_out,int *status)
{
  int ii;

  //First check if ell range is within workspace
  for(ii=0;ii<nl_out;ii++) {
    if(l_out[ii]>w->lmax) {
      *status=CCL_ERROR_SPLINE_EV;
      strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_angular_cls(); "
	     "requested l beyond range allowed by workspace\n");
      return;
    }
  }

  //Allocate array for power spectrum at interpolation nodes
  double *l_nodes=(double *)malloc(w->n_ls*sizeof(double));
  if(l_nodes==NULL) {
    *status=CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_angular_cls(); memory allocation\n");
    return;
  }
  double *cl_nodes=(double *)malloc(w->n_ls*sizeof(double));
  if(cl_nodes==NULL) {
    free(l_nodes);
    *status=CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_angular_cls(); memory allocation\n");
    return;
  }
  for(ii=0;ii<w->n_ls;ii++)
    l_nodes[ii]=(double)(w->l_arr[ii]);

  //Now check if angpow is needed at all
  int method_use=w->nlimb_method;
  if(method_use==CCL_NONLIMBER_METHOD_ANGPOW) {
    int do_angpow=0;
    for(ii=0;ii<w->n_ls;ii++) {
      if(w->l_arr[ii]<=w->l_limber)
	do_angpow=1;
    }
    //Resort to native method if we have lensing (this will hopefully only be temporary)
    if(clt1->tracer_type==CL_TRACER_WL || clt2->tracer_type==CL_TRACER_WL ||
       clt1->has_magnification || clt2->has_magnification) {
      do_angpow=0;
      method_use=CCL_NONLIMBER_METHOD_NATIVE;
    }

    //Use angpow if non-limber is needed
    if(do_angpow)
      ccl_angular_cls_angpow(cosmo,w,clt1,clt2,cl_nodes,status);
    ccl_check_status(cosmo,status);
  }

  //Compute limber nodes
  for(ii=0;ii<w->n_ls;ii++) {
    if((method_use==CCL_NONLIMBER_METHOD_NATIVE) || (w->l_arr[ii]>w->l_limber))
      cl_nodes[ii]=ccl_angular_cl_native(cosmo,w,w->l_arr[ii],clt1,clt2,status);
  }

  //Interpolate into ells requested by user
  SplPar *spcl_nodes=spline_init(w->n_ls,l_nodes,cl_nodes,0,0);
  if(spcl_nodes==NULL) {
    free(cl_nodes);
    *status=CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_angular_cls(); memory allocation\n");
    return;
  }
  for(ii=0;ii<nl_out;ii++)
    cl_out[ii]=spline_eval((double)(l_out[ii]),spcl_nodes);

  //Cleanup
  spline_free(spcl_nodes);
  free(cl_nodes);
  free(l_nodes);
}
