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

//Spline creator
//n     -> number of points
//x     -> x-axis
//y     -> f(x)-axis
//y0,yf -> values of f(x) to use beyond the interpolation range
SplPar *spline_init(int n,double *x,double *y,double y0,double yf)
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
double spline_eval(double x,SplPar *spl)
{
  if(x<=spl->x0)
    return spl->y0;
  else if(x>=spl->xf) 
    return spl->yf;
  else
    return gsl_spline_eval(spl->spline,x,spl->intacc);
}

//Wrapper around spline_eval with GSL function syntax
double speval_bis(double x,void *params)
{
  return spline_eval(x,(SplPar *)params);
}

//Spline destructor
void spline_free(SplPar *spl)
{
  gsl_spline_free(spl->spline);
  gsl_interp_accel_free(spl->intacc);
  free(spl);
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
    clt->chimax=ccl_comoving_radial_distance(cosmo,1./(1+z_n[nz_n-1]), status);
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

	clt->spl_sz=spline_init(nz_s,z_s,s,s[0],s[nz_s-1]);
	if(clt->spl_sz==NULL) {
	  spline_free(clt->spl_nz);
	  spline_free(clt->spl_bz);
	  free(clt);
	  *status=CCL_ERROR_SPLINE;
	  strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error initializing spline for s(z)\n");
	  return NULL;
	}

	clt->chimin=0;
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
      clt->chimin=ccl_comoving_radial_distance(cosmo,1./(1+z_n[0]),status);
    }
    else if(tracer_type==CL_TRACER_WL) {
      //Compute weak lensing kernel
      int nchi;
      double *x,*y;
      double dchi=5.;
      double zmax=clt->spl_nz->xf;
      double chimax=ccl_comoving_radial_distance(cosmo,1./(1+zmax),status);
      //TODO: The interval in chi (5. Mpc) should be made a macro
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

//Transfer function for density contribution in number counts
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//clt -> CCL_ClTracer object (must be of the CL_TRACER_NC type)
static double transfer_dens(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double chi=(l+0.5)/k;
  if(chi<=clt->chimax) {
    double z,pz,bz,h;
    double a=ccl_scale_factor_of_chi(cosmo,chi, status);
    if(a>0)
      z=1./a-1;
    else
      z=1E6;
    pz=spline_eval(z,clt->spl_nz);
    bz=spline_eval(z,clt->spl_bz);
    h=cosmo->params.h*ccl_h_over_h0(cosmo,a, status)/CLIGHT_HMPC;
    return pz*bz*h;
  }
  else {
    return 0;
  }
}

//Transfer function for RSD contribution in number counts
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//clt -> CCL_ClTracer object (must be of the CL_TRACER_NC type)
static double transfer_rsd(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double chi0=(l+0.5)/k;
  double chi1=(l+1.5)/k;
  if((chi0<=clt->chimax) || (chi1<=clt->chimax)) {
    double a0,a1,z0,z1,pz0,pz1,gf0,gf1,fg0,fg1,h0,h1,term0,term1;
    if(chi0<=clt->chimax) a0=ccl_scale_factor_of_chi(cosmo,chi0, status);
    else a0=ccl_scale_factor_of_chi(cosmo,clt->chimax, status);
    if(a0>0) z0=1./a0-1;
    else z0=1E6;
    if(chi1<=clt->chimax) a1=ccl_scale_factor_of_chi(cosmo,chi1, status);
    else a1=ccl_scale_factor_of_chi(cosmo,clt->chimax, status);
    if(a1>0) z1=1./a1-1;
    else z1=1E6;
    pz0=spline_eval(z0,clt->spl_nz);
    pz1=spline_eval(z1,clt->spl_nz);
    gf0=1;
    gf1=ccl_growth_factor(cosmo,a1,status)/ccl_growth_factor(cosmo,a0, status);
    fg0=ccl_growth_rate(cosmo,a0, status);
    fg1=ccl_growth_rate(cosmo,a1, status);
    h0=cosmo->params.h*ccl_h_over_h0(cosmo,a0,status)/CLIGHT_HMPC;
    h1=cosmo->params.h*ccl_h_over_h0(cosmo,a1,status)/CLIGHT_HMPC;
    term0=pz0*fg0*gf0*h0*(1+8.*l)/((2*l+1.)*(2*l+1.));
    term1=pz1*fg1*gf1*h1*sqrt((l+0.5)/(l+1.5))*4./(2*l+3);

    return term0-term1;
  }
  else {
    return 0;
  }
}

//Transfer function for magnification contribution in number counts
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//clt -> CCL_ClTracer object (must be of the CL_TRACER_NC type)
static double transfer_mag(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double chi=(l+0.5)/k;
  if(chi<=clt->chimax) {
    double a=ccl_scale_factor_of_chi(cosmo,chi,status);
    double wM=spline_eval(chi,clt->spl_wM);
    
    if(wM<=0)
      return 0;
    else
      return -2*clt->prefac_lensing*l*(l+1)*wM/(a*chi*k*k);
    //The actual prefactor on large scales should be sqrt((l+2.)*(l+1.)*l*(l-1.)) instead of l*(l+1)
    //      return clt->prefac_lensing*sqrt((l+2.)*(l+1.)*l*(l-1.))*gf*wL/(a*chi*k*k);
  }
  else
    return 0;
}

//Transfer function for shear
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//clt -> CCL_ClTracer object (must be of the CL_TRACER_WL type)
static double transfer_wl(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double chi=(l+0.5)/k;
  if(chi<=clt->chimax) {
    double a=ccl_scale_factor_of_chi(cosmo,chi, status);
    double wL=spline_eval(chi,clt->spl_wL);
    
    if(wL<=0)
      return 0;
    else
      return clt->prefac_lensing*l*(l+1)*wL/(a*chi*k*k);
    //The actual prefactor on large scales should be sqrt((l+2.)*(l+1.)*l*(l-1.)) instead of l*(l+1)
    //      return clt->prefac_lensing*sqrt((l+2.)*(l+1.)*l*(l-1.))*gf*wL/(a*chi*k*k);
  }
  else
    return 0;
}

//Transfer function for intrinsic alignment contribution in shear
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//clt -> CCL_ClTracer object (must be of the CL_TRACER_WL type)
static double transfer_IA_NLA(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double chi=(l+0.5)/k;
  if(chi<=clt->chimax) {
    double z,pz,ba,rf,h;
    double a=ccl_scale_factor_of_chi(cosmo,chi, status);
    if(a>0)
      z=1./a-1;
    else
      z=1E6;
    pz=spline_eval(z,clt->spl_nz);
    ba=spline_eval(z,clt->spl_ba);
    rf=spline_eval(z,clt->spl_rf);
    h=cosmo->params.h*ccl_h_over_h0(cosmo,a, status)/CLIGHT_HMPC;
    return pz*ba*rf*h*sqrt((l+2.)*(l+1.)*l*(l-1.))/((l+0.5)*(l+0.5));
  }
  else {
    return 0;
  }
}

//Wrapper for transfer function
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//clt -> CCL_ClTracer object
static double transfer_wrap(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double transfer_out=0;

  if(clt->tracer_type==CL_TRACER_NC) {
    transfer_out=transfer_dens(l,k,cosmo,clt, status);
    if(clt->has_rsd)
      transfer_out+=transfer_rsd(l,k,cosmo,clt, status);
    if(clt->has_magnification)
      transfer_out+=transfer_mag(l,k,cosmo,clt, status);
  }
  else if(clt->tracer_type==CL_TRACER_WL) {
    transfer_out=transfer_wl(l,k,cosmo,clt, status);
    if(clt->has_intrinsic_alignment)
      transfer_out+=transfer_IA_NLA(l,k,cosmo,clt, status);
  }
  else
    transfer_out=-1;
  return transfer_out;
}

//Params for power spectrum integrand
typedef struct {
  int l;
  ccl_cosmology *cosmo;
  CCL_ClTracer *clt1;
  CCL_ClTracer *clt2;
  int *status;
} IntClPar;

//Integrand for integral power spectrum
static double cl_integrand(double lk,void *params)
{
  IntClPar *p=(IntClPar *)params;
  double k=pow(10.,lk);
  double chimax=fmax(p->clt1->chimax,p->clt2->chimax);
  double chi=(p->l+0.5)/k;
  if(chi>chimax)
    return 0;
  else {
    double t1,t2;
    double a=ccl_scale_factor_of_chi(p->cosmo,chi, p->status); //Limber
    double pk=ccl_nonlin_matter_power(p->cosmo,k,a, p->status);
    t1=transfer_wrap(p->l,k,p->cosmo,p->clt1, p->status);
    t2=transfer_wrap(p->l,k,p->cosmo,p->clt2, p->status);
    return k*t1*t2*pk;
  }
}

//Figure out k intervals where the Limber kernel has support
//clt1 -> tracer #1
//clt2 -> tracer #2
//l    -> angular multipole
//lkmin, lkmax -> log10 of the range of scales where the transfer functions have support
static void get_k_interval(ccl_cosmology *cosmo,CCL_ClTracer *clt1,CCL_ClTracer *clt2,int l,
			   double *lkmin,double *lkmax)
{
  double chimin,chimax;
  if(clt1->tracer_type==CL_TRACER_NC) {
    if(clt2->tracer_type==CL_TRACER_NC) {
      chimin=fmax(clt1->chimin,clt2->chimin);
      chimax=fmin(clt1->chimax,clt2->chimax);
    }
    else {
      chimin=clt1->chimin;
      chimax=clt1->chimax;
    }
  }
  else if(clt2->tracer_type==CL_TRACER_NC) {
    chimin=clt2->chimin;
    chimax=clt2->chimax;
  }
  else {
    chimin=0.5*(l+0.5)/ccl_splines->K_MAX;
    chimax=2*(l+0.5)/ccl_splines->K_MIN_DEFAULT;
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
double ccl_angular_cl(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,CCL_ClTracer *clt2, int * status)
{
  int clastatus=0, qagstatus;
  IntClPar ipar;
  double result=0,eresult;
  double lkmin,lkmax;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);

  get_k_interval(cosmo,clt1,clt2,l,&lkmin,&lkmax);

  ipar.l=l;
  ipar.cosmo=cosmo;
  ipar.clt1=clt1;
  ipar.clt2=clt2;
  ipar.status = &clastatus;
  F.function=&cl_integrand;
  F.params=&ipar;
  qagstatus=gsl_integration_qag(&F,lkmin,lkmax,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
  gsl_integration_workspace_free(w);
  if(qagstatus!=GSL_SUCCESS || *ipar.status) {
    *status=CCL_ERROR_INTEG;
    strcpy(cosmo->status_message,"ccl_cls.c: ccl_angular_cl(): error integrating over k\n");
    return -1;
  }
  ccl_check_status(cosmo,status);

  return M_LN10*result/(l+0.5);
}
//TODO: currently using linear power spectrum



//! Here CCL is passed to Angpow base classes
namespace Angpow {

//! Selection window W(z) with spline input from CCL 
class RadSplineSelect : public RadSelectBase {
 public:
 RadSplineSelect(SplPar* spl): 
  RadSelectBase(spl->x0, spl->xf), spl_(spl) {}


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
CosmoCoordCCL(ccl_cosmology * cosmo, double zmin=0., double zmax=9., size_t npts=1000)
  : ccl_cosmo_(cosmo), zmin_(zmin), zmax_(zmax), npts_(npts) 
  {
    int status = 0;
    std::vector<double> vlos(npts_);
    std::vector<double> vz(npts_);
    for(size_t i=0; i<npts_; i++) {
      vz[i]=i*(zmax-zmin)/(double)npts_;
      vlos[i]=ccl_comoving_radial_distance(cosmo, 1.0/(1+vz[i]), &status);
    }
    rofzfunc_.DefinePoints(vz,vlos); //r(z0)
    zofrfunc_.DefinePoints(vlos,vz); //z(r0) = r^{-1}(r0)
  }//Ctor

  //! Dtor
  virtual ~CosmoCoordCCL() {}


  //! r(z): radial comoving distance Mpc
  inline double getLOS(double z) const { return  r(z); } 
  inline virtual double r(double z) const { return rofzfunc_.YInterp(z); } 
  //! z(r): the inverse of radial comoving distance (Mpc)
  inline virtual double z(double r) const { return zofrfunc_.YInterp(r); }
  inline double getInvLOS(double r) const { return z(r); }
  
  //!Hubble Cte
  inline double h() const { return ccl_cosmo_->params.h; }
  //!Hubble function
  inline double Ez(double z) const { int status=0; return ccl_h_over_h0(ccl_cosmo_,1.0/(1+z),&status); }
  //inline double EzMpcm1(double z) const {int status=0; double HL=cosmofunc_.HubbleLengthMpc(); return Ez(z)/HL; }

  //! z=0 matter density 
  inline double OmegaMatter() const { return ccl_cosmo_->params.Omega_m; }
  //! z=0 cosmological constant density 
  inline double OmegaLambda() const { return ccl_cosmo_->params.Omega_l; }
    
 protected:
  ccl_cosmology* ccl_cosmo_;  //!< access to CCL cosmology
  SLinInterp1D  rofzfunc_; //!< linear interpolation r(z0)
  SLinInterp1D  zofrfunc_; //!< linear interpolation z(r0) = r^{-1}(r0)
  double zmin_;           //!< minimal z
  double zmax_;           //!< maximal z
  size_t npts_;        //!< number of points to define the interpolation in [zmin, zmax]
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
    R_ = ccl_comoving_radial_distance(cosmo_, 1.0/(1+z), &status);
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
    r_8 Pk = ccl_linear_matter_power(cosmo_, k , 1./(1+z_), &status);
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
SplPar * ccl_angular_cls_angpow(ccl_cosmology *ccl_cosmo, int lmax, CCL_ClTracer *clt_gc1, CCL_ClTracer *clt_gc2, int * status)
{
  if(clt_gc1->has_magnification || clt_gc2->has_magnification)
    printf("Magnification term not implemented in Angpow yet: will be ignored");
  if(clt_gc1->tracer_type==CL_TRACER_WL || clt_gc2->tracer_type==CL_TRACER_WL)
    printf("Weak lensing functions not implemented in Agnpow yet: will fail");
  
  // Initialize the Angpow parameters
  Angpow::Parameters para = Angpow::Param::Instance().GetParam();
  para.chebyshev_order_1 = 9;
  para.chebyshev_order_2 = 9;
  para.cl_kmax = 10;
  para.linearStep = 40;
  para.logStep = 1.15;

  // Initialize the radial selection windows W(z)
  Angpow::RadSplineSelect Z1win(clt_gc1->spl_nz);
  Angpow::RadSplineSelect Z2win(clt_gc2->spl_nz);

  // The cosmological distance tool to make the conversion z <-> r(z)
  Angpow::CosmoCoordCCL cosmo(ccl_cosmo, 1./ccl_splines->A_SPLINE_MAX-1, 1./ccl_splines->A_SPLINE_MIN-1, ccl_splines->A_SPLINE_NA); //, para.cosmo_precision);

  // Initilaie the two integrand functions f(ell,k,z)
  Angpow::IntegrandCCL int1(clt_gc1, ccl_cosmo);
  Angpow::IntegrandCCL int2(clt_gc2, ccl_cosmo);

  // Initialize the Cl with parameters to select the ell set which is interpolated after the processing
  Angpow::Clbase clout(lmax,para.linearStep, para.logStep);

  // Main class to compute Cl with Angpow
  Angpow::Pk2Cl pk2cl; //Default: the user parameters are used in the Constructor 
  pk2cl.PrintParam();
  pk2cl.Compute(int1, int2, cosmo, &Z1win, &Z2win, lmax, clout);

  // Pass the Clbase class values (ell and C_ell) to the output spline
  int n_l = clout.Size();
  std::vector<double> ls(n_l);
  std::vector<double> cls(n_l);
  for(int index_l=0; index_l<n_l; index_l++) {
    ls[index_l]=clout[index_l].first; cls[index_l]=clout[index_l].second; 
  }
  SplPar * spl_cl = spline_init(clout.Size(), &ls[0], &cls[0], clout[0].second, clout[n_l-1].second );

  return spl_cl;
}
