#include "ccl_cls.h"
#include "ccl_power.h"
#include "ccl_background.h"
#include "ccl_error.h"
#include "ccl_utils.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "gsl/gsl_errno.h"
#include "gsl/gsl_integration.h"

//Spline creator
//n     -> number of points
//x     -> x-axis
//y     -> f(x)-axis
//y0,yf -> values of f(x) to use beyond the interpolation range
static SplPar *spline_init(int n,double *x,double *y,double y0,double yf)
{
  SplPar *spl=malloc(sizeof(SplPar));
  if(spl==NULL)
    return NULL;
  
  spl->intacc=gsl_interp_accel_alloc();
  spl->spline=gsl_spline_alloc(gsl_interp_cspline,n);
  int status=gsl_spline_init(spl->spline,x,y,n);
  if(status) {
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

//Params for lensing kernel integrand
typedef struct {
  double chi;
  SplPar *spl_pz;
  ccl_cosmology *cosmo;
} IntLensPar;

//Integrand for lensing kernel
static double integrand_wl(double chip,void *params)
{
  IntLensPar *p=(IntLensPar *)params;
  double chi=p->chi;
  double a=ccl_scale_factor_of_chi(p->cosmo,chip);
  double z=1./a-1;
  double pz=spline_eval(z,p->spl_pz);
  double h=p->cosmo->params.h*ccl_h_over_h0(p->cosmo,a)/CLIGHT_HMPC;

  if(chi==0)
    return h*pz;
  else
    return h*pz*(chip-chi)/chip;
}

//Integral to compute lensing window function
//chi     -> comoving distance
//cosmo   -> ccl_cosmology object
//spl_pz  -> normalized N(z) spline
//chi_max -> maximum comoving distance to which the integral is computed
//win     -> result is stored here
static int window_lensing(double chi,ccl_cosmology *cosmo,SplPar *spl_pz,double chi_max,double *win)
{
  int status;
  double result,eresult;
  IntLensPar ip;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);

  ip.chi=chi;
  ip.cosmo=cosmo;
  ip.spl_pz=spl_pz;
  F.function=&integrand_wl;
  F.params=&ip;
  status=gsl_integration_qag(&F,chi,chi_max,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
  *win=result;
  gsl_integration_workspace_free(w);
  if(status!=GSL_SUCCESS)
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
} IntMagPar;

//Integrand for magnification kernel
static double integrand_mag(double chip,void *params)
{
  IntMagPar *p=(IntMagPar *)params;
  double chi=p->chi;
  double a=ccl_scale_factor_of_chi(p->cosmo,chip);
  double z=1./a-1;
  double pz=spline_eval(z,p->spl_pz);
  double sz=spline_eval(z,p->spl_sz);
  double h=p->cosmo->params.h*ccl_h_over_h0(p->cosmo,a)/CLIGHT_HMPC;

  if(chi==0)
    return h*pz*(1-2.5*sz);
  else
    return h*pz*(1-2.5*sz)*(chip-chi)/chip;
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
  int status;
  double result,eresult;
  IntMagPar ip;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);

  ip.chi=chi;
  ip.cosmo=cosmo;
  ip.spl_pz=spl_pz;
  ip.spl_sz=spl_sz;
  F.function=&integrand_mag;
  F.params=&ip;
  status=gsl_integration_qag(&F,chi,chi_max,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
  *win=result;
  gsl_integration_workspace_free(w);
  if(status!=GSL_SUCCESS)
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
				   int nz_rf,double *z_rf,double *rf)
{
  int status=0;
  CCL_ClTracer *clt=malloc(sizeof(CCL_ClTracer));
  if(clt==NULL) {
    cosmo->status=CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): memory allocation\n");
    return NULL;
  }
  clt->tracer_type=tracer_type;

  double hub=cosmo->params.h*ccl_h_over_h0(cosmo,1.)/CLIGHT_HMPC;
  clt->prefac_lensing=1.5*hub*hub*cosmo->params.Omega_m;

  if((tracer_type==CL_TRACER_NC)||(tracer_type==CL_TRACER_WL)) {
    clt->chimax=ccl_comoving_radial_distance(cosmo,1./(1+z_n[nz_n-1]));
    clt->spl_nz=spline_init(nz_n,z_n,n,0,0);
    if(clt->spl_nz==NULL) {
      free(clt);
      cosmo->status=CCL_ERROR_SPLINE;
      strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error initializing spline for N(z)\n");
      return NULL;
    }

    //Normalize n(z)
    gsl_function F;
    double nz_norm,nz_enorm;
    double *nz_normalized=malloc(nz_n*sizeof(double));
    if(nz_normalized==NULL) {
      spline_free(clt->spl_nz);
      free(clt);
      cosmo->status=CCL_ERROR_MEMORY;
      strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): memory allocation\n");
      return NULL;
    }

    gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);
    F.function=&speval_bis;
    F.params=clt->spl_nz;
    status=gsl_integration_qag(&F,z_n[0],z_n[nz_n-1],0,1E-4,1000,GSL_INTEG_GAUSS41,w,&nz_norm,&nz_enorm);
    gsl_integration_workspace_free(w); //TODO:check for integration errors
    if(status!=GSL_SUCCESS) {
      spline_free(clt->spl_nz);
      free(clt);
      cosmo->status=CCL_ERROR_INTEG;
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
      cosmo->status=CCL_ERROR_SPLINE;
      strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error initializing normalized spline for N(z)\n");
      return NULL;
    }

    if(tracer_type==CL_TRACER_NC) {
      //Initialize bias spline
      clt->spl_bz=spline_init(nz_b,z_b,b,b[0],b[nz_b-1]);
      if(clt->spl_bz==NULL) {
	spline_free(clt->spl_nz);
	free(clt);
	cosmo->status=CCL_ERROR_SPLINE;
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
	double chimax=ccl_comoving_radial_distance(cosmo,1./(1+zmax));
	//TODO: The interval in chi (5. Mpc) should be made a macro

	clt->spl_sz=spline_init(nz_s,z_s,s,s[0],s[nz_s-1]);
	if(clt->spl_sz==NULL) {
	  spline_free(clt->spl_nz);
	  spline_free(clt->spl_bz);
	  free(clt);
	  cosmo->status=CCL_ERROR_SPLINE;
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
	  cosmo->status=CCL_ERROR_LINSPACE;
	  strcpy(cosmo->status_message,
		 "ccl_cls.c: ccl_cl_tracer_new(): Error creating linear spacing in chi\n");
	  return NULL;
	}
	y=malloc(nchi*sizeof(double));
	if(y==NULL) {
	  free(x);
	  spline_free(clt->spl_nz);
	  spline_free(clt->spl_bz);
	  spline_free(clt->spl_sz);
	  free(clt);
	  cosmo->status=CCL_ERROR_MEMORY;
	  strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): memory allocation\n");
	  return NULL;
	}
      
	for(int j=0;j<nchi;j++)
	  status|=window_magnification(x[j],cosmo,clt->spl_nz,clt->spl_sz,chimax,&(y[j]));
	if(status) {
	  free(y);
	  free(x);
	  spline_free(clt->spl_nz);
	  spline_free(clt->spl_bz);
	  spline_free(clt->spl_sz);
	  free(clt);
	  cosmo->status=CCL_ERROR_INTEG;
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
	  cosmo->status=CCL_ERROR_SPLINE;
	  strcpy(cosmo->status_message,
		 "ccl_cls.c: ccl_cl_tracer_new(): error initializing spline for lensing window\n");
	  return NULL;
	}
	free(x); free(y);
      }
      clt->chimin=ccl_comoving_radial_distance(cosmo,1./(1+z_n[0]));
    }
    else if(tracer_type==CL_TRACER_WL) {
      //Compute weak lensing kernel
      int nchi;
      double *x,*y;
      double dchi=5.;
      double zmax=clt->spl_nz->xf;
      double chimax=ccl_comoving_radial_distance(cosmo,1./(1+zmax));
      //TODO: The interval in chi (5. Mpc) should be made a macro
      clt->chimin=0;
      nchi=(int)(chimax/dchi)+1;
      x=ccl_linear_spacing(0.,chimax,nchi);
      dchi=chimax/nchi;
      if(x==NULL || (fabs(x[0]-0)>1E-5) || (fabs(x[nchi-1]-chimax)>1e-5)) {
	spline_free(clt->spl_nz);
	free(clt);
	cosmo->status=CCL_ERROR_LINSPACE;
	strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): Error creating linear spacing in chi\n");
	return NULL;
      }
      y=malloc(nchi*sizeof(double));
      if(y==NULL) {
	free(x);
	spline_free(clt->spl_nz);
	free(clt);
	cosmo->status=CCL_ERROR_MEMORY;
	strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): memory allocation\n");
	return NULL;
      }
      
      for(int j=0;j<nchi;j++)
	status|=window_lensing(x[j],cosmo,clt->spl_nz,chimax,&(y[j]));
      if(status) {
	free(y);
	free(x);
	spline_free(clt->spl_nz);
	free(clt);
	cosmo->status=CCL_ERROR_INTEG;
	strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error computing lensing window\n");
	return NULL;
      }

      clt->spl_wL=spline_init(nchi,x,y,y[0],0);
      if(clt->spl_wL==NULL) {
	free(y);
	free(x);
	spline_free(clt->spl_nz);
	free(clt);
	cosmo->status=CCL_ERROR_SPLINE;
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
	  cosmo->status=CCL_ERROR_SPLINE;
	  strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error initializing spline for rf(z)\n");
	  return NULL;
	}
	clt->spl_ba=spline_init(nz_ba,z_ba,ba,ba[0],ba[nz_ba-1]);
	if(clt->spl_ba==NULL) {
	  spline_free(clt->spl_rf);
	  spline_free(clt->spl_nz);
	  free(clt);
	  cosmo->status=CCL_ERROR_SPLINE;
	  strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer_new(): error initializing spline for ba(z)\n");
	  return NULL;
	}
      }
    }
  }
  else {
    cosmo->status=CCL_ERROR_INCONSISTENT;
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
				int nz_rf,double *z_rf,double *rf)
{
  CCL_ClTracer *clt=cl_tracer_new(cosmo,tracer_type,has_rsd,has_magnification,has_intrinsic_alignment,
				  nz_n,z_n,n,nz_b,z_b,b,nz_s,z_s,s,nz_ba,z_ba,ba,nz_rf,z_rf,rf);
  ccl_check_status(cosmo);
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
					      int nz_s,double *z_s,double *s)
{
  return ccl_cl_tracer_new(cosmo,CL_TRACER_NC,has_rsd,has_magnification,0,
			   nz_n,z_n,n,nz_b,z_b,b,nz_s,z_s,s,
			   -1,NULL,NULL,-1,NULL,NULL);
}

CCL_ClTracer *ccl_cl_tracer_number_counts_simple_new(ccl_cosmology *cosmo,
						     int nz_n,double *z_n,double *n,
						     int nz_b,double *z_b,double *b)
{
  return ccl_cl_tracer_new(cosmo,CL_TRACER_NC,0,0,0,
			   nz_n,z_n,n,nz_b,z_b,b,-1,NULL,NULL,
			   -1,NULL,NULL,-1,NULL,NULL);
}

CCL_ClTracer *ccl_cl_tracer_lensing_new(ccl_cosmology *cosmo,
					int has_alignment,
					int nz_n,double *z_n,double *n,
					int nz_ba,double *z_ba,double *ba,
					int nz_rf,double *z_rf,double *rf)
{
  return ccl_cl_tracer_new(cosmo,CL_TRACER_WL,0,0,has_alignment,
			   nz_n,z_n,n,-1,NULL,NULL,-1,NULL,NULL,
			   nz_ba,z_ba,ba,nz_rf,z_rf,rf);
}

CCL_ClTracer *ccl_cl_tracer_lensing_simple_new(ccl_cosmology *cosmo,
					       int nz_n,double *z_n,double *n)
{
  return ccl_cl_tracer_new(cosmo,CL_TRACER_WL,0,0,0,
			   nz_n,z_n,n,-1,NULL,NULL,-1,NULL,NULL,
			   -1,NULL,NULL,-1,NULL,NULL);
}

//Transfer function for density contribution in number counts
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//clt -> CCL_ClTracer object (must be of the CL_TRACER_NC type)
static double transfer_dens(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt)
{
  double chi=(l+0.5)/k;
  if(chi<=clt->chimax) {
    double z,pz,bz,h;
    double a=ccl_scale_factor_of_chi(cosmo,chi);
    if(a>0)
      z=1./a-1;
    else
      z=1E6;
    pz=spline_eval(z,clt->spl_nz);
    bz=spline_eval(z,clt->spl_bz);
    h=cosmo->params.h*ccl_h_over_h0(cosmo,a)/CLIGHT_HMPC;
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
static double transfer_rsd(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt)
{
  double chi0=(l+0.5)/k;
  double chi1=(l+1.5)/k;
  if((chi0<=clt->chimax) || (chi1<=clt->chimax)) {
    double a0,a1,z0,z1,pz0,pz1,gf0,gf1,fg0,fg1,h0,h1,term0,term1;
    if(chi0<=clt->chimax) a0=ccl_scale_factor_of_chi(cosmo,chi0);
    else a0=ccl_scale_factor_of_chi(cosmo,clt->chimax);
    if(a0>0) z0=1./a0-1;
    else z0=1E6;
    if(chi1<=clt->chimax) a1=ccl_scale_factor_of_chi(cosmo,chi1);
    else a1=ccl_scale_factor_of_chi(cosmo,clt->chimax);
    if(a1>0) z1=1./a1-1;
    else z1=1E6;
    pz0=spline_eval(z0,clt->spl_nz);
    pz1=spline_eval(z1,clt->spl_nz);
    gf0=1;
    gf1=ccl_growth_factor(cosmo,a1)/ccl_growth_factor(cosmo,a0);
    fg0=ccl_growth_rate(cosmo,a0);
    fg1=ccl_growth_rate(cosmo,a1);
    h0=cosmo->params.h*ccl_h_over_h0(cosmo,a0)/CLIGHT_HMPC;
    h1=cosmo->params.h*ccl_h_over_h0(cosmo,a1)/CLIGHT_HMPC;
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
static double transfer_mag(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt)
{
  double chi=(l+0.5)/k;
  if(chi<=clt->chimax) {
    double a=ccl_scale_factor_of_chi(cosmo,chi);
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
static double transfer_wl(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt)
{
  double chi=(l+0.5)/k;
  if(chi<=clt->chimax) {
    double a=ccl_scale_factor_of_chi(cosmo,chi);
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
static double transfer_IA_NLA(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt)
{
  double chi=(l+0.5)/k;
  if(chi<=clt->chimax) {
    double z,pz,ba,rf,h;
    double a=ccl_scale_factor_of_chi(cosmo,chi);
    if(a>0)
      z=1./a-1;
    else
      z=1E6;
    pz=spline_eval(z,clt->spl_nz);
    ba=spline_eval(z,clt->spl_ba);
    rf=spline_eval(z,clt->spl_rf);
    h=cosmo->params.h*ccl_h_over_h0(cosmo,a)/CLIGHT_HMPC;
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
static double transfer_wrap(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt)
{
  double transfer_out=0;
  if(clt->tracer_type==CL_TRACER_NC) {
    transfer_out=transfer_dens(l,k,cosmo,clt);
    if(clt->has_rsd)
      transfer_out+=transfer_rsd(l,k,cosmo,clt);
    if(clt->has_magnification)
      transfer_out+=transfer_mag(l,k,cosmo,clt);
  }
  else if(clt->tracer_type==CL_TRACER_WL) {
    transfer_out=transfer_wl(l,k,cosmo,clt);
    if(clt->has_intrinsic_alignment)
      transfer_out+=transfer_IA_NLA(l,k,cosmo,clt);
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
    double a=ccl_scale_factor_of_chi(p->cosmo,chi); //Limber
    double pk=ccl_nonlin_matter_power(p->cosmo,a,k);
    t1=transfer_wrap(p->l,k,p->cosmo,p->clt1);
    t2=transfer_wrap(p->l,k,p->cosmo,p->clt2);

    return k*t1*t2*pk;
  }
}

//Figure out k intervals where the Limber kernel has support
//clt1 -> tracer #1
//clt2 -> tracer #2
//l    -> angular multipole
//lkmin, lkmax -> log10 of the range of scales where the transfer functions have support
static void get_k_interval(CCL_ClTracer *clt1,CCL_ClTracer *clt2,int l,
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
    chimin=0.5*(l+0.5)/K_MAX_INT;
    chimax=2*(l+0.5)/K_MIN_INT;
  }

  if(chimin<=0)
    chimin=0.5*(l+0.5)/K_MAX_INT;

  *lkmax=fmin( 2,log10(2  *(l+0.5)/chimin));
  *lkmin=fmax(-4,log10(0.5*(l+0.5)/chimax));
}

//Compute angular power spectrum between two bins
//cosmo -> ccl_cosmology object
//l -> angular multipole
//clt1 -> tracer #1
//clt2 -> tracer #2
double ccl_angular_cl(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,CCL_ClTracer *clt2)
{
  int status=0;
  IntClPar ipar;
  double result=0,eresult;
  double lkmin,lkmax;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);

  get_k_interval(clt1,clt2,l,&lkmin,&lkmax);

  ipar.l=l;
  ipar.cosmo=cosmo;
  ipar.clt1=clt1;
  ipar.clt2=clt2;
  F.function=&cl_integrand;
  F.params=&ipar;
  status=gsl_integration_qag(&F,lkmin,lkmax,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
  gsl_integration_workspace_free(w);
  if(status!=GSL_SUCCESS) {
    cosmo->status=CCL_ERROR_INTEG;
    strcpy(cosmo->status_message,"ccl_cls.c: ccl_angular_cl(): error integrating over k\n");
    return -1;
  }
  ccl_check_status(cosmo);

  return M_LN10*result/(l+0.5);
}
//TODO: currently using linear power spectrum
