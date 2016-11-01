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
				   int nz_n,double *z_n,double *n,
				   int nz_b,double *z_b,double *b)
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
      nchi=(int)(chimax/dchi)+1; dchi=chimax/nchi; nchi=0;
      x=ccl_linear_spacing(0.,chimax,dchi,&nchi);
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
				int nz_n,double *z_n,double *n,
				int nz_b,double *z_b,double *b)
{
  CCL_ClTracer *clt=cl_tracer_new(cosmo,tracer_type,nz_n,z_n,n,nz_b,z_b,b);
  ccl_check_status(cosmo);
  return clt;
}

//CCL_ClTracer destructor
void ccl_cl_tracer_free(CCL_ClTracer *clt)
{
  spline_free(clt->spl_nz);
  if(clt->tracer_type==CL_TRACER_NC)
    spline_free(clt->spl_bz);
  else if(clt->tracer_type==CL_TRACER_WL)
    spline_free(clt->spl_wL);
  free(clt);
}

//Transfer function for number counts
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//clt -> CCL_ClTracer object (must be of the CL_TRACER_NC type)
static double transfer_nc(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt)
{
  double chi=(l+0.5)/k;
  if(chi<=clt->chimax) {
    double z,pz,bz,gf,h;
    double a=ccl_scale_factor_of_chi(cosmo,chi);
    if(a>0)
      z=1./a-1;
    else
      z=1E6;
    pz=spline_eval(z,clt->spl_nz);
    bz=spline_eval(z,clt->spl_bz);
    gf=ccl_growth_factor(cosmo,a);
    h=cosmo->params.h*ccl_h_over_h0(cosmo,a)/CLIGHT_HMPC;
    return pz*bz*gf*h;
  }
  else {
    return 0;
  }
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
    double gf=ccl_growth_factor(cosmo,a);
    double wL=spline_eval(chi,clt->spl_wL);
    
    if(wL<=0)
      return 0;
    else
      return clt->prefac_lensing*sqrt((l+2.)*(l+1.)*l*(l-1.))*gf*wL/(a*chi*k*k);
  }
  else
    return 0;
}

//Wrapper for transfer function
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//clt -> CCL_ClTracer object
static double transfer_wrap(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt)
{
  if(clt->tracer_type==CL_TRACER_NC)
    return transfer_nc(l,k,cosmo,clt);
  else if(clt->tracer_type==CL_TRACER_WL)
    return transfer_wl(l,k,cosmo,clt);
  else
    return -1;
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
  double t1,t2;
  IntClPar *p=(IntClPar *)params;
  double k=pow(10.,lk);
  double pk=ccl_linear_matter_power(p->cosmo,1.,k);
  t1=transfer_wrap(p->l,k,p->cosmo,p->clt1);
  t2=transfer_wrap(p->l,k,p->cosmo,p->clt2);

  return k*t1*t2*pk;
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
//TODO: implement RSD? magnification? IA?
//TODO: currently using linear power spectrum
