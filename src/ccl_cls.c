#include "ccl_cls.h"
#include "ccl_power.h"
#include "ccl_background.h"
#include "ccl_utils.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "gsl/gsl_errno.h"
#include "gsl/gsl_integration.h"

static void *my_malloc(size_t size)
{
  void *outptr=malloc(size);
  if(outptr==NULL) {
    fprintf(stderr,"Out of memory\n");
    exit(1);
  }

  return outptr;
}

static SplPar *spline_init(int n,double *x,double *y,double y0,double yf)
{
  SplPar *spl=(SplPar *)my_malloc(sizeof(SplPar));
  spl->intacc=gsl_interp_accel_alloc();
  spl->spline=gsl_spline_alloc(gsl_interp_cspline,n);
  //TODO: check for spline init errors
  gsl_spline_init(spl->spline,x,y,n);
  spl->x0=x[0];
  spl->xf=x[n-1];
  spl->y0=y0;
  spl->yf=yf;

  return spl;
}

static double spline_eval(double x,SplPar *spl)
{
  if(x<=spl->x0)
    return spl->y0;
  else if(x>=spl->xf) 
    return spl->yf;
  else
    return gsl_spline_eval(spl->spline,x,spl->intacc);
}

static double speval_bis(double x,void *params)
{
  return spline_eval(x,(SplPar *)params);
}

static void spline_free(SplPar *spl)
{
  gsl_spline_free(spl->spline);
  gsl_interp_accel_free(spl->intacc);
  free(spl);
}

typedef struct {
  double chi;
  SplPar *spl_pz;
  ccl_cosmology *cosmo;
} IntLensPar;

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

static double window_lensing(double chi,ccl_cosmology *cosmo,SplPar *spl_pz,double chi_max)
{
  double result,eresult;
  IntLensPar ip;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);

  ip.chi=chi;
  ip.cosmo=cosmo;
  ip.spl_pz=spl_pz;
  F.function=&integrand_wl;
  F.params=&ip;
  gsl_integration_qag(&F,chi,chi_max,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
  //TODO: chi_max should be changed to chi_horizon
  //we should precompute this quantity and store it in cosmo by default
  //TODO: check for integration errors
  gsl_integration_workspace_free(w);

  return result;
}

ClTracer *ccl_tracer_new(ccl_cosmology *cosmo,int tracer_type,
			 int nz_n,double *z_n,double *n,
			 int nz_b,double *z_b,double *b)
{
  ClTracer *clt=my_malloc(sizeof(ClTracer));
  clt->tracer_type=tracer_type;

  double hub=cosmo->params.h*ccl_h_over_h0(cosmo,1.)/CLIGHT_HMPC;
  clt->prefac_lensing=1.5*hub*hub*cosmo->params.Omega_m;

  if((tracer_type==CL_TRACER_NC)||(tracer_type==CL_TRACER_WL)) {
    clt->chimax=ccl_comoving_radial_distance(cosmo,1./(1+z_n[nz_n-1]));
    clt->spl_nz=spline_init(nz_n,z_n,n,0,0);

    //Normalize n(z)
    gsl_function F;
    double nz_norm,nz_enorm;
    double *nz_normalized=my_malloc(nz_n*sizeof(double));
    gsl_integration_workspace *w=gsl_integration_workspace_alloc(1000);
    F.function=&speval_bis;
    F.params=clt->spl_nz;
    gsl_integration_qag(&F,z_n[0],z_n[nz_n-1],0,1E-4,1000,GSL_INTEG_GAUSS41,w,&nz_norm,&nz_enorm);
    gsl_integration_workspace_free(w); //TODO:check for integration errors
    for(int ii=0;ii<nz_n;ii++)
      nz_normalized[ii]=n[ii]/nz_norm;
    spline_free(clt->spl_nz);
    clt->spl_nz=spline_init(nz_n,z_n,nz_normalized,0,0);
    free(nz_normalized);

    if(tracer_type==CL_TRACER_NC) {
      //Initialize bias spline
      clt->spl_bz=spline_init(nz_b,z_b,b,b[0],b[nz_b-1]);
      clt->chimin=ccl_comoving_radial_distance(cosmo,1./(1+z_n[0]));
    }
    else if(tracer_type==CL_TRACER_WL) {
      //Compute weak lensing kernel
      int nchi;
      double *x,*y;
      double dchi=3.;
      double zmax=clt->spl_nz->xf;
      double chimax=ccl_comoving_radial_distance(cosmo,1./(1+zmax));
      clt->chimin=0;
      nchi=(int)(chimax/dchi)+1; dchi=chimax/nchi; nchi=0;
      x=ccl_linear_spacing(0.,chimax,dchi,&nchi);
      y=my_malloc(nchi*sizeof(double));
      
      for(int j=0;j<nchi;j++)
	y[j]=window_lensing(x[j],cosmo,clt->spl_nz,chimax);
      clt->spl_wL=spline_init(nchi,x,y,y[0],0);
      free(x); free(y);
    }
  }
  else {
    fprintf(stderr,"Wrong tracer type\n");
    exit(1);
  }

  return clt;
}

void ccl_tracer_free(ClTracer *clt)
{
  spline_free(clt->spl_nz);
  if(clt->tracer_type==CL_TRACER_NC)
    spline_free(clt->spl_bz);
  else if(clt->tracer_type==CL_TRACER_WL)
    spline_free(clt->spl_wL);
  else {
    fprintf(stderr,"Wrong tracer type\n");
    exit(1);
  }
  free(clt);
}

static double transfer_nc(int l,double k,ccl_cosmology *cosmo,ClTracer *clt,int *status)
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
    gf=ccl_growth_factor(cosmo,a,status);
    h=cosmo->params.h*ccl_h_over_h0(cosmo,a)/CLIGHT_HMPC;
    return pz*bz*gf*h;
  }
  else {
    return 0;
  }
}

static double transfer_wl(int l,double k,ccl_cosmology *cosmo,ClTracer *clt,int *status)
{
  double chi=(l+0.5)/k;
  if(chi<=clt->chimax) {
    double a=ccl_scale_factor_of_chi(cosmo,chi);
    double gf=ccl_growth_factor(cosmo,a,status);
    double wL=spline_eval(chi,clt->spl_wL);
    
    if(wL<=0)
      return 0;
    else
      return clt->prefac_lensing*sqrt((l+2.)*(l+1.)*l*(l-1.))*gf*wL/(a*chi*k*k);
  }
  else
    return 0;
}

static double transfer_wrap(int l,double k,ccl_cosmology *cosmo,ClTracer *clt,int *status)
{
  if(clt->tracer_type==CL_TRACER_NC)
    return transfer_nc(l,k,cosmo,clt,status);
  else if(clt->tracer_type==CL_TRACER_WL)
    return transfer_wl(l,k,cosmo,clt,status);
  else {
    fprintf(stderr,"Wrong tracer type\n");
    exit(1);
  }
}

typedef struct {
  int l;
  ccl_cosmology *cosmo;
  ClTracer *clt1;
  ClTracer *clt2;
  int *status;
} IntClPar;

static double cl_integrand(double lk,void *params)
{
  double t1,t2;
  IntClPar *p=(IntClPar *)params;
  double k=pow(10.,lk);
  double pk=ccl_linear_matter_power(p->cosmo,1.,k,p->status);
  t1=transfer_wrap(p->l,k,p->cosmo,p->clt1,p->status);
  t2=transfer_wrap(p->l,k,p->cosmo,p->clt2,p->status);

  return k*t1*t2*pk;
}

//Figure out k intervals where the Limber kernel has support
static void get_k_interval(ClTracer *clt1,ClTracer *clt2,int l,
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

double ccl_angular_cl(ccl_cosmology *cosmo,int l,ClTracer *clt1,ClTracer *clt2,int *status)
{
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
  ipar.status=status;
  F.function=&cl_integrand;
  F.params=&ipar;
  *status |= gsl_integration_qag(&F,lkmin,lkmax,0,1E-4,1000,GSL_INTEG_GAUSS41,w,&result,&eresult);
  //TODO: check for integration errors
  gsl_integration_workspace_free(w);

  return M_LN10*result/(l+0.5);
}
//TODO: implement RSD? magnification? IA?
//TODO: using linear power spectrum
//TODO: carry around status according to the new scheme
