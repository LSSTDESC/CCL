#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

#include "ccl.h"

#ifdef HAVE_ANGPOW
#include "Angpow/angpow_ccl.h"
#endif

#define CCL_FRAC_RELEVANT 5E-4
//#define CCL_FRAC_RELEVANT 1E-3
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

//Wrapper around spline_eval with GSL function syntax
static double speval_bis(double x,void *params)
{
  return ccl_spline_eval(x,(SplPar *)params);
}


void ccl_cl_workspace_free(CCL_ClWorkspace *w)
{
  free(w->l_arr);
  free(w);
}

CCL_ClWorkspace *ccl_cl_workspace_new(int lmax,int l_limber,
					  double l_logstep,int l_linstep,int *status)
{
  CCL_ClWorkspace *w=(CCL_ClWorkspace *)malloc(sizeof(CCL_ClWorkspace));
  if(w==NULL) {
    *status=CCL_ERROR_MEMORY;
    return NULL;
  }

  //Set params
  w->lmax=lmax;
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
  //Don't go further than lmaw
  w->l_arr[w->n_ls-1]=w->lmax;

  return w;
}

CCL_ClWorkspace *ccl_cl_workspace_new_limber(int lmax,double l_logstep,int l_linstep,int *status)
{
  return ccl_cl_workspace_new(lmax,-1,l_logstep,l_linstep,status);
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
  double pz=ccl_spline_eval(z,p->spl_pz);
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
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(ccl_gsl->N_ITERATION);

  ip.chi=chi;
  ip.cosmo=cosmo;
  ip.spl_pz=spl_pz;
  ip.status = &status;
  F.function=&integrand_wl;
  F.params=&ip;
  gslstatus=gsl_integration_qag(&F, chi, chi_max, 0,
                                ccl_gsl->INTEGRATION_EPSREL, ccl_gsl->N_ITERATION,
                                ccl_gsl->INTEGRATION_GAUSS_KRONROD_POINTS,
                                w, &result, &eresult);
  *win=result;
  gsl_integration_workspace_free(w);
  if(gslstatus!=GSL_SUCCESS || *ip.status) {
    ccl_raise_gsl_warning(gslstatus, "ccl_cls.c: window_lensing():");
    return 1;
  }
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
  double chi=p->chi;
  double a=ccl_scale_factor_of_chi(p->cosmo,chip, p->status);
  double z=1./a-1;
  double pz=ccl_spline_eval(z,p->spl_pz);
  double sz=ccl_spline_eval(z,p->spl_sz);
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
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(ccl_gsl->N_ITERATION);

  ip.chi=chi;
  ip.cosmo=cosmo;
  ip.spl_pz=spl_pz;
  ip.spl_sz=spl_sz;
  ip.status = &status;
  F.function=&integrand_mag;
  F.params=&ip;
  gslstatus=gsl_integration_qag(&F, chi, chi_max, 0,
                                ccl_gsl->INTEGRATION_EPSREL, ccl_gsl->N_ITERATION,
                                ccl_gsl->INTEGRATION_GAUSS_KRONROD_POINTS,
                                w, &result, &eresult);
  *win=result;
  gsl_integration_workspace_free(w);
  if(gslstatus!=GSL_SUCCESS || *ip.status) {
    ccl_raise_gsl_warning(gslstatus, "ccl_cls.c: window_magnification():");
    return 1;
  }
  //TODO: chi_max should be changed to chi_horizon
  //we should precompute this quantity and store it in cosmo by default

  return 0;
}

static void clt_init_nz(CCL_ClTracer *clt,ccl_cosmology *cosmo,
			int nz_n,double *z_n,double *n,int *status)
{
  int gslstatus;
  //Find redshift range where the N(z) has support
  get_support_interval(nz_n,z_n,n,CCL_FRAC_RELEVANT,&(clt->zmin),&(clt->zmax));
  clt->chimax=ccl_comoving_radial_distance(cosmo,1./(1+clt->zmax),status);
  clt->chimin=ccl_comoving_radial_distance(cosmo,1./(1+clt->zmin),status);
  clt->spl_nz=ccl_spline_init(nz_n,z_n,n,0,0);
  if(clt->spl_nz==NULL) {
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_nz(): error initializing spline for N(z)\n");
  }

  //Normalize n(z)
  gsl_function F;
  double nz_norm,nz_enorm;
  double *nz_normalized=(double *)malloc(nz_n*sizeof(double));
  if(nz_normalized==NULL) {
    ccl_spline_free(clt->spl_nz);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_nz(): memory allocation\n");
    return;
  }
  
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(ccl_gsl->N_ITERATION);
  F.function=&speval_bis;
  F.params=clt->spl_nz;
  gslstatus=gsl_integration_qag(&F, z_n[0], z_n[nz_n-1], 0,
				ccl_gsl->INTEGRATION_EPSREL, ccl_gsl->N_ITERATION,
				ccl_gsl->INTEGRATION_GAUSS_KRONROD_POINTS,
				w, &nz_norm, &nz_enorm);
  gsl_integration_workspace_free(w);
  if(gslstatus!=GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_cls.c: clt_init_nz():");
    ccl_spline_free(clt->spl_nz);
    free(nz_normalized);
    *status=CCL_ERROR_INTEG;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_nz(): integration error when normalizing N(z)\n");
  }
  for(int ii=0;ii<nz_n;ii++)
    nz_normalized[ii]=n[ii]/nz_norm;
  ccl_spline_free(clt->spl_nz);
  clt->spl_nz=ccl_spline_init(nz_n,z_n,nz_normalized,0,0);
  free(nz_normalized);
  if(clt->spl_nz==NULL) {
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_nc_init(): error initializing normalized spline for N(z)\n");
    return;
  }
}

static void clt_init_bz(CCL_ClTracer *clt,ccl_cosmology *cosmo,
			int nz_b,double *z_b,double *b,int *status)
{
  //Initialize bias spline
  clt->spl_bz=ccl_spline_init(nz_b,z_b,b,b[0],b[nz_b-1]);
  if(clt->spl_bz==NULL) {
    ccl_spline_free(clt->spl_nz);
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_nc_init(): error initializing spline for b(z)\n");
    return;
  }
}

static void clt_init_wM(CCL_ClTracer *clt,ccl_cosmology *cosmo,
			int nz_s,double *z_s,double *s,int *status)
{
  //Compute magnification kernel
  int nchi;
  double *x,*y;
  double dchi_here=5.;
  double zmax=clt->spl_nz->xf;
  double chimax=ccl_comoving_radial_distance(cosmo,1./(1+zmax),status);
  //TODO: The interval in chi (5. Mpc) should be made a macro

  //In this case we need to integrate all the way to z=0. Reset zmin and chimin
  clt->zmin=0;
  clt->chimin=0;
  clt->spl_sz=ccl_spline_init(nz_s,z_s,s,s[0],s[nz_s-1]);
  if(clt->spl_sz==NULL) {
    ccl_spline_free(clt->spl_nz);
    ccl_spline_free(clt->spl_bz);
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo,
				     "ccl_cls.c: clt_nc_init(): error initializing spline for s(z)\n");
    return;
  }

  nchi=(int)(chimax/dchi_here)+1;
  x=ccl_linear_spacing(0.,chimax,nchi);
  dchi_here=chimax/nchi;
  if(x==NULL || (fabs(x[0]-0)>1E-5) || (fabs(x[nchi-1]-chimax)>1e-5)) {
      ccl_spline_free(clt->spl_nz);
      ccl_spline_free(clt->spl_bz);
      ccl_spline_free(clt->spl_sz);
      *status=CCL_ERROR_LINSPACE;
      ccl_cosmology_set_status_message(cosmo,
				       "ccl_cls.c: clt_nc_init(): Error creating linear spacing in chi\n");
      return;
  }
  y=(double *)malloc(nchi*sizeof(double));
  if(y==NULL) {
    free(x);
    ccl_spline_free(clt->spl_nz);
    ccl_spline_free(clt->spl_bz);
    ccl_spline_free(clt->spl_sz);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_nc_init(): memory allocation\n");
    return;
  }

  int clstatus=0;
  for(int j=0;j<nchi;j++)
    clstatus|=window_magnification(x[j],cosmo,clt->spl_nz,clt->spl_sz,chimax,&(y[j]));
  if(clstatus) {
    free(y);
    free(x);
    ccl_spline_free(clt->spl_nz);
    ccl_spline_free(clt->spl_bz);
    ccl_spline_free(clt->spl_sz);
    *status=CCL_ERROR_INTEG;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_nc_init(): error computing lensing window\n");
    return;
  }

  clt->spl_wM=ccl_spline_init(nchi,x,y,y[0],0);
  if(clt->spl_wM==NULL) {
    free(y);
    free(x);
    ccl_spline_free(clt->spl_nz);
    ccl_spline_free(clt->spl_bz);
    ccl_spline_free(clt->spl_sz);
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo,
				     "ccl_cls.c: ccl_cl_tracer(): error initializing spline for lensing window\n");
    return;
  }
  free(x); free(y);
}

//CCL_ClTracer initializer for number counts
static void clt_nc_init(CCL_ClTracer *clt,ccl_cosmology *cosmo,
			int has_rsd,int has_magnification,
			int nz_n,double *z_n,double *n,
			int nz_b,double *z_b,double *b,
			int nz_s,double *z_s,double *s,int *status)
{
  clt->has_rsd=has_rsd;
  clt->has_magnification=has_magnification;

  clt_init_nz(clt,cosmo,nz_n,z_n,n,status);
  clt_init_bz(clt,cosmo,nz_b,z_b,b,status);
  if(clt->has_magnification)
    clt_init_wM(clt,cosmo,nz_s,z_s,s,status);
}

static void clt_init_wL(CCL_ClTracer *clt,ccl_cosmology *cosmo,
			int *status)
{
  //Compute weak lensing kernel
  int nchi;
  double *x,*y;
  double dchi_here=5.;
  double zmax=clt->spl_nz->xf;
  double chimax=ccl_comoving_radial_distance(cosmo,1./(1+zmax),status);
  //TODO: The interval in chi (5. Mpc) should be made a macro
  
  //In this case we need to integrate all the way to z=0. Reset zmin and chimin
  clt->zmin=0;
  clt->chimin=0;
  nchi=(int)(chimax/dchi_here)+1;
  x=ccl_linear_spacing(0.,chimax,nchi);
  dchi_here=chimax/nchi;
  if(x==NULL || (fabs(x[0]-0)>1E-5) || (fabs(x[nchi-1]-chimax)>1e-5)) {
    ccl_spline_free(clt->spl_nz);
    *status=CCL_ERROR_LINSPACE;
    ccl_cosmology_set_status_message(cosmo,
				     "ccl_cls.c: ccl_cl_tracer(): Error creating linear spacing in chi\n");
    return;
  }
  y=(double *)malloc(nchi*sizeof(double));
  if(y==NULL) {
    free(x);
    ccl_spline_free(clt->spl_nz);
    free(clt);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_cl_tracer(): memory allocation\n");
    return;
  }

  int clstatus=0;
  for(int j=0;j<nchi;j++)
    clstatus|=window_lensing(x[j],cosmo,clt->spl_nz,chimax,&(y[j]));
  if(clstatus) {
    free(y);
    free(x);
    ccl_spline_free(clt->spl_nz);
    *status=CCL_ERROR_INTEG;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_cl_tracer(): error computing lensing window\n");
    return;
  }
  
  clt->spl_wL=ccl_spline_init(nchi,x,y,y[0],0);
  if(clt->spl_wL==NULL) {
    free(y);
    free(x);
    ccl_spline_free(clt->spl_nz);
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo,
				     "ccl_cls.c: ccl_cl_tracer(): error initializing spline for lensing window\n");
    return;
  }
  free(x); free(y);
}

static void clt_init_rf(CCL_ClTracer *clt,ccl_cosmology *cosmo,
			int nz_rf,double *z_rf,double *rf,int *status)
{
  //Initialize bias spline
  clt->spl_rf=ccl_spline_init(nz_rf,z_rf,rf,rf[0],rf[nz_rf-1]);
  if(clt->spl_rf==NULL) {
    ccl_spline_free(clt->spl_nz);
    ccl_spline_free(clt->spl_wL);
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_nc_init(): error initializing spline for b(z)\n");
    return;
  }
}

static void clt_init_ba(CCL_ClTracer *clt,ccl_cosmology *cosmo,
			int nz_ba,double *z_ba,double *ba,int *status)
{
  //Initialize bias spline
  clt->spl_ba=ccl_spline_init(nz_ba,z_ba,ba,ba[0],ba[nz_ba-1]);
  if(clt->spl_ba==NULL) {
    ccl_spline_free(clt->spl_nz);
    ccl_spline_free(clt->spl_wL);
    ccl_spline_free(clt->spl_rf);
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_nc_init(): error initializing spline for b(z)\n");
    return;
  }
}

static void clt_wl_init(CCL_ClTracer *clt,ccl_cosmology *cosmo,
			int has_intrinsic_alignment,
			int nz_n,double *z_n,double *n,
			int nz_ba,double *z_ba,double *ba,
			int nz_rf,double *z_rf,double *rf,int *status)
{
  clt->has_intrinsic_alignment=has_intrinsic_alignment;

  clt_init_nz(clt,cosmo,nz_n,z_n,n,status);
  clt_init_wL(clt,cosmo,status);
  if(clt->has_intrinsic_alignment) {
    clt_init_rf(clt,cosmo,nz_rf,z_rf,rf,status);
    clt_init_ba(clt,cosmo,nz_ba,z_ba,ba,status);
  }
}

//CCL_ClTracer creator
//cosmo   -> ccl_cosmology object
//tracer_type -> type of tracer. Supported: ccl_number_counts_tracer, ccl_weak_lensing_tracer
//nz_n -> number of points for N(z)
//z_n  -> array of z-values for N(z)
//n    -> corresponding N(z)-values. Normalization is irrelevant
//        N(z) will be set to zero outside the range covered by z_n
//nz_b -> number of points for b(z)
//z_b  -> array of z-values for b(z)
//b    -> corresponding b(z)-values.
//        b(z) will be assumed constant outside the range covered by z_n
static CCL_ClTracer *cl_tracer(ccl_cosmology *cosmo,int tracer_type,
				   int has_rsd,int has_magnification,int has_intrinsic_alignment,
				   int nz_n,double *z_n,double *n,
				   int nz_b,double *z_b,double *b,
				   int nz_s,double *z_s,double *s,
				   int nz_ba,double *z_ba,double *ba,
				   int nz_rf,double *z_rf,double *rf,
				   double z_source, int * status)
{
  int clstatus=0;
  CCL_ClTracer *clt=(CCL_ClTracer *)malloc(sizeof(CCL_ClTracer));
  if(clt==NULL) {

    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_cl_tracer(): memory allocation\n");
    return NULL;
  }

  if ( ((cosmo->params.N_nu_mass)>0) && tracer_type==ccl_number_counts_tracer && has_rsd){
	  free(clt);
	  *status=CCL_ERROR_NOT_IMPLEMENTED;
	  ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_cl_tracer_new(): Number counts tracers with rsd not yet implemented in cosmologies with massive neutrinos.");
	  return NULL;
  }

  clt->tracer_type=tracer_type;

  double hub=cosmo->params.h*ccl_h_over_h0(cosmo,1.,status)/CLIGHT_HMPC;
  clt->prefac_lensing=1.5*hub*hub*cosmo->params.Omega_m;

  if(tracer_type==ccl_number_counts_tracer)
    clt_nc_init(clt,cosmo,has_rsd,has_magnification,
		nz_n,z_n,n,nz_b,z_b,b,nz_s,z_s,s,status);
  else if(tracer_type==ccl_weak_lensing_tracer)
    clt_wl_init(clt,cosmo,has_intrinsic_alignment,
		nz_n,z_n,n,nz_ba,z_ba,ba,nz_rf,z_rf,rf,status);
  else if(tracer_type==ccl_cmb_lensing_tracer) {
    clt->chi_source=ccl_comoving_radial_distance(cosmo,1./(1+z_source),status);
    clt->chimax=clt->chi_source;
    clt->chimin=0;
  }
  else {
    *status=CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_cl_tracer(): unknown tracer type\n");
    return NULL;
  }

  return clt;
}

//CCL_ClTracer constructor with error checking
//cosmo   -> ccl_cosmology object
//tracer_type -> type of tracer. Supported: ccl_number_counts_tracer, ccl_weak_lensing_tracer
//nz_n -> number of points for N(z)
//z_n  -> array of z-values for N(z)
//n    -> corresponding N(z)-values. Normalization is irrelevant
//        N(z) will be set to zero outside the range covered by z_n
//nz_b -> number of points for b(z)
//z_b  -> array of z-values for b(z)
//b    -> corresponding b(z)-values.
//        b(z) will be assumed constant outside the range covered by z_n
CCL_ClTracer *ccl_cl_tracer(ccl_cosmology *cosmo,int tracer_type,
				int has_rsd,int has_magnification,int has_intrinsic_alignment,
				int nz_n,double *z_n,double *n,
				int nz_b,double *z_b,double *b,
				int nz_s,double *z_s,double *s,
				int nz_ba,double *z_ba,double *ba,
				int nz_rf,double *z_rf,double *rf,
				double z_source, int * status)
{
  CCL_ClTracer *clt=cl_tracer(cosmo,tracer_type,has_rsd,has_magnification,has_intrinsic_alignment,
			      nz_n,z_n,n,nz_b,z_b,b,nz_s,z_s,s,
			      nz_ba,z_ba,ba,nz_rf,z_rf,rf,z_source,status);
  ccl_check_status(cosmo,status);
  return clt;
}

//CCL_ClTracer destructor
void ccl_cl_tracer_free(CCL_ClTracer *clt)
{
  if((clt->tracer_type==ccl_number_counts_tracer) || (clt->tracer_type==ccl_weak_lensing_tracer))
    ccl_spline_free(clt->spl_nz);

  if(clt->tracer_type==ccl_number_counts_tracer) {
    ccl_spline_free(clt->spl_bz);
    if(clt->has_magnification) {
      ccl_spline_free(clt->spl_sz);
      ccl_spline_free(clt->spl_wM);
    }
  }
  else if(clt->tracer_type==ccl_weak_lensing_tracer) {
    ccl_spline_free(clt->spl_wL);
    if(clt->has_intrinsic_alignment) {
      ccl_spline_free(clt->spl_ba);
      ccl_spline_free(clt->spl_rf);
    }
  }
  free(clt);
}

CCL_ClTracer *ccl_cl_tracer_cmblens(ccl_cosmology *cosmo,double z_source,int *status)
{
  return ccl_cl_tracer(cosmo,ccl_cmb_lensing_tracer,
			   0,0,0,
			   0,NULL,NULL,0,NULL,NULL,0,NULL,NULL,
			   0,NULL,NULL,0,NULL,NULL,z_source,status);
}

CCL_ClTracer *ccl_cl_tracer_number_counts(ccl_cosmology *cosmo,
					      int has_rsd,int has_magnification,
					      int nz_n,double *z_n,double *n,
					      int nz_b,double *z_b,double *b,
					      int nz_s,double *z_s,double *s, int * status)
{
  return ccl_cl_tracer(cosmo,ccl_number_counts_tracer,has_rsd,has_magnification,0,
			   nz_n,z_n,n,nz_b,z_b,b,nz_s,z_s,s,
			   -1,NULL,NULL,-1,NULL,NULL,0, status);
}

CCL_ClTracer *ccl_cl_tracer_number_counts_simple(ccl_cosmology *cosmo,
						     int nz_n,double *z_n,double *n,
						     int nz_b,double *z_b,double *b, int * status)
{
  return ccl_cl_tracer(cosmo,ccl_number_counts_tracer,0,0,0,
			   nz_n,z_n,n,nz_b,z_b,b,-1,NULL,NULL,
			   -1,NULL,NULL,-1,NULL,NULL,0, status);
}

CCL_ClTracer *ccl_cl_tracer_lensing(ccl_cosmology *cosmo,
					int has_alignment,
					int nz_n,double *z_n,double *n,
					int nz_ba,double *z_ba,double *ba,
					int nz_rf,double *z_rf,double *rf, int * status)
{
  return ccl_cl_tracer(cosmo,ccl_weak_lensing_tracer,0,0,has_alignment,
			   nz_n,z_n,n,-1,NULL,NULL,-1,NULL,NULL,
			   nz_ba,z_ba,ba,nz_rf,z_rf,rf,0, status);
}

CCL_ClTracer *ccl_cl_tracer_lensing_simple(ccl_cosmology *cosmo,
					       int nz_n,double *z_n,double *n, int * status)
{
  return ccl_cl_tracer(cosmo,ccl_weak_lensing_tracer,0,0,0,
			   nz_n,z_n,n,-1,NULL,NULL,-1,NULL,NULL,
			   -1,NULL,NULL,-1,NULL,NULL,0, status);
}

static double f_dens(double a,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double z=1./a-1;
  double pz=ccl_spline_eval(z,clt->spl_nz);
  double bz=ccl_spline_eval(z,clt->spl_bz);
  double h=cosmo->params.h*ccl_h_over_h0(cosmo,a,status)/CLIGHT_HMPC;

  return pz*bz*h;
}

static double f_rsd(double a,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double z=1./a-1;
  double pz=ccl_spline_eval(z,clt->spl_nz);
  double fg=ccl_growth_rate(cosmo,a,status);
  double h=cosmo->params.h*ccl_h_over_h0(cosmo,a,status)/CLIGHT_HMPC;

  return pz*fg*h;
}

static double f_mag(double a,double chi,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double wM=ccl_spline_eval(chi,clt->spl_wM);

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
//clt -> CCL_ClTracer object (must be of the ccl_number_counts_tracer type)
static double transfer_nc(int l,double k,
			  ccl_cosmology *cosmo,CCL_ClWorkspace *w,CCL_ClTracer *clt, int * status)
{
  double ret=0;
  double x0=(l+0.5);
  double chi0=x0/k;
  if(chi0<=clt->chimax) {
    double a0=ccl_scale_factor_of_chi(cosmo,chi0,status);
    double f_all=f_dens(a0,cosmo,clt,status);
    if(clt->has_rsd) {
      double x1=(l+1.5);
      double chi1=x1/k;
      if(chi1<=clt->chimax) {
	double a1=ccl_scale_factor_of_chi(cosmo,chi1,status);
	double pk0=ccl_nonlin_matter_power(cosmo,k,a0,status);
	double pk1=ccl_nonlin_matter_power(cosmo,k,a1,status);
	double fg0=f_rsd(a0,cosmo,clt,status);
	double fg1=f_rsd(a1,cosmo,clt,status);
	f_all+=fg0*(1.-l*(l-1.)/(x0*x0))-fg1*2.*sqrt((l+0.5)*pk1/((l+1.5)*pk0))/x1;
      }
    }
    if(clt->has_magnification)
      f_all+=-2*clt->prefac_lensing*l*(l+1)*f_mag(a0,chi0,cosmo,clt,status)/(k*k);
    ret=f_all;
  }

  return ret;
}

static double f_lensing(double a,double chi,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double wL=ccl_spline_eval(chi,clt->spl_wL);

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
    double pz=ccl_spline_eval(z,clt->spl_nz);
    double ba=ccl_spline_eval(z,clt->spl_ba);
    double rf=ccl_spline_eval(z,clt->spl_rf);
    double h=cosmo->params.h*ccl_h_over_h0(cosmo,a,status)/CLIGHT_HMPC;

    return pz*ba*rf*h/(chi*chi);
  }
}

//Transfer function for shear
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//w -> CCL_ClWorskpace object
//clt -> CCL_ClTracer object (must be of the ccl_weak_lensing_tracer type)
static double transfer_wl(int l,double k,
			  ccl_cosmology *cosmo,CCL_ClWorkspace *w,CCL_ClTracer *clt, int * status)
{
  double ret=0;
  double chi=(l+0.5)/k;
  if(chi<=clt->chimax) {
    double a=ccl_scale_factor_of_chi(cosmo,chi,status);
    double f_all=f_lensing(a,chi,cosmo,clt,status);
    if(clt->has_intrinsic_alignment)
      f_all+=f_IA_NLA(a,chi,cosmo,clt,status);
    
    ret=f_all;
  }

  return sqrt((l+2.)*(l+1.)*l*(l-1.))*ret/(k*k);
  //return (l+1.)*l*ret/(k*k);
}

static double transfer_cmblens(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt,int *status)
{
  double chi=(l+0.5)/k;
  if(chi>=clt->chi_source)
    return 0;

  if(chi<=clt->chimax) {
    double a=ccl_scale_factor_of_chi(cosmo,chi,status);
    double w=1-chi/clt->chi_source;
    return clt->prefac_lensing*l*(l+1.)*w/(a*chi*k*k);
  }
  return 0;
}

//Wrapper for transfer function
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//clt -> CCL_ClTracer object
static double transfer_wrap(int il,double lk,ccl_cosmology *cosmo,
			    CCL_ClWorkspace *w,CCL_ClTracer *clt, int * status)
{
  double transfer_out=0;
  double k=pow(10.,lk);

  if(clt->tracer_type==ccl_number_counts_tracer)
    transfer_out=transfer_nc(w->l_arr[il],k,cosmo,w,clt,status);
  else if(clt->tracer_type==ccl_weak_lensing_tracer)
    transfer_out=transfer_wl(w->l_arr[il],k,cosmo,w,clt,status);
  else if(clt->tracer_type==ccl_cmb_lensing_tracer)
    transfer_out=transfer_cmblens(w->l_arr[il],k,cosmo,clt,status);
  else
    transfer_out=-1;
  return transfer_out;
}

//Params for power spectrum integrand
typedef struct {
  int il;
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
  d1=transfer_wrap(p->il,lk,p->cosmo,p->w,p->clt1,p->status);
  if(d1==0)
    return 0;
  d2=transfer_wrap(p->il,lk,p->cosmo,p->w,p->clt2,p->status);
  if(d2==0)
    return 0;

  double k=pow(10.,lk);
  double chi=(p->w->l_arr[p->il]+0.5)/k;
  double a=ccl_scale_factor_of_chi(p->cosmo,chi,p->status);
  double pk=ccl_nonlin_matter_power(p->cosmo,k,a,p->status);
  
  return k*pk*d1*d2;
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
  if(l<w->l_limber) {
    //If non-Limber, we need to integrate over the whole range of k.
    *lkmin=log10(ccl_splines->K_MIN);
    *lkmax=log10(ccl_splines->K_MAX);
  }
  else {
    double chimin,chimax;
    int cut_low_1=0,cut_low_2=0;

    //Define a minimum distance only if no lensing is needed
    if((clt1->tracer_type==ccl_number_counts_tracer) && (clt1->has_magnification==0)) cut_low_1=1;
    if((clt2->tracer_type==ccl_number_counts_tracer) && (clt2->has_magnification==0)) cut_low_2=1;

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
      chimax=2*(l+0.5)/ccl_splines->K_MIN;
    }

    if(chimin<=0)
      chimin=0.5*(l+0.5)/ccl_splines->K_MAX;

    *lkmax=log10(fmin( ccl_splines->K_MAX  ,2  *(l+0.5)/chimin));
    *lkmin=log10(fmax( ccl_splines->K_MIN  ,0.5*(l+0.5)/chimax));
  }
}

//Compute angular power spectrum between two bins
//cosmo -> ccl_cosmology object
//il -> index in angular multipole array
//clt1 -> tracer #1
//clt2 -> tracer #2
static double ccl_angular_cl_native(ccl_cosmology *cosmo,CCL_ClWorkspace *cw,int il,
				    CCL_ClTracer *clt1,CCL_ClTracer *clt2,int * status)
{
  int clastatus=0, gslstatus;
  IntClPar ipar;
  double result=0,eresult;
  double lkmin,lkmax;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(ccl_gsl->N_ITERATION);

  ipar.il=il;
  ipar.cosmo=cosmo;
  ipar.w=cw;
  ipar.clt1=clt1;
  ipar.clt2=clt2;
  ipar.status = &clastatus;
  F.function=&cl_integrand;
  F.params=&ipar;
  get_k_interval(cosmo,cw,clt1,clt2,cw->l_arr[il],&lkmin,&lkmax);
  gslstatus=gsl_integration_qag(&F, lkmin, lkmax, 0,
                                ccl_gsl->INTEGRATION_LIMBER_EPSREL, ccl_gsl->N_ITERATION,
                                ccl_gsl->INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS,
                                w, &result, &eresult);
  gsl_integration_workspace_free(w);

  // Test if a round-off error occured in the evaluation of the integral
  // If so, try another integration function, more robust but potentially slower
  if(gslstatus == GSL_EROUND) {
    ccl_raise_gsl_warning(gslstatus, "ccl_cls.c: ccl_angular_cl_native(): Default GSL integration failure, attempting backup method.");
    gsl_integration_cquad_workspace *w_cquad= gsl_integration_cquad_workspace_alloc (ccl_gsl->N_ITERATION);
    size_t nevals=0;
    gslstatus=gsl_integration_cquad(&F, lkmin, lkmax, 0,
				    ccl_gsl->INTEGRATION_LIMBER_EPSREL,
				    w_cquad, &result, &eresult, &nevals);
    gsl_integration_cquad_workspace_free(w_cquad);
  }
  if(gslstatus!=GSL_SUCCESS || *ipar.status) {
    ccl_raise_gsl_warning(gslstatus, "ccl_cls.c: ccl_angular_cl_native():");
    // If an error status was already set, don't overwrite it.
    if(*status == 0){
        *status=CCL_ERROR_INTEG;
        ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_angular_cl_native(): error integrating over k\n");
    }
    return -1;
  }
  ccl_check_status(cosmo,status);

  return result*M_LN10/(cw->l_arr[il]+0.5);
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
      ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_angular_cls(); "
	     "requested l beyond range allowed by workspace\n");
      return;
    }
  }

  //Allocate array for power spectrum at interpolation nodes
  double *l_nodes=(double *)malloc(w->n_ls*sizeof(double));
  if(l_nodes==NULL) {
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_angular_cls(); memory allocation\n");
    return;
  }
  double *cl_nodes=(double *)malloc(w->n_ls*sizeof(double));
  if(cl_nodes==NULL) {
    free(l_nodes);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_cl_angular_cls(); memory allocation\n");
    return;
  }
  for(ii=0;ii<w->n_ls;ii++)
    l_nodes[ii]=(double)(w->l_arr[ii]);

  int do_angpow=0;
  //Now check if angpow is needed at all
  if(w->l_limber>0) {
    for(ii=0;ii<w->n_ls;ii++) {
      if(w->l_arr[ii]<=w->l_limber)
	do_angpow=1;
    }
  }
#ifndef HAVE_ANGPOW
  do_angpow=0;
#endif //HAVE_ANGPOW
  
  //Resort to Limber if we have lensing (this will hopefully only be temporary)
  if(clt1->tracer_type==ccl_weak_lensing_tracer || clt2->tracer_type==ccl_weak_lensing_tracer ||
     clt1->has_magnification || clt2->has_magnification) {
    do_angpow=0;
  }

  //Use angpow if non-limber is needed
  if(do_angpow)
    ccl_angular_cls_angpow(cosmo,w,clt1,clt2,cl_nodes,status);
  ccl_check_status(cosmo,status);

  //Compute limber nodes
  for(ii=0;ii<w->n_ls;ii++) {
    if((!do_angpow) || (w->l_arr[ii]>w->l_limber))
      cl_nodes[ii]=ccl_angular_cl_native(cosmo,w,ii,clt1,clt2,status);
  }

  //Interpolate into ells requested by user
  SplPar *spcl_nodes=ccl_spline_init(w->n_ls,l_nodes,cl_nodes,0,0);
  if(spcl_nodes==NULL) {
    free(cl_nodes);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_cl_angular_cls(); memory allocation\n");
    return;
  }
  for(ii=0;ii<nl_out;ii++)
    cl_out[ii]=ccl_spline_eval((double)(l_out[ii]),spcl_nodes);

  //Cleanup
  ccl_spline_free(spcl_nodes);
  free(cl_nodes);
  free(l_nodes);
}

static int check_clt_fa_inconsistency(CCL_ClTracer *clt,int func_code)
{
  if(((func_code==ccl_trf_nz) && (clt->tracer_type==ccl_cmb_lensing_tracer)) || //lensing has no n(z)
     (((func_code==ccl_trf_bz) || (func_code==ccl_trf_sz) || (func_code==ccl_trf_wM)) &&
      (clt->tracer_type!=ccl_number_counts_tracer)) || //bias and magnification only for clustering
     (((func_code==ccl_trf_rf) || (func_code==ccl_trf_ba) || (func_code==ccl_trf_wL)) &&
      (clt->tracer_type!=ccl_weak_lensing_tracer))) //IAs only for weak lensing
    return 1;
  if((((func_code==ccl_trf_sz) || (func_code==ccl_trf_wM)) &&
      (clt->has_magnification==0)) || //correct combination, but no magnification
     (((func_code==ccl_trf_rf) || (func_code==ccl_trf_ba)) &&
      (clt->has_intrinsic_alignment==0))) //Correct combination, but no IAs
    return 1;
  return 0;
}

double ccl_get_tracer_fa(ccl_cosmology *cosmo,CCL_ClTracer *clt,double a,int func_code,int *status)
{
  SplPar *spl;

  if(check_clt_fa_inconsistency(clt,func_code)) {
    *status=CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: inconsistent combination of tracer and internal function to be evaluated");
    return -1;
  }

  switch(func_code) {
  case ccl_trf_nz :
    spl=clt->spl_nz;
    break;
  case ccl_trf_bz :
    spl=clt->spl_bz;
    break;
  case ccl_trf_sz :
    spl=clt->spl_sz;
    break;
  case ccl_trf_rf :
    spl=clt->spl_rf;
    break;
  case ccl_trf_ba :
    spl=clt->spl_ba;
    break;
  case ccl_trf_wL :
    spl=clt->spl_wL;
    break;
  case ccl_trf_wM :
    spl=clt->spl_wM;
    break;
  }

  double x;
  if((func_code==ccl_trf_wL) || (func_code==ccl_trf_wM))
    x=ccl_comoving_radial_distance(cosmo,a,status); //x-variable is comoving distance for lensing kernels
  else
    x=1./a-1; //x-variable is redshift by default
  
  return ccl_spline_eval(x,spl);
}

int ccl_get_tracer_fas(ccl_cosmology *cosmo,CCL_ClTracer *clt,int na,double *a,double *fa,
		       int func_code,int *status)
{
  SplPar *spl;

  if(check_clt_fa_inconsistency(clt,func_code)) {
    *status=CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: inconsistent combination of tracer and internal function to be evaluated");
    return -1;
  }
  
  switch(func_code) {
  case ccl_trf_nz :
    spl=clt->spl_nz;
    break;
  case ccl_trf_bz :
    spl=clt->spl_bz;
    break;
  case ccl_trf_sz :
    spl=clt->spl_sz;
    break;
  case ccl_trf_rf :
    spl=clt->spl_rf;
    break;
  case ccl_trf_ba :
    spl=clt->spl_ba;
    break;
  case ccl_trf_wL :
    spl=clt->spl_wL;
    break;
  case ccl_trf_wM :
    spl=clt->spl_wM;
    break;
  }
  
  int compchi = (func_code==ccl_trf_wL) || (func_code==ccl_trf_wM);

  int ia;
  for(ia=0;ia<na;ia++) {
    double x;
    if(compchi) //x-variable is comoving distance for lensing kernels
      x=ccl_comoving_radial_distance(cosmo,a[ia],status);
    else //x-variable is redshift by default
      x=1./a[ia]-1;
    fa[ia]=ccl_spline_eval(x,spl);
  }

  return 0;
}
