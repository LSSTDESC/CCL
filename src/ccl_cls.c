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

static void ccl_cl_workspace_free(CCL_ClWorkspace *w)
{
  free(w->l_arr);
  free(w);
}

static CCL_ClWorkspace *ccl_cl_workspace_new(int lmax,int l_limber,
					     double l_logstep,int l_linstep,int *status)
{
  int i_l,l0,increment;
  CCL_ClWorkspace *w=(CCL_ClWorkspace *)malloc(sizeof(CCL_ClWorkspace));
  if(w==NULL)
    *status=CCL_ERROR_MEMORY;

  if(*status==0) {
    //Set params
    w->lmax=lmax;
    w->l_limber=l_limber;
    w->l_logstep=l_logstep;
    w->l_linstep=l_linstep;

    //Compute number of multipoles
    i_l=0; l0=0;
    increment=CCL_MAX(((int)(l0*(w->l_logstep-1.))),1);
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
    if(w->l_arr==NULL)
      *status=CCL_ERROR_MEMORY;
  }

  if(*status==0) {
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
  }

  return w;
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
  double h=p->cosmo->params.h*ccl_h_over_h0(p->cosmo,a, p->status)/ccl_constants.CLIGHT_HMPC;

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
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(cosmo->gsl_params.N_ITERATION);

  ip.chi=chi;
  ip.cosmo=cosmo;
  ip.spl_pz=spl_pz;
  ip.status = &status;
  F.function=&integrand_wl;
  F.params=&ip;
  // This conputes the lensing kernel:
  //   w_L(chi) = Integral[ dN/dchi(chi') * f(chi'-chi)/f(chi') , chi < chi' < chi_horizon ]
  // Where f(chi) is the comoving angular distance (which is just chi for zero curvature).
  gslstatus=gsl_integration_qag(&F, chi, chi_max, 0,
                                cosmo->gsl_params.INTEGRATION_EPSREL, cosmo->gsl_params.N_ITERATION,
                                cosmo->gsl_params.INTEGRATION_GAUSS_KRONROD_POINTS,
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

//Params for magnification kernel integrand
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
  double h=p->cosmo->params.h*ccl_h_over_h0(p->cosmo,a, p->status)/ccl_constants.CLIGHT_HMPC;

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
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(cosmo->gsl_params.N_ITERATION);

  ip.chi=chi;
  ip.cosmo=cosmo;
  ip.spl_pz=spl_pz;
  ip.spl_sz=spl_sz;
  ip.status = &status;
  F.function=&integrand_mag;
  F.params=&ip;
  // This conputes the magnification lensing kernel:
  //   w_M(chi) = Integral[ dN/dchi(chi') * (1-5/2 * s(chi)) * f(chi'-chi)/f(chi') , chi < chi' < chi_horizon ]
  // Where f(chi) is the comoving angular distance (which is just chi for zero curvature)
  // and s(chi) is the magnification bias parameter.
  gslstatus=gsl_integration_qag(&F, chi, chi_max, 0,
                                cosmo->gsl_params.INTEGRATION_EPSREL, cosmo->gsl_params.N_ITERATION,
                                cosmo->gsl_params.INTEGRATION_GAUSS_KRONROD_POINTS,
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
  gsl_function F;
  double nz_norm,nz_enorm;
  double *nz_normalized;
  
  //Find redshift range where the N(z) has support
  get_support_interval(nz_n,z_n,n,CCL_FRAC_RELEVANT,&(clt->zmin),&(clt->zmax));
  clt->chimax=ccl_comoving_radial_distance(cosmo,1./(1+clt->zmax),status);
  clt->chimin=ccl_comoving_radial_distance(cosmo,1./(1+clt->zmin),status);
  clt->spl_nz=ccl_spline_init(nz_n,z_n,n,0,0);
  if(clt->spl_nz==NULL) {
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_nz(): error initializing spline for N(z)\n");
  }

  if(*status==0) {
    //Normalize n(z)
    nz_normalized=(double *)malloc(nz_n*sizeof(double));
    if(nz_normalized==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_nz(): memory allocation\n");
      return;
    }
  }
  
  if(*status==0) {
    gsl_integration_workspace *w=gsl_integration_workspace_alloc(cosmo->gsl_params.N_ITERATION);
    F.function=&speval_bis;
    F.params=clt->spl_nz;
    //Here we're just integrating the N(z) to normalize it to unit probability.
    gslstatus=gsl_integration_qag(&F, z_n[0], z_n[nz_n-1], 0,
				  cosmo->gsl_params.INTEGRATION_EPSREL, cosmo->gsl_params.N_ITERATION,
				  cosmo->gsl_params.INTEGRATION_GAUSS_KRONROD_POINTS,
				  w, &nz_norm, &nz_enorm);
    gsl_integration_workspace_free(w);
    if(gslstatus!=GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_cls.c: clt_init_nz():");
      *status=CCL_ERROR_INTEG;
      ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_nz(): integration error when normalizing N(z)\n");
    }
  }
  
  if(*status==0) {
    for(int ii=0;ii<nz_n;ii++)
      nz_normalized[ii]=n[ii]/nz_norm;
    ccl_spline_free(clt->spl_nz);
    clt->spl_nz=ccl_spline_init(nz_n,z_n,nz_normalized,0,0);
    if(clt->spl_nz==NULL) {
      *status=CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_nz(): error initializing normalized spline for N(z)\n");
    }
  }
  
  free(nz_normalized);
}


static void clt_init_bz(CCL_ClTracer *clt,ccl_cosmology *cosmo,
			int nz_b,double *z_b,double *b,int *status)
{
  //Initialize bias spline
  clt->spl_bz=ccl_spline_init(nz_b,z_b,b,b[0],b[nz_b-1]);
  if(clt->spl_bz==NULL) {
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_bz(): error initializing spline for b(z)\n");
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
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo,
				     "ccl_cls.c: clt_init_wM(): error initializing spline for s(z)\n");
  }

  if(*status==0) {
    nchi=(int)(chimax/dchi_here)+1;
    x=ccl_linear_spacing(0.,chimax,nchi);
    dchi_here=chimax/nchi;
    if(x==NULL || (fabs(x[0]-0)>1E-5) || (fabs(x[nchi-1]-chimax)>1e-5)) {
      *status=CCL_ERROR_LINSPACE;
      ccl_cosmology_set_status_message(cosmo,
				       "ccl_cls.c: clt_init_wM(): Error creating linear spacing in chi\n");
    }
  }

  if(*status==0) {
    y=(double *)malloc(nchi*sizeof(double));
    if(y==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_wM(): memory allocation\n");
    }
  }

  if(*status==0) {
    int clstatus=0;
    for(int j=0;j<nchi;j++){
      clstatus|=window_magnification(x[j],cosmo,clt->spl_nz,clt->spl_sz,chimax,&(y[j]));
      // If mu / Sigma parameterisation of modified gravity is in effect,
	  // add appropriate factors of Sigma before splining:
      if ( fabs(cosmo->params.sigma_0) ){
		    y[j] = y[j] * (1. + ccl_Sig_MG(cosmo,ccl_scale_factor_of_chi(cosmo,x[j], status), status));
	     }
	  }
    if(clstatus) {
      *status=CCL_ERROR_INTEG;
      ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_wM(): error computing lensing window\n");
    }
    if(*status){
	  ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_wM(): error computing MG factor\n");	
	}
    
  }

  if(*status==0) {
    clt->spl_wM=ccl_spline_init(nchi,x,y,y[0],0);
    if(clt->spl_wM==NULL) {
      *status=CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(cosmo,
				       "ccl_cls.c: clt_init_wM(): error initializing spline for lensing window\n");
    }
  }
  free(x); free(y);
}

//CCL_ClTracer initializer for number counts
static void clt_nc_init(CCL_ClTracer *clt,ccl_cosmology *cosmo,
			int has_density,int has_rsd,int has_magnification,
			int nz_n,double *z_n,double *n,
			int nz_b,double *z_b,double *b,
			int nz_s,double *z_s,double *s,int *status)
{
  clt->has_density=has_density;
  clt->has_rsd=has_rsd;
  clt->has_magnification=has_magnification;
  clt->has_shear=0;
  clt->has_intrinsic_alignment=0;

  if ( ((cosmo->params.N_nu_mass)>0) && clt->has_rsd){
    *status=CCL_ERROR_NOT_IMPLEMENTED;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_cl_tracer_new(): Number counts tracers with RSD not yet implemented in cosmologies with massive neutrinos.");
    return;
  }

  clt_init_nz(clt,cosmo,nz_n,z_n,n,status);
  if(clt->has_density)
    clt_init_bz(clt,cosmo,nz_b,z_b,b,status);
  if(clt->has_magnification)
    // If magnification is present within mu / Sigma parameterisation
    // of modified gravity, that is accounted for in this function.
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
    *status=CCL_ERROR_LINSPACE;
    ccl_cosmology_set_status_message(cosmo,
				     "ccl_cls.c: clt_init_wL(): Error creating linear spacing in chi\n");
  }
  
  if(*status==0) {
    y=(double *)malloc(nchi*sizeof(double));
    if(y==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_wL(): memory allocation\n");
    }
  }

  if(*status==0) {
    int clstatus=0;
    for(int j=0;j<nchi;j++){
      clstatus|=window_lensing(x[j],cosmo,clt->spl_nz,chimax,&(y[j]));
      // If mu / Sigma parameterisation of modified gravity is in effect,
	  // add appropriate factors of Sigma before splining:
      if ( fabs(cosmo->params.sigma_0) ){
		    y[j] = y[j] * (1. + ccl_Sig_MG(cosmo,ccl_scale_factor_of_chi(cosmo,x[j], status), status));
	     }
	  }
    if(clstatus) {
      *status=CCL_ERROR_INTEG;
      ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_wL(): error computing lensing window\n");
    }
    if(*status){
	  ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_wL(): error computing MG factor\n"); 	
      }
  }

  if(*status==0) {
    clt->spl_wL=ccl_spline_init(nchi,x,y,y[0],0);
    if(clt->spl_wL==NULL) {
      *status=CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(cosmo,
				     "ccl_cls.c: clt_init_wL(): error initializing spline for lensing window\n");
    }
  }
  free(x); free(y);
}

static void clt_init_rf(CCL_ClTracer *clt,ccl_cosmology *cosmo,
			int nz_rf,double *z_rf,double *rf,int *status)
{
  //Initialize bias spline
  clt->spl_rf=ccl_spline_init(nz_rf,z_rf,rf,rf[0],rf[nz_rf-1]);
  if(clt->spl_rf==NULL) {
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_rf(): error initializing spline for b(z)\n");
  }
}

static void clt_init_ba(CCL_ClTracer *clt,ccl_cosmology *cosmo,
			int nz_ba,double *z_ba,double *ba,int *status)
{
  //Initialize bias spline
  clt->spl_ba=ccl_spline_init(nz_ba,z_ba,ba,ba[0],ba[nz_ba-1]);
  if(clt->spl_ba==NULL) {
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: clt_init_ba(): error initializing spline for b(z)\n");
  }
}

static void clt_wl_init(CCL_ClTracer *clt,ccl_cosmology *cosmo,
			int has_shear,int has_intrinsic_alignment,
			int nz_n,double *z_n,double *n,
			int nz_ba,double *z_ba,double *ba,
			int nz_rf,double *z_rf,double *rf,int *status)
{
  clt->has_density=0;
  clt->has_rsd=0;
  clt->has_magnification=0;
  clt->has_shear=has_shear;
  clt->has_intrinsic_alignment=has_intrinsic_alignment;

  clt_init_nz(clt,cosmo,nz_n,z_n,n,status);
  if(clt->has_shear)
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
			       int has_density,int has_rsd,int has_magnification,
			       int has_shear,int has_intrinsic_alignment,
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
  }

  if(*status==0) {
    clt->tracer_type=tracer_type;
    
    double hub=cosmo->params.h*ccl_h_over_h0(cosmo,1.,status)/ccl_constants.CLIGHT_HMPC;
    clt->prefac_lensing=1.5*hub*hub*cosmo->params.Omega_m;

    if(tracer_type==ccl_number_counts_tracer)
      clt_nc_init(clt,cosmo,has_density,has_rsd,has_magnification,
		  nz_n,z_n,n,nz_b,z_b,b,nz_s,z_s,s,status);
    else if(tracer_type==ccl_weak_lensing_tracer)
      clt_wl_init(clt,cosmo,has_shear,has_intrinsic_alignment,
		  nz_n,z_n,n,nz_ba,z_ba,ba,nz_rf,z_rf,rf,status);
    else if(tracer_type==ccl_cmb_lensing_tracer) {
      clt->chi_source=ccl_comoving_radial_distance(cosmo,1./(1+z_source),status);
      clt->chimax=clt->chi_source;
      clt->chimin=0;
    }
    else {
      free(clt);
      *status=CCL_ERROR_INCONSISTENT;
      ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_cl_tracer(): unknown tracer type\n");
      return NULL;
    }
  }

  if(*status) {
    free(clt);
    clt=NULL;
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
                            int has_density,int has_rsd,int has_magnification,
			                int has_shear,int has_intrinsic_alignment,
			                int nz_n,double *z_n,double *n,
			                int nz_b,double *z_b,double *b,
			                int nz_s,double *z_s,double *s,
			                int nz_ba,double *z_ba,double *ba,
			                int nz_rf,double *z_rf,double *rf,
			                double z_source, int * status)
{	  	  

  CCL_ClTracer *clt=cl_tracer(cosmo,tracer_type,has_density,has_rsd,has_magnification,
			      has_shear,has_intrinsic_alignment,
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
    if(clt->has_density)
      ccl_spline_free(clt->spl_bz);
    if(clt->has_magnification) {
      ccl_spline_free(clt->spl_sz);
      ccl_spline_free(clt->spl_wM);
    }
  }
  else if(clt->tracer_type==ccl_weak_lensing_tracer) {
    if(clt->has_shear)
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
		       0,0,0,0,0,
		       0,NULL,NULL,0,NULL,NULL,0,NULL,NULL,
		       0,NULL,NULL,0,NULL,NULL,z_source,status);
}

CCL_ClTracer *ccl_cl_tracer_number_counts(ccl_cosmology *cosmo,
					  int has_density,int has_rsd,int has_magnification,
					  int nz_n,double *z_n,double *n,
					  int nz_b,double *z_b,double *b,
					  int nz_s,double *z_s,double *s, int * status)
{
  return ccl_cl_tracer(cosmo,ccl_number_counts_tracer,
		       has_density,has_rsd,has_magnification,0,0,
		       nz_n,z_n,n,nz_b,z_b,b,nz_s,z_s,s,
		       -1,NULL,NULL,-1,NULL,NULL,0, status);
}

CCL_ClTracer *ccl_cl_tracer_number_counts_simple(ccl_cosmology *cosmo,
						     int nz_n,double *z_n,double *n,
						     int nz_b,double *z_b,double *b, int * status)
{
  return ccl_cl_tracer(cosmo,ccl_number_counts_tracer,1,0,0,0,0,
		       nz_n,z_n,n,nz_b,z_b,b,-1,NULL,NULL,
		       -1,NULL,NULL,-1,NULL,NULL,0, status);
}

CCL_ClTracer *ccl_cl_tracer_lensing(ccl_cosmology *cosmo,
				    int has_shear,int has_alignment,
				    int nz_n,double *z_n,double *n,
				    int nz_ba,double *z_ba,double *ba,
				    int nz_rf,double *z_rf,double *rf, int * status)
{
  return ccl_cl_tracer(cosmo,ccl_weak_lensing_tracer,0,0,0,has_shear,has_alignment,
		       nz_n,z_n,n,-1,NULL,NULL,-1,NULL,NULL,
		       nz_ba,z_ba,ba,nz_rf,z_rf,rf,0, status);
}

CCL_ClTracer *ccl_cl_tracer_lensing_simple(ccl_cosmology *cosmo,
					   int nz_n,double *z_n,double *n, int * status)
{
  return ccl_cl_tracer(cosmo,ccl_weak_lensing_tracer,0,0,0,1,0,
		       nz_n,z_n,n,-1,NULL,NULL,-1,NULL,NULL,
		       -1,NULL,NULL,-1,NULL,NULL,0, status);
}

static double f_dens(double a,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double z=1./a-1;
  double pz=ccl_spline_eval(z,clt->spl_nz);
  double bz=ccl_spline_eval(z,clt->spl_bz);
  double h=cosmo->params.h*ccl_h_over_h0(cosmo,a,status)/ccl_constants.CLIGHT_HMPC;

  return pz*bz*h;
}

static double f_rsd(double a,ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double z=1./a-1;
  double pz=ccl_spline_eval(z,clt->spl_nz);
  double fg=ccl_growth_rate(cosmo,a,status);
  double h=cosmo->params.h*ccl_h_over_h0(cosmo,a,status)/ccl_constants.CLIGHT_HMPC;

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
static double transfer_nc(double l,double k,
			  ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double ret=0;
  double x0=(l+0.5);
  double chi0=x0/k;
  if(chi0<=clt->chimax) {
    double a0=ccl_scale_factor_of_chi(cosmo,chi0,status);
    double f_all=0;
    if(clt->has_density)
      f_all+=f_dens(a0,cosmo,clt,status);
    if(clt->has_rsd) {
      double x1=(l+1.5);
      double chi1=x1/k;
      if(chi1<=clt->chimax) {
	double a1=ccl_scale_factor_of_chi(cosmo,chi1,status);
	// if mu / Sigma parameterisation of modified gravity is in effect,
	// pk0 and pk1 will be modified power spectra affected mu0
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
    double h=cosmo->params.h*ccl_h_over_h0(cosmo,a,status)/ccl_constants.CLIGHT_HMPC;

    return pz*ba*rf*h/(chi*chi);
  }
}

//Transfer function for shear
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//clt -> CCL_ClTracer object (must be of the ccl_weak_lensing_tracer type)
static double transfer_wl(double l,double k,
			  ccl_cosmology *cosmo,CCL_ClTracer *clt, int * status)
{
  double ret=0;
  double chi=(l+0.5)/k;
  if(chi<=clt->chimax) {
    double a=ccl_scale_factor_of_chi(cosmo,chi,status);
    double f_all=0;
    if(clt->has_shear)
      f_all+=f_lensing(a,chi,cosmo,clt,status);
    if(clt->has_intrinsic_alignment)
      f_all+=f_IA_NLA(a,chi,cosmo,clt,status);
    
    ret=f_all;
  }

  return sqrt((l+2.)*(l+1.)*l*(l-1.))*ret/(k*k);
}

static double transfer_cmblens(int l,double k,ccl_cosmology *cosmo,CCL_ClTracer *clt,int *status)
{
  double chi=(l+0.5)/k;
  if(chi>=clt->chi_source)
    return 0;

  if(chi<=clt->chimax) {
    double a=ccl_scale_factor_of_chi(cosmo,chi,status);
    double w=1-chi/clt->chi_source;
    // If muSigma parameterisation of gravity is in effect and 
    // Sigma0>0, add the relevant factor here.
    if (fabs(cosmo->params.sigma_0)>1e-15){
        w = w * (1. + ccl_Sig_MG(cosmo,ccl_scale_factor_of_chi(cosmo,chi, status), status)); 
    }
    if (*status){
		ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: transfer_cmblens: error computing MG factor\n"); 	
      }
		       
    return clt->prefac_lensing*l*(l+1.)*w/(a*chi*k*k);
  }
  return 0;
}

//Wrapper for transfer function
//l -> angular multipole
//k -> wavenumber modulus
//cosmo -> ccl_cosmology object
//clt -> CCL_ClTracer object
static double transfer_wrap(double l,double k,ccl_cosmology *cosmo,
			    CCL_ClTracer *clt, int * status)
{
  double transfer_out=0;

  if(clt->tracer_type==ccl_number_counts_tracer)
    transfer_out=transfer_nc(l,k,cosmo,clt,status);
  else if(clt->tracer_type==ccl_weak_lensing_tracer)
    transfer_out=transfer_wl(l,k,cosmo,clt,status);
  else if(clt->tracer_type==ccl_cmb_lensing_tracer)
    transfer_out=transfer_cmblens(l,k,cosmo,clt,status);
  else
    transfer_out=-1;
  return transfer_out;
}

//Params for power spectrum integrand
typedef struct {
  double l;
  ccl_cosmology *cosmo;
  CCL_ClTracer *clt1;
  CCL_ClTracer *clt2;
  ccl_p2d_t *psp;
  int *status;
} IntClPar;

//Integrand for integral power spectrum
static double cl_integrand(double lk,void *params)
{
  double d1,d2;
  IntClPar *p=(IntClPar *)params;
  double k=exp(lk);
  d1=transfer_wrap(p->l,k,p->cosmo,p->clt1,p->status);
  if(d1==0)
    return 0;
  d2=transfer_wrap(p->l,k,p->cosmo,p->clt2,p->status);
  if(d2==0)
    return 0;

  double chi=(p->l+0.5)/k;
  double a=ccl_scale_factor_of_chi(p->cosmo,chi,p->status);
  double pk=ccl_p2d_t_eval(p->psp,lk,a,p->cosmo,p->status);
  
  return k*pk*d1*d2;
}

//Figure out k intervals where the Limber kernel has support
//clt1 -> tracer #1
//clt2 -> tracer #2
//l    -> angular multipole
//lkmin, lkmax -> log of the range of scales where the transfer functions have support
static void get_k_interval(ccl_cosmology *cosmo,
			   CCL_ClTracer *clt1,CCL_ClTracer *clt2,double l,
			   double *lkmin,double *lkmax)
{
  double chimin,chimax;
  int cut_low_1=0,cut_low_2=0;

  // The next couple of lines determine whether the transfer function of a given
  // tracer is localized in redshift. This is important in order to determine optimal
  // integration limits for the Limber integrator. Otherwise the GSL integration
  // methods can sometimes just take values in ranges where the transfer function
  // is zero and just return zero after a few function evaluations.
  // This is also a good idea in order to speed up the Limber integrator a bit.
  // The only contributions that would make a tracer not localized in redshift would
  // be lensing shear, lensing magnification or CMB lensing, all of which have a
  // cumulative kernel.
  if((clt1->has_shear==0) &&
     (clt1->has_magnification==0) &&
     (clt1->tracer_type!=ccl_cmb_lensing_tracer)) cut_low_1=1;
  if((clt2->has_shear==0) &&
     (clt2->has_magnification==0) &&
     (clt2->tracer_type!=ccl_cmb_lensing_tracer)) cut_low_2=1;
  
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
    chimin=0.5*(l+0.5)/cosmo->spline_params.K_MAX;
    chimax=2*(l+0.5)/cosmo->spline_params.K_MIN;
  }
  
  if(chimin<=0)
    chimin=0.5*(l+0.5)/cosmo->spline_params.K_MAX;
  
  *lkmax=log(fmin( cosmo->spline_params.K_MAX  ,2  *(l+0.5)/chimin));
  *lkmin=log(fmax( cosmo->spline_params.K_MIN  ,0.5*(l+0.5)/chimax));
}

//Compute angular power spectrum between two bins using Limber approximation
//cosmo -> ccl_cosmology object
//il -> index in angular multipole array
//clt1 -> tracer #1
//clt2 -> tracer #2
//psp -> 3D power spectrum to integrate over
double ccl_angular_cl_limber(ccl_cosmology *cosmo,
			     CCL_ClTracer *clt1,CCL_ClTracer *clt2,
			     ccl_p2d_t *psp,double l,int * status)
{
  int clastatus=0, gslstatus;
  IntClPar ipar;
  double result=0,eresult;
  double lkmin,lkmax;
  ccl_p2d_t *psp_use;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(cosmo->gsl_params.N_ITERATION);

  if(psp==NULL) {
    if (!cosmo->computed_power) ccl_cosmology_compute_power(cosmo, status);
    // Return if compilation failed
    if (!cosmo->computed_power) return NAN;
    // If muSigma modification to gravity is in effect, this p(k)
    // will be modified by mu_0.
    psp_use=cosmo->data.p_nl;
  }
  else
    psp_use=psp;

  ipar.l=l;
  ipar.cosmo=cosmo;
  ipar.clt1=clt1;
  ipar.clt2=clt2;
  ipar.psp=psp_use;
  ipar.status = &clastatus;
  F.function=&cl_integrand;
  F.params=&ipar;
  get_k_interval(cosmo,clt1,clt2,l,&lkmin,&lkmax);
  // This computes the angular power spectra in the Limber approximation between two quantities a and b:
  //  C_ell^ab = 2/(2*ell+1) * Integral[ Delta^a_ell(k) Delta^b_ell(k) * P(k) , k_min < k < k_max ]
  // Note that we use log(k) as an integration variable, and the ell-dependent prefactor is included
  // at the end of this function.
  gslstatus=gsl_integration_qag(&F, lkmin, lkmax, 0,
                                cosmo->gsl_params.INTEGRATION_LIMBER_EPSREL,
				cosmo->gsl_params.N_ITERATION,
                                cosmo->gsl_params.INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS,
                                w, &result, &eresult);
  gsl_integration_workspace_free(w);

  // Test if a round-off error occured in the evaluation of the integral
  // If so, try another integration function, more robust but potentially slower
  if(gslstatus == GSL_EROUND) {
    ccl_raise_gsl_warning(gslstatus, "ccl_cls.c: ccl_angular_cl_native(): Default GSL integration failure, attempting backup method.");
    gsl_integration_cquad_workspace *w_cquad= gsl_integration_cquad_workspace_alloc (cosmo->gsl_params.N_ITERATION);
    size_t nevals=0;
    gslstatus=gsl_integration_cquad(&F, lkmin, lkmax, 0,
				    cosmo->gsl_params.INTEGRATION_LIMBER_EPSREL,
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

  return result/(l+0.5);
}

void ccl_angular_cls_nonlimber(ccl_cosmology *cosmo,double l_logstep,int l_linstep,
			       CCL_ClTracer *clt1,CCL_ClTracer *clt2,ccl_p2d_t *psp,
			       int nl_out,int *l_out,double *cl_out,int *status)
{
  int ii,lmax;
  double *l_nodes=NULL,*cl_nodes=NULL;
  SplPar *spcl_nodes=NULL;
  CCL_ClWorkspace *w=NULL;

  //Check if we can use ANGPOW at all
#ifndef HAVE_ANGPOW
  *status=CCL_ERROR_INCONSISTENT;
  ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_angular_cls_nonlimber(): non-Limber integrator not loaded\n");
#endif //HAVE_ANGPOW

  if(*status==0) { //Check if the conditions for ANGPOW apply
    if(clt1->tracer_type!=ccl_number_counts_tracer ||
       clt2->tracer_type!=ccl_number_counts_tracer ||
       clt1->has_magnification ||
       clt2->has_magnification) {
      *status=CCL_ERROR_INCONSISTENT;
      ccl_cosmology_set_status_message(cosmo,
				       "ccl_cls.c: ccl_angular_cls_nonlimber(): "
				       "non-Limber integrator only implemented "
				       "for galaxy clustering without magnification\n");
    }
  }

  if(*status==0) {
    //First, find maximum ell
    lmax=0;
    for(ii=0;ii<nl_out;ii++) {
      if(l_out[ii]>lmax)
	lmax=l_out[ii];
    }
  
    //Now initialize workspace
    w=ccl_cl_workspace_new(lmax,2*lmax,l_logstep,l_linstep,status);
  }
  
  if(*status==0) {
    //Allocate array for power spectrum at interpolation nodes
    l_nodes=(double *)malloc(w->n_ls*sizeof(double));
    if(l_nodes==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_angular_cls(); memory allocation\n");
    }
  }

  if(*status==0) {
    cl_nodes=(double *)malloc(w->n_ls*sizeof(double));
    if(cl_nodes==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_cl_angular_cls(); memory allocation\n");
    }
  }

  if(*status==0) {
    //Compute power spectra at interpolation nodes
    for(ii=0;ii<w->n_ls;ii++)
      l_nodes[ii]=(double)(w->l_arr[ii]);
    ccl_angular_cls_angpow(cosmo,w,clt1,clt2,cl_nodes,status);
    ccl_check_status(cosmo,status);
  }
    
  if(*status==0) {
    //Interpolate into ells requested by user
    spcl_nodes=ccl_spline_init(w->n_ls,l_nodes,cl_nodes,0,0);
    if(spcl_nodes==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_cls.c: ccl_cl_angular_cls(); memory allocation\n");
    }
  }  
  
  if(*status==0) {
    //Interpolate into input multipoles
    for(ii=0;ii<nl_out;ii++)
      cl_out[ii]=ccl_spline_eval((double)(l_out[ii]),spcl_nodes);
  }
  
  //Cleanup
  if(spcl_nodes!=NULL)
    ccl_spline_free(spcl_nodes);
  free(cl_nodes);
  free(l_nodes);
  if(w!=NULL)
    ccl_cl_workspace_free(w);
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
