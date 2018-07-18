#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "gsl/gsl_errno.h"
#include "gsl/gsl_integration.h"
#include "ccl_cls.h"
#include "ccl_power.h"
#include "ccl_background.h"
#include "ccl_error.h"
#include "ccl_utils.h"
#include "ccl_params.h"

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

CCL_ClWorkspace *ccl_cl_workspace_default(int lmax,int l_limber,int non_limber_method,
					  double l_logstep,int l_linstep,
					  double dchi,double dlk,double zmin,int *status)
{
  CCL_ClWorkspace *w=(CCL_ClWorkspace *)malloc(sizeof(CCL_ClWorkspace));
  if(w==NULL) {
    *status=CCL_ERROR_MEMORY;
    //Can't access cosmology object
    //    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_workspace_default(); memory allocation\n");
    return NULL;
  }

  //Set params
  w->dchi=dchi;
  w->dlk =dlk ;
  if(zmin<=0) { //We should make sure that zmin is always strictly positive
    free(w);
    *status=CCL_ERROR_INCONSISTENT;
    return NULL;
  }
  w->zmin=zmin;
  w->lmax=lmax;
  if((non_limber_method!=CCL_NONLIMBER_METHOD_NATIVE) && (non_limber_method!=CCL_NONLIMBER_METHOD_ANGPOW)) {
    free(w);
    *status=CCL_ERROR_INCONSISTENT;
    //Can't access cosmology object
    //    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_workspace_default(); unknown non-limber method\n");
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
    //    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_workspace_default(); memory allocation\n");
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

CCL_ClWorkspace *ccl_cl_workspace_default_limber(int lmax,double l_logstep,int l_linstep,
						 double dlk,int *status)
{
  return ccl_cl_workspace_default(lmax,-1,CCL_NONLIMBER_METHOD_NATIVE,l_logstep,l_linstep,3.,dlk,0.05,status);
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
//EK: added local status here as the status testing is done in routines called from this function
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
static CCL_ClTracer *cl_tracer(ccl_cosmology *cosmo,int tracer_type,
				   int has_rsd,int has_magnification,int has_intrinsic_alignment,
				   int nz_n,double *z_n,double *n,
				   int nz_b,double *z_b,double *b,
				   int nz_s,double *z_s,double *s,
				   int nz_ba,double *z_ba,double *ba,
				   int nz_rf,double *z_rf,double *rf,
				   double z_source, int * status)
{
  int clstatus=0, gslstatus;
  CCL_ClTracer *clt=(CCL_ClTracer *)malloc(sizeof(CCL_ClTracer));
  if(clt==NULL) {

    *status=CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer(): memory allocation\n");
    return NULL;
  }

  if ( ((cosmo->params.N_nu_mass)>0) && tracer_type==CL_TRACER_NC && has_rsd){
	  free(clt);
	  *status=CCL_ERROR_NOT_IMPLEMENTED;
	  strcpy(cosmo->status_message, "ccl_cls.c: ccl_cl_tracer_new(): Number counts tracers with rsd not yet implemented in cosmologies with massive neutrinos.");
	  return NULL;
  }

  clt->tracer_type=tracer_type;
  clt->computed_transfer=0;

  double hub=cosmo->params.h*ccl_h_over_h0(cosmo,1.,status)/CLIGHT_HMPC;
  clt->prefac_lensing=1.5*hub*hub*cosmo->params.Omega_m;

  if((tracer_type==CL_TRACER_NC)||(tracer_type==CL_TRACER_WL)) {
    get_support_interval(nz_n,z_n,n,CCL_FRAC_RELEVANT,&(clt->zmin),&(clt->zmax));
    clt->chimax=ccl_comoving_radial_distance(cosmo,1./(1+clt->zmax),status);
    clt->chimin=ccl_comoving_radial_distance(cosmo,1./(1+clt->zmin),status);
    clt->spl_nz=ccl_spline_init(nz_n,z_n,n,0,0);
    if(clt->spl_nz==NULL) {
      free(clt);
      *status=CCL_ERROR_SPLINE;
      strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer(): error initializing spline for N(z)\n");
      return NULL;
    }

    //Normalize n(z)
    gsl_function F;
    double nz_norm,nz_enorm;
    double *nz_normalized=(double *)malloc(nz_n*sizeof(double));
    if(nz_normalized==NULL) {
      ccl_spline_free(clt->spl_nz);
      free(clt);
      *status=CCL_ERROR_MEMORY;
      strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer(): memory allocation\n");
      return NULL;
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
      ccl_raise_gsl_warning(gslstatus, "ccl_cls.c: cl_tracer():");
      ccl_spline_free(clt->spl_nz);
      free(clt);
      *status=CCL_ERROR_INTEG;
      strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer(): integration error when normalizing N(z)\n");
      return NULL;
    }
    for(int ii=0;ii<nz_n;ii++)
      nz_normalized[ii]=n[ii]/nz_norm;
    ccl_spline_free(clt->spl_nz);
    clt->spl_nz=ccl_spline_init(nz_n,z_n,nz_normalized,0,0);
    free(nz_normalized);
    if(clt->spl_nz==NULL) {
      free(clt);
      *status=CCL_ERROR_SPLINE;
      strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer(): error initializing normalized spline for N(z)\n");
      return NULL;
    }

    if(tracer_type==CL_TRACER_NC) {
      //Initialize bias spline
      clt->spl_bz=ccl_spline_init(nz_b,z_b,b,b[0],b[nz_b-1]);
      if(clt->spl_bz==NULL) {
	ccl_spline_free(clt->spl_nz);
	free(clt);
	*status=CCL_ERROR_SPLINE;
	strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer(): error initializing spline for b(z)\n");
	return NULL;
      }
      clt->has_rsd=has_rsd;
      clt->has_magnification=has_magnification;
      if(clt->has_magnification) {
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
	clt->spl_sz=ccl_spline_init(nz_s,z_s,s,s[0],s[nz_s-1]);
	if(clt->spl_sz==NULL) {
	  ccl_spline_free(clt->spl_nz);
	  ccl_spline_free(clt->spl_bz);
	  free(clt);
	  *status=CCL_ERROR_SPLINE;
	  strcpy(cosmo->status_message,
		 "ccl_cls.c: ccl_cl_tracer(): error initializing spline for s(z)\n");
	  return NULL;
	}

	nchi=(int)(chimax/dchi_here)+1;
	x=ccl_linear_spacing(0.,chimax,nchi);
	dchi_here=chimax/nchi;
	if(x==NULL || (fabs(x[0]-0)>1E-5) || (fabs(x[nchi-1]-chimax)>1e-5)) {
	  ccl_spline_free(clt->spl_nz);
	  ccl_spline_free(clt->spl_bz);
	  ccl_spline_free(clt->spl_sz);
	  free(clt);
	  *status=CCL_ERROR_LINSPACE;
	  strcpy(cosmo->status_message,
		 "ccl_cls.c: ccl_cl_tracer(): Error creating linear spacing in chi\n");
	  return NULL;
	}
	y=(double *)malloc(nchi*sizeof(double));
	if(y==NULL) {
	  free(x);
	  ccl_spline_free(clt->spl_nz);
	  ccl_spline_free(clt->spl_bz);
	  ccl_spline_free(clt->spl_sz);
	  free(clt);
	  *status=CCL_ERROR_MEMORY;
	  strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer(): memory allocation\n");
	  return NULL;
	}

	for(int j=0;j<nchi;j++)
	  clstatus|=window_magnification(x[j],cosmo,clt->spl_nz,clt->spl_sz,chimax,&(y[j]));
	if(clstatus) {
	  free(y);
	  free(x);
	  ccl_spline_free(clt->spl_nz);
	  ccl_spline_free(clt->spl_bz);
	  ccl_spline_free(clt->spl_sz);
	  free(clt);
	  *status=CCL_ERROR_INTEG;
	  strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer(): error computing lensing window\n");
	  return NULL;
	}

	clt->spl_wM=ccl_spline_init(nchi,x,y,y[0],0);
	if(clt->spl_wM==NULL) {
	  free(y);
	  free(x);
	  ccl_spline_free(clt->spl_nz);
	  ccl_spline_free(clt->spl_bz);
	  ccl_spline_free(clt->spl_sz);
	  free(clt);
	  *status=CCL_ERROR_SPLINE;
	  strcpy(cosmo->status_message,
		 "ccl_cls.c: ccl_cl_tracer(): error initializing spline for lensing window\n");
	  return NULL;
	}
	free(x); free(y);
      }
    }
    else if(tracer_type==CL_TRACER_WL) {
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
	free(clt);
	*status=CCL_ERROR_LINSPACE;
	strcpy(cosmo->status_message,
	       "ccl_cls.c: ccl_cl_tracer(): Error creating linear spacing in chi\n");
	return NULL;
      }
      y=(double *)malloc(nchi*sizeof(double));
      if(y==NULL) {
	free(x);
	ccl_spline_free(clt->spl_nz);
	free(clt);
	*status=CCL_ERROR_MEMORY;
	strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer(): memory allocation\n");
	return NULL;
      }

      for(int j=0;j<nchi;j++)
	clstatus|=window_lensing(x[j],cosmo,clt->spl_nz,chimax,&(y[j]));
      if(clstatus) {
	free(y);
	free(x);
	ccl_spline_free(clt->spl_nz);
	free(clt);
	*status=CCL_ERROR_INTEG;
	strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer(): error computing lensing window\n");
	return NULL;
      }

      clt->spl_wL=ccl_spline_init(nchi,x,y,y[0],0);
      if(clt->spl_wL==NULL) {
	free(y);
	free(x);
	ccl_spline_free(clt->spl_nz);
	free(clt);
	*status=CCL_ERROR_SPLINE;
	strcpy(cosmo->status_message,
	       "ccl_cls.c: ccl_cl_tracer(): error initializing spline for lensing window\n");
	return NULL;
      }
      free(x); free(y);

      clt->has_intrinsic_alignment=has_intrinsic_alignment;
      if(clt->has_intrinsic_alignment) {
	clt->spl_rf=ccl_spline_init(nz_rf,z_rf,rf,rf[0],rf[nz_rf-1]);
	if(clt->spl_rf==NULL) {
	  ccl_spline_free(clt->spl_nz);
	  free(clt);
	  *status=CCL_ERROR_SPLINE;
	  strcpy(cosmo->status_message,
		 "ccl_cls.c: ccl_cl_tracer(): error initializing spline for rf(z)\n");
	  return NULL;
	}
	clt->spl_ba=ccl_spline_init(nz_ba,z_ba,ba,ba[0],ba[nz_ba-1]);
	if(clt->spl_ba==NULL) {
	  ccl_spline_free(clt->spl_rf);
	  ccl_spline_free(clt->spl_nz);
	  free(clt);
	  *status=CCL_ERROR_SPLINE;
	  strcpy(cosmo->status_message,
		 "ccl_cls.c: ccl_cl_tracer(): error initializing spline for ba(z)\n");
	  return NULL;
	}
      }
    }
  }
  else if(tracer_type==CL_TRACER_CL) {
    clt->chi_source=ccl_comoving_radial_distance(cosmo,1./(1+z_source),status);
    clt->chimax=clt->chi_source;
    clt->chimin=0;
  }
  else {
    *status=CCL_ERROR_INCONSISTENT;
    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_tracer(): unknown tracer type\n");
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
  if((clt->tracer_type==CL_TRACER_NC) || (clt->tracer_type==CL_TRACER_WL))
    ccl_spline_free(clt->spl_nz);

  if(clt->tracer_type==CL_TRACER_NC) {
    ccl_spline_free(clt->spl_bz);
    if(clt->has_magnification) {
      ccl_spline_free(clt->spl_sz);
      ccl_spline_free(clt->spl_wM);
    }
  }
  else if(clt->tracer_type==CL_TRACER_WL) {
    ccl_spline_free(clt->spl_wL);
    if(clt->has_intrinsic_alignment) {
      ccl_spline_free(clt->spl_ba);
      ccl_spline_free(clt->spl_rf);
    }
  }
  if(clt->computed_transfer) {
    int il;
    free(clt->n_k);
    for(il=0;il<clt->n_ls;il++)
      ccl_spline_free(clt->spl_transfer[il]);
    free(clt->spl_transfer);
  }
  free(clt);
}

CCL_ClTracer *ccl_cl_tracer_cmblens(ccl_cosmology *cosmo,double z_source,int *status)
{
  return ccl_cl_tracer(cosmo,CL_TRACER_CL,
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
  return ccl_cl_tracer(cosmo,CL_TRACER_NC,has_rsd,has_magnification,0,
			   nz_n,z_n,n,nz_b,z_b,b,nz_s,z_s,s,
			   -1,NULL,NULL,-1,NULL,NULL,0, status);
}

CCL_ClTracer *ccl_cl_tracer_number_counts_simple(ccl_cosmology *cosmo,
						     int nz_n,double *z_n,double *n,
						     int nz_b,double *z_b,double *b, int * status)
{
  return ccl_cl_tracer(cosmo,CL_TRACER_NC,0,0,0,
			   nz_n,z_n,n,nz_b,z_b,b,-1,NULL,NULL,
			   -1,NULL,NULL,-1,NULL,NULL,0, status);
}

CCL_ClTracer *ccl_cl_tracer_lensing(ccl_cosmology *cosmo,
					int has_alignment,
					int nz_n,double *z_n,double *n,
					int nz_ba,double *z_ba,double *ba,
					int nz_rf,double *z_rf,double *rf, int * status)
{
  return ccl_cl_tracer(cosmo,CL_TRACER_WL,0,0,has_alignment,
			   nz_n,z_n,n,-1,NULL,NULL,-1,NULL,NULL,
			   nz_ba,z_ba,ba,nz_rf,z_rf,rf,0, status);
}

CCL_ClTracer *ccl_cl_tracer_lensing_simple(ccl_cosmology *cosmo,
					       int nz_n,double *z_n,double *n, int * status)
{
  return ccl_cl_tracer(cosmo,CL_TRACER_WL,0,0,0,
			   nz_n,z_n,n,-1,NULL,NULL,-1,NULL,NULL,
			   -1,NULL,NULL,-1,NULL,NULL,0, status);
}

static void limits_bessel(double l,double thr,double *xmin,double *xmax)
{
  double thrb=thr*0.5635/pow(l+0.53,0.834);
  *xmax=1./thrb;
  if(l<=0)
    *xmin=0;
  else {
    double logxmin=((l+1.)*(log(l+1.)+M_LN2-1.)+0.5*M_LN2+log(thrb))/l;
    *xmin=exp(logxmin);
  }
}

static double j_bessel_limber(int l,double k)
{
  return sqrt(M_PI/(2*l+1.))/k;
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
    double jl=j_bessel_limber(l,k);
    double pk=ccl_nonlin_matter_power(cosmo,k,a,status);
    return clt->prefac_lensing*l*(l+1.)*w*sqrt(pk)*jl/(a*chi*k*k);
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

  if(clt->tracer_type==CL_TRACER_NC)
    transfer_out=transfer_nc(w->l_arr[il],k,cosmo,w,clt,status);
  else if(clt->tracer_type==CL_TRACER_WL)
    transfer_out=transfer_wl(w->l_arr[il],k,cosmo,w,clt,status);
  else if(clt->tracer_type==CL_TRACER_CL)
    transfer_out=transfer_cmblens(w->l_arr[il],k,cosmo,clt,status);
  else
    transfer_out=-1;
  return transfer_out;
}

static double *get_lkarr(ccl_cosmology *cosmo,CCL_ClWorkspace *w,
			 double l,double chimin,double chimax,int *nk,
			 int *status)
{
  int ik;

  //First compute relevant k-range for this ell
  double kmin,kmax,lkmin,lkmax;
  if(l>w->l_limber) {
    kmin=CCL_MAX(ccl_splines->K_MIN_DEFAULT,0.8*(l+0.5)/chimax);
    kmax=CCL_MIN(ccl_splines->K_MAX,1.2*(l+0.5)/chimin);
  }
  else {
    double xmin,xmax;
    limits_bessel(l,CCL_FRAC_RELEVANT,&xmin,&xmax);
    kmin=CCL_MAX(ccl_splines->K_MIN_DEFAULT,xmin/chimax);
    kmax=CCL_MIN(ccl_splines->K_MAX,xmax/chimin);
    //Cap by maximum meaningful argument of the Bessel function
    kmax=CCL_MIN(kmax,2*(w->l_arr[w->n_ls-1]+0.5)/chimin); //Cap by 2 x inverse scale corresponding to l_max
  }
  lkmin=log10(kmin);
  lkmax=log10(kmax);

  //Allocate memory for transfer function
  double *lkarr;
  double lknew=lkmin;
  double k_period=2*M_PI/chimax;
  double dklin=0.45;
  ik=0;
  while(lknew<=lkmax) {
    double kk=pow(10.,lknew);
    double dk1=k_period*dklin;
    double dk2=kk*(pow(10.,w->dlk)-1.);
    double dk=CCL_MIN(dk1,dk2);
    lknew=log10(kk+dk);
    ik++;
  }

  *nk=CCL_MAX(10,ik+1);
  lkarr=(double *)malloc((*nk)*sizeof(double));
  if(lkarr==NULL) {
    return NULL;
  }

  if((*nk)==10) {
    for(ik=0;ik<(*nk);ik++)
      lkarr[ik]=lkmin+(lkmax-lkmin)*ik/((*nk)-1.);
  }
  else {
    ik=0;
    lkarr[ik]=lkmin;
    while(lkarr[ik]<=lkmax) {
      double kk=pow(10.,lkarr[ik]);
      double dk1=k_period*dklin;
      double dk2=kk*(pow(10.,w->dlk)-1.);
      double dk=CCL_MIN(dk1,dk2);
      ik++;
      lkarr[ik]=log10(kk+dk);
    }
  }

  return lkarr;
}

static void compute_transfer(CCL_ClTracer *clt,ccl_cosmology *cosmo,CCL_ClWorkspace *w,int *status)
{
  int il;
  double zmin=CCL_MAX(w->zmin,clt->zmin);
  double chimin=ccl_comoving_radial_distance(cosmo,1./(1+zmin),status);
  double chimax=clt->chimax;

  //Get how many multipoles and allocate info for each of them
  clt->n_ls=w->n_ls;
  clt->n_k=(int *)malloc(clt->n_ls*sizeof(int));
  if(clt->n_k==NULL) {
    *status=CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,"ccl_cls.c: compute_transfer(): memory allocation\n");
    return;
  }
  clt->spl_transfer=(SplPar **)malloc(clt->n_ls*sizeof(SplPar *));
  if(clt->spl_transfer==NULL) {
    free(clt->n_k);
    *status=CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,"ccl_cls.c: compute_transfer(): memory allocation\n");
    return;
  }

  //Loop over multipoles and compute transfer function for each
  for(il=0;il<clt->n_ls;il++) {
    int ik,nk;
    double l=(double)(w->l_arr[il]);
    double *lkarr=get_lkarr(cosmo,w,l,chimin,chimax,&nk,status);
    if(lkarr==NULL) {
      *status=CCL_ERROR_MEMORY;
      strcpy(cosmo->status_message,"ccl_cls.c: compute_transfer(): memory allocation\n");
      break;
    }

    double *tkarr=(double *)malloc(nk*sizeof(double));
    if(tkarr==NULL) {
      free(lkarr);
      *status=CCL_ERROR_MEMORY;
      strcpy(cosmo->status_message,"ccl_cls.c: compute_transfer(): memory allocation\n");
      break;
    }
    clt->n_k[il]=nk;

    //Loop over k and compute transfer function
    for(ik=0;ik<nk;ik++)
      tkarr[ik]=transfer_wrap(il,lkarr[ik],cosmo,w,clt,status);
    if(*status) {
      free(clt->n_k);
      free(lkarr);
      free(tkarr);
      break;
    }

    //Initialize spline for this ell
    clt->spl_transfer[il]=ccl_spline_init(nk,lkarr,tkarr,0,0);
    if(clt->spl_transfer[il]==NULL) {
      free(lkarr);
      free(tkarr);
      *status=CCL_ERROR_MEMORY;
      strcpy(cosmo->status_message,"ccl_cls.c: compute_transfer(): memory allocation\n");
      break;
    }
    free(lkarr);
    free(tkarr);
  }
  if(*status) {
    free(clt->n_k);
    int ill;
    for(ill=0;ill<il;ill++)
      ccl_spline_free(clt->spl_transfer[il]);
    free(clt->spl_transfer);
  }

  clt->computed_transfer=1;
}

static double transfer(int il,double lk,ccl_cosmology *cosmo,
		       CCL_ClWorkspace *w,CCL_ClTracer *clt,int *status)
{
  if(il<w->l_limber) {
    if(!(clt->computed_transfer))
      compute_transfer(clt,cosmo,w,status);

    return ccl_spline_eval(lk,clt->spl_transfer[il]);
  } else {
    return transfer_wrap(il,lk,cosmo,w,clt,status);
  }
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
  d1=transfer(p->il,lk,p->cosmo,p->w,p->clt1,p->status);
  d2=transfer(p->il,lk,p->cosmo,p->w,p->clt2,p->status);

  return pow(10.,3*lk)*d1*d2;
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
        strcpy(cosmo->status_message,"ccl_cls.c: ccl_angular_cl_native(): error integrating over k\n");
    }
    return -1;
  }
  ccl_check_status(cosmo,status);

  return result*M_LN10*2./M_PI;
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
      strcpy(cosmo->status_message,"ccl_cls.c: ccl_angular_cls(); "
	     "requested l beyond range allowed by workspace\n");
      return;
    }
  }

  //Allocate array for power spectrum at interpolation nodes
  double *l_nodes=(double *)malloc(w->n_ls*sizeof(double));
  if(l_nodes==NULL) {
    *status=CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,"ccl_cls.c: ccl_angular_cls(); memory allocation\n");
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
#ifdef HAVE_ANGPOW
    if(do_angpow)
      ccl_angular_cls_angpow(cosmo,w,clt1,clt2,cl_nodes,status);
    ccl_check_status(cosmo,status);
#else
    do_angpow=0;
    method_use=CCL_NONLIMBER_METHOD_NATIVE;
#endif
  }

  //Compute limber nodes
  for(ii=0;ii<w->n_ls;ii++) {
    if((method_use==CCL_NONLIMBER_METHOD_NATIVE) || (w->l_arr[ii]>w->l_limber))
      cl_nodes[ii]=ccl_angular_cl_native(cosmo,w,ii,clt1,clt2,status);
  }

  //Interpolate into ells requested by user
  SplPar *spcl_nodes=ccl_spline_init(w->n_ls,l_nodes,cl_nodes,0,0);
  if(spcl_nodes==NULL) {
    free(cl_nodes);
    *status=CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,"ccl_cls.c: ccl_cl_angular_cls(); memory allocation\n");
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
  if(((func_code==CCL_CLT_NZ) && (clt->tracer_type==CL_TRACER_CL)) || //Lensing has no N(z)
     (((func_code==CCL_CLT_BZ) || (func_code==CCL_CLT_SZ) || (func_code==CCL_CLT_WM)) &&
      (clt->tracer_type!=CL_TRACER_NC)) || //bias and magnification only for clustering
     (((func_code==CCL_CLT_RF) || (func_code==CCL_CLT_BA) || (func_code==CCL_CLT_WL)) &&
      (clt->tracer_type!=CL_TRACER_WL))) //IAs only for weak lensing
    return 1;
  if((((func_code==CCL_CLT_SZ) || (func_code==CCL_CLT_WM)) &&
      (clt->has_magnification==0)) || //Correct combination, but no magnification
     (((func_code==CCL_CLT_RF) || (func_code==CCL_CLT_BA)) &&
      (clt->has_intrinsic_alignment==0))) //Correct combination, but no IAs
    return 1;
  return 0;
}

double ccl_get_tracer_fa(ccl_cosmology *cosmo,CCL_ClTracer *clt,double a,int func_code,int *status)
{
  SplPar *spl;
  double x=1./a-1; //x-variable is redshift by default

  if(check_clt_fa_inconsistency(clt,func_code)) {
    *status=CCL_ERROR_INCONSISTENT;
    sprintf(cosmo->status_message ,"ccl_cls.c: inconsistent combination of tracer and internal function to be evaluated");
    return -1;
  }

  if(func_code==CCL_CLT_NZ)
    spl=clt->spl_nz;
  if(func_code==CCL_CLT_BZ)
    spl=clt->spl_bz;
  if(func_code==CCL_CLT_SZ)
    spl=clt->spl_sz;
  if(func_code==CCL_CLT_RF)
    spl=clt->spl_rf;
  if(func_code==CCL_CLT_BA)
    spl=clt->spl_ba;
  if((func_code==CCL_CLT_WL) || (func_code==CCL_CLT_WM)) {
    x=ccl_comoving_radial_distance(cosmo,a,status);
    if(func_code==CCL_CLT_WL)
      spl=clt->spl_wL;
    if(func_code==CCL_CLT_WM)
      spl=clt->spl_wM;
  }

  return ccl_spline_eval(x,spl);
}

int ccl_get_tracer_fas(ccl_cosmology *cosmo,CCL_ClTracer *clt,int na,double *a,double *fa,
		       int func_code,int *status)
{
  SplPar *spl;

  if(check_clt_fa_inconsistency(clt,func_code)) {
    *status=CCL_ERROR_INCONSISTENT;
    sprintf(cosmo->status_message ,"ccl_cls.c: inconsistent combination of tracer and internal function to be evaluated");
    return -1;
  }

  if(func_code==CCL_CLT_NZ)
    spl=clt->spl_nz;
  if(func_code==CCL_CLT_BZ)
    spl=clt->spl_bz;
  if(func_code==CCL_CLT_SZ)
    spl=clt->spl_sz;
  if(func_code==CCL_CLT_RF)
    spl=clt->spl_rf;
  if(func_code==CCL_CLT_BA)
    spl=clt->spl_ba;
  if(func_code==CCL_CLT_WL)
    spl=clt->spl_wL;
  if(func_code==CCL_CLT_WM)
    spl=clt->spl_wM;

  int compchi=0;
  if((func_code==CCL_CLT_WL) || (func_code==CCL_CLT_WM))
    compchi=1;

  int ia;
  for(ia=0;ia<na;ia++) {
    double x;
    if(compchi)
      x=ccl_comoving_radial_distance(cosmo,a[ia],status);
    else
      x=1./a[ia]-1;
    fa[ia]=ccl_spline_eval(x,spl);
  }

  return 0;
}
