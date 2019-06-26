#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

#include "ccl.h"

ccl_cl_tracer_collection_t *ccl_cl_tracer_collection_t_new(int *status)
{
  ccl_cl_tracer_collection_t *trc=malloc(sizeof(ccl_cl_tracer_collection_t));
  if(trc==NULL)
    *status=CCL_ERROR_MEMORY;

  if(*status==0) {
    trc->n_tracers=0;
    //Currently this is hard-coded.
    //It should be enough for any practical application with minimal memory overhead 
    trc->ts=malloc(100*sizeof(ccl_cl_tracer_t *));
    if(trc->ts==NULL)
      *status=CCL_ERROR_MEMORY;
  }

  return trc;
}

void ccl_cl_tracer_collection_t_free(ccl_cl_tracer_collection_t *trc)
{
  if(trc!=NULL) {
    if(trc->ts!=NULL)
      free(trc->ts);
    free(trc);
  }
}

void ccl_add_cl_tracer_to_collection(ccl_cl_tracer_collection_t *trc,ccl_cl_tracer_t *tr)
{
  trc->ts[trc->n_tracers]=tr;
  trc->n_tracers++;
}

  
//Takes an array of z-dependent numbers and the corresponding z values
//and returns an array of a values and the corresponding a-dependent values.
//The order of the original arrays is assumed to be ascending in z, and
//the order of the returned arrays is swapped (so it has ascending a).
static void from_z_to_a(ccl_cosmology *cosmo,int nz,double *z_arr,double *fz_arr,
			double **a_arr,double **fa_arr,int *status)
{
  *a_arr=malloc(nz*sizeof(double));
  *fa_arr=malloc(nz*sizeof(double));
  if((a_arr==NULL) || (fa_arr==NULL)) {
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: from_z_to_a(): memory allocation error\n");
  }

  if(*status==0) {
    int ia;
    //Populate array of scale factors in reverse order
    for(ia=0;ia<nz;ia++) {
      (*a_arr)[ia]=1./(1+z_arr[nz-1-ia]);
      (*fa_arr)[ia]=fz_arr[nz-1-ia];
    }
  }
}

static double nz_integrand(double z,void *pars)
{
  ccl_f1d_t *nz_f=(ccl_f1d_t *)pars;

  return ccl_f1d_t_eval(nz_f,z);
}

static double get_nz_norm(ccl_cosmology *cosmo,ccl_f1d_t *nz_f,double z0,double zf,
			  int *status)
{
  double nz_norm=-1,nz_enorm;

  //Get N(z) norm
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(cosmo->gsl_params.N_ITERATION);
  F.function=&nz_integrand;
  F.params=nz_f;
  int gslstatus=gsl_integration_qag(&F,z0,zf,0,
				    cosmo->gsl_params.INTEGRATION_EPSREL, cosmo->gsl_params.N_ITERATION,
				    cosmo->gsl_params.INTEGRATION_GAUSS_KRONROD_POINTS,
				    w, &nz_norm, &nz_enorm);
  gsl_integration_workspace_free(w);
  if(gslstatus!=GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_tracers.c: get_nz_norm():");
    *status=CCL_ERROR_INTEG;
    ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: get_nz_norm(): integration error when normalizing N(z)\n");
  }

  return nz_norm;
}

static void from_z_to_chi(ccl_cosmology *cosmo,int nz,double *z_arr,
			  double *chi_arr,int *status)
{
  int ichi;
  for(ichi=0;ichi<nz;ichi++)
    chi_arr[ichi]=ccl_comoving_radial_distance(cosmo,1./(1+z_arr[ichi]),status);
}

void ccl_get_number_counts_kernel(ccl_cosmology *cosmo,
				  int nz,double *z_arr,double *nz_arr,
				  int normalize_nz,
				  double *pchi_arr,int *status)
{
  //Returns dn/dchi normalized to unit area from an unnormalized dn/dz.
  //Prepare N(z) spline
  ccl_f1d_t *nz_f=ccl_f1d_t_new(nz,z_arr,nz_arr,0,0);
  if(nz_f==NULL) {
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: ccl_get_number_counts_kernel: error initializing spline\n");
  }

  //Get N(z) normalization
  double i_nz_norm=-1;
  if(*status==0) {
    if(normalize_nz)
      i_nz_norm=1./get_nz_norm(cosmo,nz_f,z_arr[0],z_arr[nz-1],status);
    else
      i_nz_norm=1;
  }

  if(*status==0) {
    //Populate arrays
    int ichi;
    for(ichi=0;ichi<nz;ichi++) {
      double a=1./(1+z_arr[ichi]);
      double h=cosmo->params.h*ccl_h_over_h0(cosmo,a,status)/ccl_constants.CLIGHT_HMPC;
      pchi_arr[ichi]=h*nz_arr[ichi]*i_nz_norm; //H(z) * dN/dz * 1/Ngal
    }
  }

  ccl_f1d_t_free(nz_f);
}


static double get_lensing_prefactor(ccl_cosmology *cosmo,int *status)
{
  double hub=cosmo->params.h*ccl_h_over_h0(cosmo,1.,status)/ccl_constants.CLIGHT_HMPC;
  return 1.5*hub*hub*cosmo->params.Omega_m;
}

typedef struct {
  ccl_cosmology *cosmo;
  double z_max;
  double z_end;
  double chi_end;
  double i_nz_norm;
  ccl_f1d_t *nz_f;
  ccl_f1d_t *sz_f;
  int *status;
} integ_lensing_pars;

static double lensing_kernel_integrand(double z,void *pars)
{
  integ_lensing_pars *p=(integ_lensing_pars *)pars;
  double pz=ccl_f1d_t_eval(p->nz_f,z);
  double qz;
  if(p->sz_f==NULL) //No magnification factor
    qz=1;
  else //With magnification factor
    qz=(1-2.5*ccl_f1d_t_eval(p->sz_f,z));

  if(z==0)
    return pz*qz;
  else {
    double chi=ccl_comoving_radial_distance(p->cosmo,1./(1+z),p->status);
    return pz*qz*ccl_sinn(p->cosmo,chi-p->chi_end,p->status)/ccl_sinn(p->cosmo,chi,p->status);
  }
}

static double lensing_kernel_integrate(ccl_cosmology *cosmo,integ_lensing_pars *pars)
{
  // Returns
  // Integral[ p(z') * (1-5s(z')/2) * chi_end * (chi'-chi_end)/chi' , {z',z_end,z_max} ]
  int gslstatus=0;
  double result,eresult;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(cosmo->gsl_params.N_ITERATION);
  F.function=&lensing_kernel_integrand;
  F.params=pars;
  gslstatus=gsl_integration_qag(&F, pars->z_end, pars->z_max, 0,
				cosmo->gsl_params.INTEGRATION_EPSREL, cosmo->gsl_params.N_ITERATION,
				cosmo->gsl_params.INTEGRATION_GAUSS_KRONROD_POINTS,
				w, &result, &eresult);
  gsl_integration_workspace_free(w);
  if((gslstatus!=GSL_SUCCESS) || (*(pars->status))) {
    ccl_raise_gsl_warning(gslstatus, "ccl_tracers.c: lensing_kernel_integrate():");
    return -1;
  }

  return result*pars->i_nz_norm*pars->chi_end;
}

int ccl_get_nchi_lensing_kernel(int nz,double *z_arr,int *status)
{
  int nchi;
  double dz=-1,z_max=-1;
  dz=(z_arr[nz-1]-z_arr[0])/(nz-1);

  return (int)(z_arr[nz-1]/dz+0.5);
}

void ccl_get_chis_lensing_kernel(ccl_cosmology *cosmo,
				 int nchi,double z_max,
				 double *chis,int *status)
{
  int ichi;
  double dz=z_max/nchi;
  for(ichi=0;ichi<nchi;ichi++) {
    double z=dz*ichi+1E-15;
    double a=1./(1+z);
    chis[ichi]=ccl_comoving_radial_distance(cosmo,a,status);
  }
}

void ccl_get_lensing_mag_kernel(ccl_cosmology *cosmo,
				int nz,double *z_arr,double *nz_arr,
				int normalize_nz,double z_max,
				int nz_s,double *zs_arr,double *sz_arr,
				int nchi,double *chi_arr,double *wL_arr,int *status)
{
  //Prepare N(z) spline
  ccl_f1d_t *nz_f=ccl_f1d_t_new(nz,z_arr,nz_arr,0,0);
  if(nz_f==NULL) {
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: get_lensing_mag_kernel: error initializing spline\n");
  }

  //Get N(z) normalization
  double i_nz_norm=-1;
  if(*status==0) {
    if(normalize_nz)
      i_nz_norm=1./get_nz_norm(cosmo,nz_f,z_arr[0],z_arr[nz-1],status);
    else
      i_nz_norm=1.;
  }

  //Prepare magnification bias spline if needed
  ccl_f1d_t *sz_f=NULL;
  if(*status==0) {
    if((nz_s>0) && (zs_arr!=NULL) && (sz_arr!=NULL)) {
      sz_f=ccl_f1d_t_new(nz_s,zs_arr,sz_arr,sz_arr[0],sz_arr[nz_s-1]);
      if(sz_f==NULL) {
	*status=CCL_ERROR_SPLINE;
	ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: get_lensing_mag_kernel: error initializing spline\n");
      }
    }
  }

  integ_lensing_pars *ipar=NULL;
  if(*status==0) {
    ipar=malloc(sizeof(integ_lensing_pars));
    if(ipar==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: get_lensing_mag_kernel(): memory allocation error\n");
    }
  }

  if(*status==0) {
    ipar->cosmo=cosmo;
    ipar->z_max=z_max;
    ipar->i_nz_norm=i_nz_norm;
    ipar->sz_f=sz_f;
    ipar->nz_f=nz_f;

    //Populate arrays
    int ichi;
    double lens_prefac=get_lensing_prefactor(cosmo,status);
    for(ichi=0;ichi<nchi;ichi++) {
      double chi=chi_arr[ichi];
      double a=ccl_scale_factor_of_chi(cosmo,chi,status);
      double z=1./a-1;
      ipar->status=status;
      ipar->z_end=z;
      ipar->chi_end=chi;
      wL_arr[ichi]=lensing_kernel_integrate(cosmo,ipar)*(1+z)*lens_prefac; // divide by scale factor
    }
  }

  ccl_f1d_t_free(nz_f);
  ccl_f1d_t_free(sz_f);
}

void ccl_get_kappa_kernel(ccl_cosmology *cosmo,
			  double chi_source,
			  int nchi,double *chi_arr,
			  double *wchi,int *status)
{
  double lens_prefac=get_lensing_prefactor(cosmo,status)/ccl_sinn(cosmo,chi_source,status);

  int ichi;
  for(ichi=0;ichi<nchi;ichi++) {
    double chi=chi_arr[ichi];
    double a=ccl_scale_factor_of_chi(cosmo,chi,status);
    // 3H0^2Om/2 * chi * (chi_s - chi) / chi_s / a
    wchi[ichi]=lens_prefac*(ccl_sinn(cosmo,chi_source-chi,status))*chi/a;
  }
}

ccl_cl_tracer_t *ccl_cl_tracer_t_new(ccl_cosmology *cosmo,
				     int der_bessel,
				     int der_angles,
				     int n_w,double *chi_w,double *w_w,
				     int na_ka,double *a_ka,
				     int nk_ka,double *lk_ka,
				     double *fka_arr,
				     double *fk_arr,
				     double *fa_arr,
				     int is_fka_log,
				     int is_factorizable,
				     int extrap_order_lok,
				     int extrap_order_hik,
				     int *status)
{
  ccl_cl_tracer_t *tr=NULL;

  if((der_angles<0) || (der_angles>2)) {
    *status=CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: ccl_cl_tracer_new: "
				     "der_angles must be between 0 and 2\n");
  }
  if((der_bessel<-1) || (der_bessel>2)) {
    *status=CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: ccl_cl_tracer_new: "
				     "der_bessel must be between -1 and 2\n");
  }

  if(*status==0) {
    tr=malloc(sizeof(ccl_cl_tracer_t));
    if(tr==NULL)
      *status=CCL_ERROR_MEMORY;
  }

  if(*status==0) {
    tr->der_angles=der_angles;
    tr->der_bessel=der_bessel;
    tr->kernel=NULL; //Initialize these to NULL
    tr->transfer=NULL; //Initialize these to NULL
    tr->chi_min=0;
    tr->chi_max=1E15;
  }
  
  if(*status==0) {
    //Initialize radial kernel
    if((n_w>0) && (chi_w!=NULL) && (w_w!=NULL)) {
      tr->kernel=ccl_f1d_t_new(n_w,chi_w,w_w,0,0);
      if(tr->kernel==NULL) //CHECK IF THIS IS EXPECTED
	*status=CCL_ERROR_MEMORY;
    }
  }

  //Find kernel edges
  if(*status==0) {
    //If no radial kernel, set limits to zero and maximum distance
    if(tr->kernel==NULL) {
      tr->chi_min=0;
      tr->chi_max=ccl_comoving_radial_distance(cosmo,cosmo->spline_params.A_SPLINE_MIN,status);
    }
    else {
      int ichi;
      double w_max=w_w[0];

      //Find maximum of radial kernel
      for(ichi=0;ichi<n_w;ichi++) {
	if(w_w[ichi]>=w_max)
	  w_max=w_w[ichi];
      }

      //Multiply by fraction
      w_max*=CCL_FRAC_RELEVANT;

      // Initialize as the original edges in case we don't find an interval
      tr->chi_min=chi_w[0];
      tr->chi_max=chi_w[n_w-1];

      //Find minimum
      for(ichi=0;ichi<n_w;ichi++) {
	if(w_w[ichi]>=w_max) {
	  tr->chi_min=chi_w[ichi];
	  break;
	}
      }

      //Find maximum
      for(ichi=n_w-1;ichi>=0;ichi--) {
	if(w_w[ichi]>=w_max) {
	  tr->chi_max=chi_w[ichi];
	  break;
	}
      }
    }
  }

  if(*status==0) {
    if((fka_arr!=NULL) || (fk_arr!=NULL) || (fa_arr!=NULL)) {
      tr->transfer=ccl_f2d_t_new(na_ka,a_ka, //na, a_arr
				 nk_ka,lk_ka, //nk, lk_arr
				 fka_arr, //fka_arr
				 fk_arr, //fk_arr
				 fa_arr, //fa_arr
				 is_factorizable, //is factorizable
				 extrap_order_lok, //extrap_order_lok
				 extrap_order_hik, //extrap_order_hik
				 ccl_f2d_constantgrowth, //extrap_linear_growth
				 is_fka_log, //is_fka_log
				 NULL, //growth (function)
				 1, //growth_factor_0 -> will assume constant transfer function
				 0, //growth_exponent
				 ccl_f2d_3, //interp_type
				 status);
      if(tr->transfer==NULL) //CHECK IF THIS IS EXPECTED
	*status=CCL_ERROR_MEMORY;
    }
  }
  return tr;
}

void ccl_cl_tracer_t_free(ccl_cl_tracer_t *tr)
{
  if(tr!=NULL) {
    if(tr->transfer!=NULL)
      ccl_f2d_t_free(tr->transfer);
    if(tr->kernel!=NULL)
      ccl_f1d_t_free(tr->kernel);
    free(tr);
  }
}

double ccl_cl_tracer_t_get_f_ell(ccl_cl_tracer_t *tr,double ell,int *status)
{
  if(tr!=NULL) {
    if(tr->der_angles==1)
      return ell*(ell+1.);
    else if(tr->der_angles==2) {
      if(ell>1)
	return sqrt((ell+2)*(ell+1)*ell*(ell-1));
      else
	return 0;
    }
    else
      return 1;
  }
  else
    return 1;
}

double ccl_cl_tracer_t_get_kernel(ccl_cl_tracer_t *tr,double chi,int *status)
{
  if(tr!=NULL) {
    if(tr->kernel!=NULL)
      ccl_f1d_t_eval(tr->kernel,chi);
    else
      return 1;
  }
  else
    return 1;
}

double ccl_cl_tracer_t_get_transfer(ccl_cl_tracer_t *tr,double lk,double a,int *status)
{
  if(tr!=NULL) {
    if(tr->transfer!=NULL)
      return ccl_f2d_t_eval(tr->transfer,lk,a,NULL,status);
    else
      return 1;
  }
  else
    return 1;
}
