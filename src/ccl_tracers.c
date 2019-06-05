#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>

#include "ccl.h"

/*
static double nz_integral(double z,void *pars)
{
  ccl_f1d_t *nz_f=(ccl_f1d_t *)pars;

  return ccl_f1d_t_eval(nz_f,z);
}

//Takes an array of z-dependent numbers and the corresponding z values
//and returns an array of a values and the corresponding a-dependent values.
//The order of the original arrays is assumed to be ascending in z, and
//the order of the returned arrays is swapped (so it has ascending a).
static void from_z_to_a(int nz,double *z_arr,double *fz_arr,
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
      a_arr[ia]=1./(1+z_arr[nz-1-ia]);
      fa_arr[ia]=fz_arr[nz-1-ia];
    }
  }
}

static void get_shear_kernel(ccl_cosmology *cosmo,int nz,double *z_arr,double *nz_arr,
			     double **chi_arr,double **pchi_arr,int *status)
{
  //TODO
}


static void get_magnification_kernel(ccl_cosmology *cosmo,
				     int nz_n,double *z_arr,double *nz_arr,
				     int nz_s,double *zs_arr,double *sz_arr,
				     double **chi_arr,double **pchi_arr,int *status)
{
  //TODO
}

static void get_number_counts_kernel(ccl_cosmology *cosmo,int nz,double *z_arr,double *nz_arr,
				     double **chi_arr,double **pchi_arr,int *status)
{
  //Returns dn/dchi normalized to unit area from an unnormalized dn/dz.
  double nz_norm,nz_enorm;
  *chi_arr=NULL;
  *pchi_arr=NULL;

  //Interpolate N(z) arrays
  ccl_f1d_t *nz_f=ccl_f1d_t_new(nz,z_arr,nz_arr,0,0);
  if(nz_f==NULL) {
    *status=CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: get_number_counts_kernel: error initializing spline\n");
  }
  
  if(*status==0) {
    //Get N(z) norm
    gsl_integration_workspace *w=gsl_integration_workspace_alloc(cosmo->gsl_params.N_ITERATION);
    F.function=&nz_integral;
    F.params=nz_f;
    int gslstatus=gsl_integration_qag(&F,z_arr[0],z_arr[nz-1], 0,
				      cosmo->gsl_params.INTEGRATION_EPSREL, cosmo->gsl_params.N_ITERATION,
				      cosmo->gsl_params.INTEGRATION_GAUSS_KRONROD_POINTS,
				      w, &nz_norm, &nz_enorm);
    gsl_integration_workspace_free(w);
    if(gslstatus!=GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_tracers.c: clt_init_nz():");
      *status=CCL_ERROR_INTEG;
      ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: get_number_counts_kernel(): integration error when normalizing N(z)\n");
    }
  }
  
  if(*status==0) {
    //Create arrays for chi and dn/dchi
    (*chi_arr)=malloc(nz*sizeof(z_arr));
    (*pchi_arr)=malloc(nz*sizeof(z_arr));
    if(((*chi_arr)==NULL) || ((*pchi_arr)==NULL)) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: get_number_counts_kernel(): memory allocation error\n");
    }
  }

  if(*status==0) {
    //Populate arrays
    int ichi;
    for(ichi=0;ichi<nz;ichi++) {
      double a=1./(1+z_arr[ichi]);
      double h=cosmo->params.h*ccl_h_over_h0(cosmo,a,status)/ccl_constants.CLIGHT_HMPC;
      (*chi_arr)[ichi]=ccl_comoving_radial_distance(cosmo,a,status);
      (*pchi_arr)[ichi]=h*nz_arr[ichi]/nz_norm; //H(z) * dN/dz * 1/Ngal
    }
  }

  ccl_f1d_t_free(nz_f);
}

ccl_cl_tracer_t *ccl_wl_shear_tracer_new(ccl_cosmology *cosmo,
					 int nz_n,double *z_n,double *n,
					 int *status)
{
  //TODO
}

ccl_cl_tracer_t *ccl_wl_ia_tracer_new(ccl_cosmology *cosmo,
				      int nz_n,double *z_n,double *n,
				      int nz_aia,double *z_aia,double *aia,
				      int *status)
{
  ccl_cl_tracer_t *tr=NULL;
  double *chi_arr=NULL,*pchi_arr=NULL;

  //Arrays containing scale factor and alignment amplitude in reverse order
  double *a_aia=NULL,*aia_b=NULL;
  from_z_to_a(nz_aia,z_aia,aia,&a_aia,&aia_b,status);

  //Divide by chi
  if(*status==0) {
    int ia;
    for(ia=0;ia<nz_aia;ia++) {
      double a=a_aia[ia];
      aia_b[ia]/=ccl_comoving_radial_distance(cosmo,a,status);
    }
  }

  if(*status==0) {
    //Populate radial kernel arrays
    get_number_counts_kernel(cosmo,nz_n,z_n,n,&chi_arr,&pchi_arr,status);
  }

  if(*status==0) {
    //Create tracer
    tr=ccl_cl_tracer_t_new(cosmo,
			   0, //der_bessel
			   2, //der_angles
			   nz_n,chi_arr,pchi_arr, //kernel
			   nz_aia,a_aia, //na_ka, a_ka
			   -1,NULL, //nk_ka, lk_ka
			   NULL, //fka_arr
			   NULL, //fk_arr
			   aia_b, //fa_arr
			   1, 1, -2, //is_factorizable, is_k_powerlaw, k_powerlaw_exponent
			   0, 0, //extrap_order_lok, extrap_order_hik
			   status);
    if(tr==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: ccl_nc_dens_tracer_new(): error generating tracer\n");
    }
  }

  //Cleanup
  free(a_aia);
  free(aia_b);
  free(chi_arr);
  free(pchi_arr);

  // Check: division by chi, ell-factor
  return tr;
}

ccl_cl_tracer_t *ccl_nc_rsd_tracer_new(ccl_cosmology *cosmo,
				       int nz_n,double *z_n,double *n,
				       int *status)
{
  //TODO
}

ccl_cl_tracer_t *ccl_nc_mag_tracer_new(ccl_cosmology *cosmo,
				       int nz_n,double *z_n,double *n,
				       int nz_s,double *z_s,double *s,
				       int *status)
{
  //TODO
}

ccl_cl_tracer_t *ccl_nc_dens_tracer_new(ccl_cosmology *cosmo,
					int nz_n,double *z_n,double *n,
					int nz_b,double *z_b,double *b,
					int *status)
{
  ccl_cl_tracer_t *tr=NULL;
  double *chi_arr=NULL,*pchi_arr=NULL;

  //Arrays containing scale factor and bias in reverse order
  double *a_b=NULL,*b_b=NULL;
  from_z_to_a(nz_b,z_b,b,&a_b,&b_b,status);

  if(*status==0) {
    //Populate radial kernel arrays
    get_number_counts_kernel(cosmo,nz_n,z_n,n,&chi_arr,&pchi_arr,status);
  }

  if(*status==0) {
    //Create tracer
    tr=ccl_cl_tracer_t_new(cosmo,
			   0, //der_bessel
			   0, //der_angles
			   nz_n,chi_arr,pchi_arr, //kernel
			   nz_b,a_b, //na_ka, a_ka
			   -1,NULL, //nk_ka, lk_ka
			   NULL, //fka_arr
			   NULL, //fk_arr
			   b_b, //fa_arr
			   1, 1, 0, //is_factorizable, is_k_powerlaw, k_powerlaw_exponent
			   0, 0, //extrap_order_lok, extrap_order_hik
			   status);
    if(tr==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: ccl_nc_dens_tracer_new(): error generating tracer\n");
    }
  }

  //Cleanup
  free(a_b);
  free(b_b);
  free(chi_arr);
  free(pchi_arr);

  return tr;
}
*/

ccl_cl_tracer_t *ccl_cl_tracer_t_new(ccl_cosmology *cosmo,
				     int der_bessel,
				     int der_angles,
				     int n_w,double *chi_w,double *w_w,
				     int na_ka,double *a_ka,
				     int nk_ka,double *lk_ka,
				     double *fka_arr,
				     double *fk_arr,
				     double *fa_arr,
				     int is_factorizable,
				     int is_k_powerlaw,
				     double k_powerlaw_exponent,
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
  if((der_bessel<0) || (der_bessel>2)) {
    *status=CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(cosmo, "ccl_tracers.c: ccl_cl_tracer_new: "
				     "der_bessel must be between 0 and 2\n");
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
  if(*status=0) {
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

  if(*status==0) {
    if((na_ka>0) && (nk_ka>0) &&
       (a_ka!=NULL) && (lk_ka!=NULL) &&
       ((fka_arr!=NULL) ||
	((fk_arr!=NULL) && (fa_arr!=NULL)))) {
      tr->transfer=ccl_f2d_t_new(na_ka,a_ka, //na, a_arr
				 nk_ka,lk_ka, //nk, lk_arr
				 fka_arr, //fka_arr
				 fk_arr, //fk_arr
				 fa_arr, //fa_arr
				 is_factorizable, //is factorizable
				 is_k_powerlaw, //is_k_powerlaw
				 k_powerlaw_exponent, //k_powerlaw_exponent
				 extrap_order_lok, //extrap_order_lok
				 extrap_order_hik, //extrap_order_hik
				 ccl_f2d_constantgrowth, //extrap_linear_growth
				 0, //is_fka_log
				 NULL, //growth (function)
				 1, //growth_factor_0 -> will assume constant transfer function
				 0, //growth_exponent
				 ccl_f2d_3, //interp_type
				 status);
      if(tr->transfer==NULL) //CHECK IF THIS IS EXPECTED
	*status=CCL_ERROR_MEMORY;
    }
  }
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
      ccl_f2d_t_eval(tr->transfer,lk,a,NULL,status);
    else
      return 1;
  }
  else
    return 1;
}

double ccl_cl_tracer_t_get_cl_contribution(ccl_cl_tracer_t *tr,
					   double ell,double chi,double lk,double a,
					   int *status)
{
  double f_chi=ccl_cl_tracer_t_get_kernel(tr,chi,status);
  double f_ka=ccl_cl_tracer_t_get_transfer(tr,lk,a,status);
  double f_ell=ccl_cl_tracer_t_get_f_ell(tr,ell,status);
  double delta_lka=f_ka*f_chi*f_ell;

  if(*status==0)
    return delta_lka;
  else
    return -1;
}     
