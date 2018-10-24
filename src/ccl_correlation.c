#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_legendre.h>

#include "fftlog.h"

#include "ccl.h"
#include "ccl_params.h"

/*--------ROUTINE: taper_cl ------
TASK:n Apply cosine tapering to Cls to reduce aliasing
INPUT: number of ell bins for Cl, ell vector, C_ell vector, limits for tapering
       e.g., ell_limits=[low_ell_limit_lower,low_ell_limit_upper,high_ell_limit_lower,high_ell_limit_upper]
*/
static int taper_cl(int n_ell,double *ell,double *cl, double *ell_limits)
{

  for(int i=0;i<n_ell;i++) {
    if(ell[i]<ell_limits[0] || ell[i]>ell_limits[3]) {
      cl[i]=0;//ell outside desirable range
      continue;
    }
    if(ell[i]>=ell_limits[1] && ell[i]<=ell_limits[2])
      continue;//ell within good ell range
    
    if(ell[i]<ell_limits[1])//tapering low ell
      cl[i]*=cos((ell[i]-ell_limits[1])/(ell_limits[1]-ell_limits[0])*M_PI/2.);
    
    if(ell[i]>ell_limits[2])//tapering high ell
      cl[i]*=cos((ell[i]-ell_limits[2])/(ell_limits[3]-ell_limits[2])*M_PI/2.);
  }

  return 0;
}

/*--------ROUTINE: ccl_tracer_corr_fftlog ------
TASK: For a given tracer, get the correlation function
      Following function takes a function to calculate angular cl as well.
      By default above function will call it using ccl_angular_cl
INPUT: type of tracer, number of theta values to evaluate = NL, theta vector
 */
static void ccl_tracer_corr_fftlog(ccl_cosmology *cosmo,
				   int n_ell,double *ell,double *cls,
				   int n_theta,double *theta,double *wtheta,
				   int corr_type,int do_taper_cl,double *taper_cl_limits,
				   int *status)
{
  int i;
  double *l_arr,*cl_arr,*th_arr,*wth_arr;

  l_arr=ccl_log_spacing(ccl_splines->ELL_MIN_CORR,ccl_splines->ELL_MAX_CORR,ccl_splines->N_ELL_CORR);
  if(l_arr==NULL) {
    *status=CCL_ERROR_LINSPACE;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_fftlog ran out of memory\n");
    return;
  }
  cl_arr=malloc(ccl_splines->N_ELL_CORR*sizeof(double));
  if(cl_arr==NULL) {
    free(l_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_fftlog ran out of memory\n");
    return;
  }

  //Interpolate input Cl into array needed for FFTLog
  SplPar *cl_spl=ccl_spline_init(n_ell,ell,cls,cls[0],0);
  if(cl_spl==NULL) {
    free(l_arr);
    free(cl_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_fftlog ran out of memory\n");
    return;
  }

  double cl_tilt,l_edge,cl_edge;
  l_edge=ell[n_ell-1];
  if((cls[n_ell-1]*cls[n_ell-2]<0) || (cls[n_ell-2]==0)) {
    cl_tilt=0;
    cl_edge=0;
  }
  else {
    cl_tilt=log(cls[n_ell-1]/cls[n_ell-2])/log(ell[n_ell-1]/ell[n_ell-2]);
    cl_edge=cls[n_ell-1];
  }
  for(i=0;i<ccl_splines->N_ELL_CORR;i++) {
    if(l_arr[i]>=l_edge)
      cl_arr[i]=cl_edge*pow(l_arr[i]/l_edge,cl_tilt);
    else
      cl_arr[i]=ccl_spline_eval(l_arr[i],cl_spl);
  }
  ccl_spline_free(cl_spl);

  if (do_taper_cl)
    taper_cl(ccl_splines->N_ELL_CORR,l_arr,cl_arr,taper_cl_limits);

  th_arr=malloc(sizeof(double)*ccl_splines->N_ELL_CORR);
  if(th_arr==NULL) {
    free(l_arr);
    free(cl_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_fftlog ran out of memory\n");
    return;
  }
  wth_arr=(double *)malloc(sizeof(double)*ccl_splines->N_ELL_CORR);
  if(wth_arr==NULL) {
    free(l_arr); free(cl_arr); free(th_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_fftlog ran out of memory\n");
    return;
  }

  for(i=0;i<ccl_splines->N_ELL_CORR;i++)
    th_arr[i]=0;
  //Although set here to 0, theta is modified by FFTlog to obtain the correlation at ~1/l

  int i_bessel=0;
  if(corr_type==CCL_CORR_GG) i_bessel=0;
  if(corr_type==CCL_CORR_GL) i_bessel=2;
  if(corr_type==CCL_CORR_LP) i_bessel=0;
  if(corr_type==CCL_CORR_LM) i_bessel=4;
  fftlog_ComputeXi2D(i_bessel,ccl_splines->N_ELL_CORR,l_arr,cl_arr,th_arr,wth_arr);

  // Interpolate to output values of theta
  SplPar *wth_spl=ccl_spline_init(ccl_splines->N_ELL_CORR,th_arr,wth_arr,wth_arr[0],0);
  for(i=0;i<n_theta;i++)
    wtheta[i]=ccl_spline_eval(theta[i]*M_PI/180.,wth_spl);
  ccl_spline_free(wth_spl);

  free(l_arr); free(cl_arr);
  free(th_arr); free(wth_arr);

  return;
}

typedef struct {
  int nell;
  double ell0;
  double ellf;
  double cl0;
  double clf;
  int extrapol_0;
  int extrapol_f;
  double tilt0;
  double tiltf;
  SplPar *cl_spl;
  int i_bessel;
  double th;
} corr_int_par;

static double corr_bessel_integrand(double l,void *params)
{
  double cl,jbes;
  corr_int_par *p=(corr_int_par *)params;
  double x=l*p->th;

  if(l<p->ell0) {
    if(p->extrapol_0)
      cl=p->cl0*pow(l/p->ell0,p->tilt0);
    else
      cl=0;
  }
  else if(l>p->ellf) {
    if(p->extrapol_f)
      cl=p->clf*pow(l/p->ellf,p->tiltf);
    else
      cl=0;
  }
  else
    cl=ccl_spline_eval(l,p->cl_spl);

  jbes=gsl_sf_bessel_Jn(p->i_bessel,x);

  return l*jbes*cl;
}

static void ccl_tracer_corr_bessel(ccl_cosmology *cosmo,
				   int n_ell,double *ell,double *cls,
				   int n_theta,double *theta,double *wtheta,
				   int corr_type,int *status)
{
  corr_int_par *cp=malloc(sizeof(corr_int_par));
  if(cp==NULL) {
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_bessel ran out of memory\n");
    return;
  }

  cp->nell=n_ell;
  cp->ell0=ell[0];
  cp->ellf=ell[n_ell-1];
  cp->cl0=cls[0];
  cp->clf=cls[n_ell-1];
  cp->cl_spl=ccl_spline_init(n_ell,ell,cls,cls[0],0);
  if(cp->cl_spl==NULL) {
    free(cp);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_bessel ran out of memory\n");
    return;
  }
  switch(corr_type) {
  case CCL_CORR_GG :
    cp->i_bessel=0;
    break;
  case CCL_CORR_GL :
    cp->i_bessel=2;
    break;
  case CCL_CORR_LP :
    cp->i_bessel=0;
    break;
  case CCL_CORR_LM :
    cp->i_bessel=4;
    break;
  }

  if(cls[0]*cls[1]<=0)
    cp->extrapol_0=0;
  else {
    cp->extrapol_0=1;
    cp->tilt0=log10(cls[1]/cls[0])/log10(ell[1]/ell[0]);
  }

  if(cls[n_ell-2]*cls[n_ell-1]<=0)
    cp->extrapol_f=0;
  else {
    cp->extrapol_f=1;
    cp->tiltf=log10(cls[n_ell-1]/cls[n_ell-2])/log10(ell[n_ell-1]/ell[n_ell-2]);
  }

  int ith, gslstatus;
  double result,eresult;
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(ccl_gsl->N_ITERATION);
  for(ith=0;ith<n_theta;ith++) {
    cp->th=theta[ith]*M_PI/180;
    F.function=&corr_bessel_integrand;
    F.params=cp;
    gslstatus = gsl_integration_qag(&F, 0, ccl_splines->ELL_MAX_CORR, 0,
                                    ccl_gsl->INTEGRATION_EPSREL, ccl_gsl->N_ITERATION,
                                    ccl_gsl->INTEGRATION_GAUSS_KRONROD_POINTS,
                                    w, &result, &eresult);
    if(gslstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_correlation.c: ccl_tracer_corr_bessel():");
      *status |= gslstatus;
    }
    wtheta[ith]=result/(2*M_PI);
  }
  gsl_integration_workspace_free(w);
  ccl_spline_free(cp->cl_spl);
  free(cp);
}


/*--------ROUTINE: ccl_compute_legendre_polynomial ------
TASK: Compute input factor for ccl_tracer_corr_legendre
INPUT: tracer 1, tracer 2, i_bessel, theta array, n_theta, L_max, output Pl_theta
 */
static void ccl_compute_legendre_polynomial(int corr_type,double theta,int ell_max,double *Pl_theta)
{
  int i,j;
  double k=0;
  double cth=cos(theta*M_PI/180);
  
  //Initialize Pl_theta
  for (j=0;j<=ell_max;j++)
      Pl_theta[j]=0.;

  if(corr_type==CCL_CORR_GG) {
    gsl_sf_legendre_Pl_array(ell_max,cth,Pl_theta);
    for (j=0;j<=ell_max;j++)
      Pl_theta[j]*=(2*j+1);
  }
  else if(corr_type==CCL_CORR_GL) {
    for (j=2;j<=ell_max;j++) {//https://arxiv.org/pdf/1007.4809.pdf
      Pl_theta[j]=gsl_sf_legendre_Plm(j,2,cth);
      Pl_theta[j]*=(2*j+1.)/((j+0.)*(j+1.));
    }
  }
}

/*--------ROUTINE: ccl_tracer_corr_legendre ------
TASK: Compute correlation function via Legendre polynomials
INPUT: cosmology, number of theta bins, theta array, tracer 1, tracer 2, i_bessel, boolean
       for tapering, vector of tapering limits, correlation vector, angular_cl function.
 */
static void ccl_tracer_corr_legendre(ccl_cosmology *cosmo,
				     int n_ell,double *ell,double *cls,
				     int n_theta,double *theta,double *wtheta,
				     int corr_type,int do_taper_cl,double *taper_cl_limits,
				     int *status)
{
  int i;
  double *l_arr,*cl_arr,*Pl_theta;
  SplPar *cl_spl;

  if(corr_type==CCL_CORR_LM || corr_type==CCL_CORR_LP){
    *status=CCL_ERROR_NOT_IMPLEMENTED;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: CCL does not support full-sky xi+- calcuations.\nhttps://arxiv.org/abs/1702.05301 indicates flat-sky to be sufficient.\n");
  }
  
  if(*status==0) {
    l_arr=malloc(((int)(ccl_splines->ELL_MAX_CORR)+1)*sizeof(double));
    if(l_arr==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_legendre ran out of memory\n");
    }
  }
  
  if(*status==0) {
    cl_arr=malloc(((int)(ccl_splines->ELL_MAX_CORR)+1)*sizeof(double));
    if(cl_arr==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_legendre ran out of memory\n");
    }
  }

  if(*status==0) {
    //Interpolate input Cl into 
    cl_spl=ccl_spline_init(n_ell,ell,cls,cls[0],0);
    if(cl_spl==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_legendre ran out of memory\n");
    }
  }

  if(*status==0) {
    double cl_tilt,l_edge,cl_edge;
    l_edge=ell[n_ell-1];
    if((cls[n_ell-1]*cls[n_ell-2]<0) || (cls[n_ell-2]==0)) {
      cl_tilt=0;
      cl_edge=0;
    }
    else {
      cl_tilt=log(cls[n_ell-1]/cls[n_ell-2])/log(ell[n_ell-1]/ell[n_ell-2]);
      cl_edge=cls[n_ell-1];
    }
    for(i=0;i<=(int)(ccl_splines->ELL_MAX_CORR);i++) {
      double l=(double)i;
      l_arr[i]=l;
      if(l>=l_edge)
	cl_arr[i]=cl_edge*pow(l/l_edge,cl_tilt);
      else
	cl_arr[i]=ccl_spline_eval(l,cl_spl);
    }
    ccl_spline_free(cl_spl);

    if (do_taper_cl)
      *status=taper_cl((int)(ccl_splines->ELL_MAX_CORR)+1,l_arr,cl_arr,taper_cl_limits);
  }

  if(*status==0) {
    Pl_theta=malloc(sizeof(double)*((int)(ccl_splines->ELL_MAX_CORR)+1));
    if(Pl_theta==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_legendre ran out of memory\n");
    }
  }
  
  if(*status==0) {
    for (int i=0;i<n_theta;i++) {
      wtheta[i]=0;
      ccl_compute_legendre_polynomial(corr_type,theta[i],(int)(ccl_splines->ELL_MAX_CORR),Pl_theta);
      for(int i_L=1;i_L<(int)(ccl_splines->ELL_MAX_CORR);i_L+=1)
	wtheta[i]+=cl_arr[i_L]*Pl_theta[i_L];
      wtheta[i]/=(M_PI*4);
    }
  }

  free(Pl_theta);
  free(l_arr);
  free(cl_arr);
}

/*--------ROUTINE: ccl_tracer_corr ------
TASK: For a given tracer, get the correlation function. Do so by running
      ccl_angular_cls. If you already have Cls calculated, go to the next
      function to pass them directly.
INPUT: cosmology, number of theta values to evaluate = NL, theta vector,
       tracer 1, tracer 2, i_bessel, key for tapering, limits of tapering
       correlation function.
 */
void ccl_correlation(ccl_cosmology *cosmo,
		     int n_ell,double *ell,double *cls,
		     int n_theta,double *theta,double *wtheta,
		     int corr_type,int do_taper_cl,double *taper_cl_limits,int flag_method,
		     int *status)
{
  switch(flag_method) {
  case CCL_CORR_FFTLOG :
    ccl_tracer_corr_fftlog(cosmo,n_ell,ell,cls,n_theta,theta,wtheta,corr_type,
			   do_taper_cl,taper_cl_limits,status);
    break;
  case CCL_CORR_LGNDRE :
    ccl_tracer_corr_legendre(cosmo,n_ell,ell,cls,n_theta,theta,wtheta,corr_type,
			     do_taper_cl,taper_cl_limits,status);
    break;
  case CCL_CORR_BESSEL :
    ccl_tracer_corr_bessel(cosmo,n_ell,ell,cls,n_theta,theta,wtheta,corr_type,status);
    break;
  default :
    *status=CCL_ERROR_INCONSISTENT;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_correlation. Unknown algorithm\n");
  }

  ccl_check_status(cosmo,status);
}

/*--------ROUTINE: ccl_correlation_3d ------
TASK: Calculate the 3d-correlation function. Do so by using FFTLog. 

INPUT: cosmology, scale factor a,
       number of r values, r values, 
       key for tapering, limits of tapering

Correlation function result will be in array xi
 */

void ccl_correlation_3d(ccl_cosmology *cosmo, double a,
			int n_r,double *r,double *xi,
			int do_taper_pk,double *taper_pk_limits,
			int *status)
{
  int i,N_ARR;
  double *k_arr,*pk_arr,*r_arr,*xi_arr;

  //number of data points for k and pk array
  N_ARR=(int)(ccl_splines->N_K_3DCOR*log10(ccl_splines->K_MAX/ccl_splines->K_MIN));  

  k_arr=ccl_log_spacing(ccl_splines->K_MIN,ccl_splines->K_MAX,N_ARR);
  if(k_arr==NULL) {
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_correlation_3d ran out of memory\n");
    return;
  }

  pk_arr=malloc(N_ARR*sizeof(double));
  if(pk_arr==NULL) {
    free(k_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_correlation_3d ran out of memory\n");
    return;
  }  

  for (i=0; i<N_ARR; i++)
    pk_arr[i] = ccl_nonlin_matter_power(cosmo, k_arr[i], a, status);

  if (do_taper_pk)
    taper_cl(N_ARR,k_arr,pk_arr,taper_pk_limits);

  r_arr=malloc(sizeof(double)*N_ARR);
  if(r_arr==NULL) {
    free(k_arr);
    free(pk_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_correlation_3d ran out of memory\n");
    return;
  }
  xi_arr=malloc(sizeof(double)*N_ARR);
  if(xi_arr==NULL) {
    free(k_arr); free(pk_arr); free(r_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_correlation_3d ran out of memory\n");
    return;
  }

  for(i=0;i<N_ARR;i++)
    r_arr[i]=0;

  pk2xi(N_ARR,k_arr,pk_arr,r_arr,xi_arr);

  // Interpolate to output values of r
  SplPar *xi_spl=ccl_spline_init(N_ARR,r_arr,xi_arr,xi_arr[0],0);
  for(i=0;i<n_r;i++)
    xi[i]=ccl_spline_eval(r[i],xi_spl);
  ccl_spline_free(xi_spl);

  free(k_arr); free(pk_arr);
  free(r_arr); free(xi_arr);

  ccl_check_status(cosmo,status);

  return;
}
