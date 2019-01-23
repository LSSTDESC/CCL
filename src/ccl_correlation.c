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

// Global variable to hold the redshift space correlation multipole function spline
SplPar* xir_spline[3]={NULL,NULL,NULL};

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
#define ELL_MIN_FFTLOG 0.01
#define ELL_MAX_FFTLOG 60000
#define N_ELL_FFTLOG 5000
static void ccl_tracer_corr_fftlog(ccl_cosmology *cosmo,
				   int n_ell,double *ell,double *cls,
				   int n_theta,double *theta,double *wtheta,
				   int corr_type,int do_taper_cl,double *taper_cl_limits,
				   int *status)
{
  int i;
  double *l_arr,*cl_arr,*th_arr,*wth_arr;

  l_arr=ccl_log_spacing(ELL_MIN_FFTLOG,ELL_MAX_FFTLOG,N_ELL_FFTLOG);
  if(l_arr==NULL) {
    *status=CCL_ERROR_LINSPACE;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_fftlog ran out of memory\n");
    return;
  }
  cl_arr=malloc(N_ELL_FFTLOG*sizeof(double));
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
  for(i=0;i<N_ELL_FFTLOG;i++) {
    if(l_arr[i]>=l_edge)
      cl_arr[i]=cl_edge*pow(l_arr[i]/l_edge,cl_tilt);
    else
      cl_arr[i]=ccl_spline_eval(l_arr[i],cl_spl);
  }
  ccl_spline_free(cl_spl);

  if (do_taper_cl)
    taper_cl(N_ELL_FFTLOG,l_arr,cl_arr,taper_cl_limits);

  th_arr=malloc(sizeof(double)*N_ELL_FFTLOG);
  if(th_arr==NULL) {
    free(l_arr);
    free(cl_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_fftlog ran out of memory\n");
    return;
  }
  wth_arr=(double *)malloc(sizeof(double)*N_ELL_FFTLOG);
  if(wth_arr==NULL) {
    free(l_arr); free(cl_arr); free(th_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_fftlog ran out of memory\n");
    return;
  }

  for(i=0;i<N_ELL_FFTLOG;i++)
    th_arr[i]=0;
  //Although set here to 0, theta is modified by FFTlog to obtain the correlation at ~1/l

  int i_bessel=0;
  if(corr_type==CCL_CORR_GG) i_bessel=0;
  if(corr_type==CCL_CORR_GL) i_bessel=2;
  if(corr_type==CCL_CORR_LP) i_bessel=0;
  if(corr_type==CCL_CORR_LM) i_bessel=4;
  fftlog_ComputeXi2D(i_bessel,N_ELL_FFTLOG,l_arr,cl_arr,th_arr,wth_arr);

  // Interpolate to output values of theta
  SplPar *wth_spl=ccl_spline_init(N_ELL_FFTLOG,th_arr,wth_arr,wth_arr[0],0);
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

  if(p->i_bessel)
    jbes=gsl_sf_bessel_Jn(p->i_bessel,x);
  else
    jbes=gsl_sf_bessel_J0(x);

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
  if(corr_type==CCL_CORR_GG)
    cp->i_bessel=0;
  else if(corr_type==CCL_CORR_GL)
    cp->i_bessel=2;
  else if(corr_type==CCL_CORR_LP)
    cp->i_bessel=0;
  else if(corr_type==CCL_CORR_LM)
    cp->i_bessel=4;

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
    gslstatus = gsl_integration_qag(&F, 0, ELL_MAX_FFTLOG, 0,
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
static void ccl_compute_legendre_polynomial(int corr_type,int n_theta,double *theta,
					    int ell_max,double **Pl_theta)
{
  int i,j;
  double k=0;

  //Initialize Pl_theta
  for (i=0;i<n_theta;i++) {
    for (j=0;j<ell_max;j++)
      Pl_theta[i][j]=0.;
  }

  if(corr_type==CCL_CORR_GG) {
    for (i=0;i<n_theta;i++) {
      gsl_sf_legendre_Pl_array(ell_max,cos(theta[i]*M_PI/180),Pl_theta[i]);
      for (j=0;j<=ell_max;j++)
	Pl_theta[i][j]*=(2*j+1);
    }
  }
  else if(corr_type==CCL_CORR_GL) {
    for (i=0;i<n_theta;i++) {//https://arxiv.org/pdf/1007.4809.pdf
      for (j=2;j<=ell_max;j++) {
	Pl_theta[i][j]=gsl_sf_legendre_Plm(j,2,cos(theta[i]*M_PI/180));
	Pl_theta[i][j]*=(2*j+1.)/((j+0.)*(j+1.));
      }
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
  double *l_arr,*cl_arr;

  if(corr_type==CCL_CORR_LM || corr_type==CCL_CORR_LP){
    *status=CCL_ERROR_NOT_IMPLEMENTED;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: CCL does not support full-sky xi+- calcuations.\nhttps://arxiv.org/abs/1702.05301 indicates flat-sky to be sufficient.\n");
    return;
  }
  l_arr=malloc((ELL_MAX_FFTLOG+1)*sizeof(double));
  if(l_arr==NULL) {
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_legendre ran out of memory\n");
    return;
  }
  cl_arr=malloc((ELL_MAX_FFTLOG+1)*sizeof(double));
  if(cl_arr==NULL) {
    free(l_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_legendre ran out of memory\n");
    return;
  }


  //Interpolate input Cl into 
  SplPar *cl_spl=ccl_spline_init(n_ell,ell,cls,cls[0],0);
  if(cl_spl==NULL) {
    free(cl_arr);
    free(l_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_legendre ran out of memory\n");
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
  for(i=0;i<=ELL_MAX_FFTLOG;i++) {
    double l=(double)i;
    l_arr[i]=l;
    if(l>=l_edge)
      cl_arr[i]=cl_edge*pow(l/l_edge,cl_tilt);
    else
      cl_arr[i]=ccl_spline_eval(l,cl_spl);
  }
  ccl_spline_free(cl_spl);

  if (do_taper_cl)
    *status=taper_cl(ELL_MAX_FFTLOG+1,l_arr,cl_arr,taper_cl_limits);

  double **Pl_theta;
  Pl_theta=malloc(n_theta*sizeof(double *));
  if(Pl_theta==NULL) {
    free(cl_arr);
    free(l_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_legendre ran out of memory\n");
    return;
  }
  for (int i=0;i<n_theta;i++) {
    Pl_theta[i]=malloc(sizeof(double)*(ELL_MAX_FFTLOG+1));
    if(Pl_theta[i]==NULL) {
      int j;
      free(cl_arr);
      free(l_arr);
      for(j=0;j<i;j++)
	free(Pl_theta[j]);
      free(Pl_theta);
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_legendre ran out of memory\n");
      return;
    }
  }
  ccl_compute_legendre_polynomial(corr_type,n_theta,theta,ELL_MAX_FFTLOG,Pl_theta);

  for (int i=0;i<n_theta;i++) {
    wtheta[i]=0;
    for(int i_L=1;i_L<ELL_MAX_FFTLOG;i_L+=1)
      wtheta[i]+=cl_arr[i_L]*Pl_theta[i][i_L];
    wtheta[i]/=(M_PI*4);
  }

  for (int i=0;i<n_theta;i++)
    free(Pl_theta[i]);
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
  if(flag_method==CCL_CORR_FFTLOG) {
    ccl_tracer_corr_fftlog(cosmo,n_ell,ell,cls,n_theta,theta,wtheta,corr_type,
			   do_taper_cl,taper_cl_limits,status);
  }
  else if(flag_method==CCL_CORR_LGNDRE) {
    ccl_tracer_corr_legendre(cosmo,n_ell,ell,cls,n_theta,theta,wtheta,corr_type,
			     do_taper_cl,taper_cl_limits,status);
  }
  else if(flag_method==CCL_CORR_BESSEL) {
    ccl_tracer_corr_bessel(cosmo,n_ell,ell,cls,n_theta,theta,wtheta,corr_type,status);
  }
  else {
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

/*--------ROUTINE: ccl_correlation_multipole ------
TASK: Calculate multipole of the redshift space correlation function. Do so using FFTLog.

INPUT:  cosmology, scale factor a, beta (= growth rate / bias), 
        multipole order l = 0, 2, or 4, number of s values, s values

Multipole function result will be in array xi
 */

void ccl_correlation_multipole(ccl_cosmology *cosmo, double a, double beta,
                               int l, int n_s, double *s, double *xi,
                               int *status) {
  int i, N_ARR;
  double *k_arr, *pk_arr, *s_arr, *xi_arr, *xi_arr0;

  N_ARR = (int)(ccl_splines->N_K_3DCOR * log10(ccl_splines->K_MAX / ccl_splines->K_MIN));

  k_arr = ccl_log_spacing(ccl_splines->K_MIN, ccl_splines->K_MAX, N_ARR);
  if (k_arr == NULL) {
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_multipole ran out of memory\n");
    return;
  }

  pk_arr = malloc(N_ARR * sizeof(double));
  if (pk_arr == NULL) {
    free(k_arr);
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_multipole ran out of memory\n");
    return;
  }

  for (i = 0; i < N_ARR; i++)
    pk_arr[i] = ccl_nonlin_matter_power(cosmo, k_arr[i], a, status);

  s_arr = malloc(sizeof(double) * N_ARR);
  if (s_arr == NULL) {
    free(k_arr);
    free(pk_arr);
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_multipole ran out of memory\n");
    return;
  }
  xi_arr = malloc(sizeof(double) * N_ARR);
  if (xi_arr == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_multipole ran out of memory\n");
    return;
  }
  xi_arr0 = malloc(sizeof(double) * N_ARR);
  if (xi_arr0 == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_multipole ran out of memory\n");
    return;
  }

  for (i = 0; i < N_ARR; i++) s_arr[i] = 0;

  // Calculate multipoles

  if (l == 0) {
    fftlog_ComputeXiLM(0, 2, N_ARR, k_arr, pk_arr, s_arr, xi_arr0);
    for (i = 0; i < N_ARR; i++)
      xi_arr[i] = (1. + 2. / 3 * beta + 1. / 5 * beta * beta) * xi_arr0[i];
  } else if (l == 2) {
    fftlog_ComputeXiLM(2, 2, N_ARR, k_arr, pk_arr, s_arr, xi_arr0);
    for (i = 0; i < N_ARR; i++)
      xi_arr[i] = -(4. / 3 * beta + 4. / 7 * beta * beta) * xi_arr0[i];
  } else if (l == 4) {
    fftlog_ComputeXiLM(4, 2, N_ARR, k_arr, pk_arr, s_arr, xi_arr0);
    for (i = 0; i < N_ARR; i++) xi_arr[i] = 8. / 35 * beta * beta * xi_arr0[i];
  } else {
    strcpy(cosmo->status_message, "unavailable value of l\n");
    return;
  }

  // Interpolate to output values of s
  SplPar *xi_spl = ccl_spline_init(N_ARR, s_arr, xi_arr, xi_arr[0], 0);
  for (i = 0; i < n_s; i++) xi[i] = ccl_spline_eval(s[i], xi_spl);
  ccl_spline_free(xi_spl);

  free(k_arr);
  free(pk_arr);
  free(s_arr);
  free(xi_arr);
  free(xi_arr0);

  ccl_check_status(cosmo, status);

  return;
}

/*--------ROUTINE: ccl_correlation_multipole_spline ------
TASK: Store multipoles of the redshift-space correlation in global splines

INPUT:  cosmology, scale factor a

Result is stored in xir_spline[]
 */

void ccl_correlation_multipole_spline(ccl_cosmology *cosmo, double a,
                                      int *status) {
  int i, N_ARR;
  double *k_arr, *pk_arr, *s_arr, *xi_arr, *xi_arr0, *xi_arr2, *xi_arr4;

  N_ARR = (int)(ccl_splines->N_K_3DCOR * log10(ccl_splines->K_MAX / ccl_splines->K_MIN));

  k_arr = ccl_log_spacing(ccl_splines->K_MIN, ccl_splines->K_MAX, N_ARR);
  if (k_arr == NULL) {
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_multipole_spline ran out of "
           "memory\n");
    return;
  }

  pk_arr = malloc(N_ARR * sizeof(double));
  if (pk_arr == NULL) {
    free(k_arr);
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_multipole_spline ran out of "
           "memory\n");
    return;
  }

  for (i = 0; i < N_ARR; i++)
    pk_arr[i] = ccl_nonlin_matter_power(cosmo, k_arr[i], a, status);

  s_arr = malloc(sizeof(double) * N_ARR);
  if (s_arr == NULL) {
    free(k_arr);
    free(pk_arr);
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_multipole_spline ran out of "
           "memory\n");
    return;
  }
  xi_arr = malloc(sizeof(double) * N_ARR);
  if (xi_arr == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_multipole_spline ran out of "
           "memory\n");
    return;
  }
  xi_arr0 = malloc(sizeof(double) * N_ARR);
  if (xi_arr0 == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_multipole_spline ran out of "
           "memory\n");
    return;
  }
  xi_arr2 = malloc(sizeof(double) * N_ARR);
  if (xi_arr2 == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_multipole_spline ran out of "
           "memory\n");
    return;
  }
  xi_arr4 = malloc(sizeof(double) * N_ARR);
  if (xi_arr4 == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_multipole_spline ran out of "
           "memory\n");
    return;
  }

  for (i = 0; i < N_ARR; i++) s_arr[i] = 0;

  // Calculate multipoles

  fftlog_ComputeXiLM(0, 2, N_ARR, k_arr, pk_arr, s_arr, xi_arr0);
  fftlog_ComputeXiLM(2, 2, N_ARR, k_arr, pk_arr, s_arr, xi_arr2);
  fftlog_ComputeXiLM(4, 2, N_ARR, k_arr, pk_arr, s_arr, xi_arr4);

  // Interpolate to output values of s
  xir_spline[0] = ccl_spline_init(N_ARR, s_arr, xi_arr0, xi_arr0[0], 0);
  xir_spline[1] = ccl_spline_init(N_ARR, s_arr, xi_arr2, xi_arr2[0], 0);
  xir_spline[2] = ccl_spline_init(N_ARR, s_arr, xi_arr4, xi_arr4[0], 0);

  free(k_arr);
  free(pk_arr);
  free(s_arr);
  free(xi_arr);
  free(xi_arr0);
  free(xi_arr2);
  free(xi_arr4);

  ccl_check_status(cosmo, status);

  return;
}

/*--------ROUTINE: ccl_correlation_3dRsd ------
TASK: free the splines

*/
void ccl_correlation_multipole_spline_free() {
  ccl_spline_free(xir_spline[0]);
  ccl_spline_free(xir_spline[1]);
  ccl_spline_free(xir_spline[2]);
  xir_spline[0] = NULL;
  xir_spline[1] = NULL;
  xir_spline[2] = NULL;

  return;
}

/*--------ROUTINE: ccl_correlation_3dRsd ------
TASK: Calculate the redshift-space correlation function.  

INPUT:  cosmology, scale factor a, number of s values, s values, 
        mu = cosine of galaxy separation angle w.r.t. line of sight,
        beta (= growth rate / bias), key for using spline

Correlation function result will be in array xi
 */

void ccl_correlation_3dRsd(ccl_cosmology *cosmo, double a, int n_s, double *s,
                           double mu, double beta, double *xi, int use_spline,
                           int *status) {
  int i;
  double *xi_arr0, *xi_arr2, *xi_arr4;

  if (use_spline == 0) {
    xi_arr0 = malloc(sizeof(double) * n_s);
    if (xi_arr0 == NULL) {
      *status = CCL_ERROR_MEMORY;
      strcpy(cosmo->status_message,
             "ccl_correlation.c: ccl_correlation_3dRsd ran out of memory\n");
      return;
    }
    xi_arr2 = malloc(sizeof(double) * n_s);
    if (xi_arr2 == NULL) {
      free(xi_arr0);
      *status = CCL_ERROR_MEMORY;
      strcpy(cosmo->status_message,
             "ccl_correlation.c: ccl_correlation_3dRsd ran out of memory\n");
      return;
    }
    xi_arr4 = malloc(sizeof(double) * n_s);
    if (xi_arr4 == NULL) {
      free(xi_arr2);
      free(xi_arr4);
      *status = CCL_ERROR_MEMORY;
      strcpy(cosmo->status_message,
             "ccl_correlation.c: ccl_correlation_3dRsd ran out of memory\n");
      return;
    }

    ccl_correlation_multipole(cosmo, a, beta, 0, n_s, s, xi_arr0, status);
    ccl_correlation_multipole(cosmo, a, beta, 2, n_s, s, xi_arr2, status);
    ccl_correlation_multipole(cosmo, a, beta, 4, n_s, s, xi_arr4, status);
    for (i = 0; i < n_s; i++)
      xi[i] = xi_arr0[i] + xi_arr2[i] * gsl_sf_legendre_Pl(2, mu) +
              xi_arr4[i] * gsl_sf_legendre_Pl(4, mu);
    free(xi_arr0);
    free(xi_arr2);
    free(xi_arr4);

  } else {
    if (xir_spline[0] == NULL)
      ccl_correlation_multipole_spline(cosmo, a, status);

    for (i = 0; i < n_s; i++)
      xi[i] = (1. + 2. / 3 * beta + 1. / 5 * beta * beta) *
                  ccl_spline_eval(s[i], xir_spline[0]) -
              (4. / 3 * beta + 4. / 7 * beta * beta) *
                  ccl_spline_eval(s[i], xir_spline[1]) *
                  gsl_sf_legendre_Pl(2, mu) +
              8. / 35 * beta * beta * ccl_spline_eval(s[i], xir_spline[2]) *
                  gsl_sf_legendre_Pl(4, mu);
  }

  ccl_check_status(cosmo, status);

  return;
}

/*--------ROUTINE: ccl_correlation_3dRsd_avgmu ------
TASK: Calculate the average of redshift-space correlation function xi(s,mu) over mu at constant s 

INPUT:  cosmology, scale factor a, number of s values, s values, beta (= growth rate / bias)

The result will be in array xi
*/

void ccl_correlation_3dRsd_avgmu(ccl_cosmology *cosmo, double a, int n_s, double *s,
                                 double beta, double *xi,
                                 int *status) {
// The average is just the l=0 multipole - the higher multiples inetegrate to zero.
  ccl_correlation_multipole(cosmo, a, beta, 0, n_s, s, xi, status);  

  ccl_check_status(cosmo, status);

  return;
}

/*--------ROUTINE: ccl_correlation_pi_sigma ------
TASK: Calculate the redshift-space correlation function using longitudinal and 
      transverse coordinates pi and sigma.

INPUT:  cosmology, scale factor a, beta (= growth rate / bias),
        pi, number of sigma values, sigma values, 
        key for using spline

Correlation function result will be in array xi
*/

void ccl_correlation_pi_sigma(ccl_cosmology *cosmo, double a, double beta,
                              double pi, int n_sig, double *sig, double *xi,
                              int use_spline, int *status) {
  int i;
  double *mu_arr, *s_arr, *xi_arr;

  mu_arr = malloc(sizeof(double) * n_sig);
  if (mu_arr == NULL) {
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_pi_sigma ran out of memory\n");
    return;
  }

  s_arr = malloc(sizeof(double) * n_sig);
  if (s_arr == NULL) {
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_pi_sigma ran out of memory\n");
    return;
  }

  xi_arr = malloc(sizeof(double) * n_sig);
  if (xi_arr == NULL) {
    *status = CCL_ERROR_MEMORY;
    strcpy(cosmo->status_message,
           "ccl_correlation.c: ccl_correlation_pi_sigma ran out of memory\n");
    return;
  }

  for (i = 0; i < n_sig; i++) {
    s_arr[i] = sqrt(pi * pi + sig[i] * sig[i]);
    mu_arr[i] = pi / s_arr[i];
  }

  for (i = 0; i < n_sig; i++) {
    ccl_correlation_3dRsd(cosmo, a, n_sig, s_arr, mu_arr[i], beta, xi_arr,
                          use_spline, status);
    xi[i] = xi_arr[i];
  }

  free(mu_arr);
  free(xi_arr);
  free(s_arr);
  
  ccl_check_status(cosmo, status);

  return;
}
