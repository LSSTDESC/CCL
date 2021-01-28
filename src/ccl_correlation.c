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

#include "ccl.h"

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
                                   int *status) {
  int i;
  double *l_arr,*cl_arr,*th_arr,*wth_arr;

  l_arr=ccl_log_spacing(cosmo->spline_params.ELL_MIN_CORR,cosmo->spline_params.ELL_MAX_CORR,cosmo->spline_params.N_ELL_CORR);
  if(l_arr==NULL) {
    *status=CCL_ERROR_LINSPACE;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_fftlog(): ran out of memory\n");
    return;
  }
  cl_arr=malloc(cosmo->spline_params.N_ELL_CORR*sizeof(double));
  if(cl_arr==NULL) {
    free(l_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_tracer_corr_fftlog(): ran out of memory\n");
    return;
  }

  //Interpolate input Cl into array needed for FFTLog
  ccl_f1d_t *cl_spl=ccl_f1d_t_new(n_ell,ell,cls,cls[0],0,
				  ccl_f1d_extrap_const,
				  ccl_f1d_extrap_logx_logy,
				  status);
  if (*status) {
    free(l_arr);
    free(cl_arr);
    ccl_cosmology_set_status_message(cosmo,
                                     "ccl_correlation.c: ccl_tracer_corr_fftlog(): "
                                     "failed to create spline\n");
    if (cl_spl) ccl_f1d_t_free(cl_spl);
    return;
  }

  if(cl_spl==NULL) {
    free(l_arr);
    free(cl_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
				     "ccl_correlation.c: ccl_tracer_corr_fftlog(): "
				     "ran out of memory\n");
    return;
  }

  for(i=0;i<cosmo->spline_params.N_ELL_CORR;i++)
    cl_arr[i]=ccl_f1d_t_eval(cl_spl,l_arr[i]);
  ccl_f1d_t_free(cl_spl);

  if (do_taper_cl)
    taper_cl(cosmo->spline_params.N_ELL_CORR,l_arr,cl_arr,taper_cl_limits);

  th_arr=malloc(sizeof(double)*cosmo->spline_params.N_ELL_CORR);
  if(th_arr==NULL) {
    free(l_arr);
    free(cl_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
                                     "ccl_correlation.c: ccl_tracer_corr_fftlog(): "
                                     "ran out of memory\n");
    return;
  }
  wth_arr=(double *)malloc(sizeof(double)*cosmo->spline_params.N_ELL_CORR);
  if(wth_arr==NULL) {
    free(l_arr);
    free(cl_arr);
    free(th_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
                                     "ccl_correlation.c: ccl_tracer_corr_fftlog(): "
                                     "ran out of memory\n");
    return;
  }

  for(i=0;i<cosmo->spline_params.N_ELL_CORR;i++)
    th_arr[i]=0;
  //Although set here to 0, theta is modified by FFTlog to obtain the correlation at ~1/l

  int i_bessel=0;
  if(corr_type==CCL_CORR_GG) i_bessel=0;
  if(corr_type==CCL_CORR_GL) i_bessel=2;
  if(corr_type==CCL_CORR_LP) i_bessel=0;
  if(corr_type==CCL_CORR_LM) i_bessel=4;
  ccl_fftlog_ComputeXi2D(i_bessel,0,
			 1, cosmo->spline_params.N_ELL_CORR,l_arr,&cl_arr,
			 th_arr,&wth_arr, status);

  // Interpolate to output values of theta
  ccl_f1d_t *wth_spl=ccl_f1d_t_new(cosmo->spline_params.N_ELL_CORR,th_arr,
				   wth_arr,wth_arr[0],0,
				   ccl_f1d_extrap_const,
				   ccl_f1d_extrap_const, status);
  if (wth_spl == NULL) {
    free(l_arr);
    free(cl_arr);
    free(th_arr);
    free(wth_arr);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
                                     "ccl_correlation.c: ccl_tracer_corr_fftlog(): "
                                     "ran out of memory\n");
    return;
  }
  for(i=0;i<n_theta;i++)
    wtheta[i]=ccl_f1d_t_eval(wth_spl,theta[i]*M_PI/180.);
  ccl_f1d_t_free(wth_spl);

  free(l_arr);
  free(cl_arr);
  free(th_arr);
  free(wth_arr);

  return;
}

typedef struct {
  ccl_f1d_t *cl_spl;
  int i_bessel;
  double th;
} corr_int_par;

static double corr_bessel_integrand(double l,void *params)
{
  double cl,jbes;
  corr_int_par *p=(corr_int_par *)params;
  double x=l*p->th;

  cl=ccl_f1d_t_eval(p->cl_spl,l);

  jbes=gsl_sf_bessel_Jn(p->i_bessel,x);

  return l*jbes*cl;
}

static void ccl_tracer_corr_bessel(ccl_cosmology *cosmo,
                                   int n_ell,double *ell,double *cls,
                                   int n_theta,double *theta,double *wtheta,
                                   int corr_type,int *status) {
  corr_int_par cp;
  ccl_f1d_t *cl_spl = NULL;
  cl_spl = ccl_f1d_t_new(n_ell, ell, cls, cls[0], 0,
			 ccl_f1d_extrap_const,
			 ccl_f1d_extrap_logx_logy, status);
  if(cl_spl == NULL) {
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(
      cosmo,
      "ccl_correlation.c: ccl_tracer_corr_bessel(): "
      "ran out of memory\n");
    return;
  }

  int ith, gslstatus;
  double result,eresult;
  gsl_function F;
  gsl_integration_workspace *w = NULL;
  int local_status;

#pragma omp parallel default(none) \
                     shared(cosmo, status, wtheta, n_ell, ell, cls, \
			    corr_type, cl_spl, theta, n_theta)	\
                     private(w, F, result, eresult, local_status, ith, \
			     gslstatus, cp)
  {
    local_status = *status;

    switch(corr_type) {
      case CCL_CORR_GG:
        cp.i_bessel = 0;
        break;
      case CCL_CORR_GL:
        cp.i_bessel = 2;
        break;
      case CCL_CORR_LP:
        cp.i_bessel = 0;
        break;
      case CCL_CORR_LM:
        cp.i_bessel = 4;
        break;
    }

    cp.cl_spl = cl_spl;

    w = gsl_integration_workspace_alloc(cosmo->gsl_params.N_ITERATION);

    if (w == NULL) {
      local_status = CCL_ERROR_MEMORY;
    }
    F.function = &corr_bessel_integrand;
    F.params = &cp;

    #pragma omp for schedule(dynamic)
    for(ith=0; ith < n_theta; ith++) {
      if (local_status == 0) {
        cp.th = theta[ith]*M_PI/180;
        //TODO: Split into intervals between first bessel zeros before integrating
        //This will help both speed and accuracy of the integral.
        gslstatus = gsl_integration_qag(&F, 0, cosmo->spline_params.ELL_MAX_CORR, 0,
                                        cosmo->gsl_params.INTEGRATION_EPSREL, cosmo->gsl_params.N_ITERATION,
                                        cosmo->gsl_params.INTEGRATION_GAUSS_KRONROD_POINTS,
                                        w, &result, &eresult);
        if(gslstatus != GSL_SUCCESS) {
          ccl_raise_gsl_warning(gslstatus, "ccl_correlation.c: ccl_tracer_corr_bessel():");
          local_status |= gslstatus;
        }
        wtheta[ith] = result/(2*M_PI);
      }
    }

    if (local_status) {
      #pragma omp atomic write
      *status = local_status;
    }

    gsl_integration_workspace_free(w);
  }
  ccl_f1d_t_free(cl_spl);
}


/*--------ROUTINE: ccl_compute_legendre_polynomial ------
TASK: Compute input factor for ccl_tracer_corr_legendre
INPUT: tracer 1, tracer 2, i_bessel, theta array, n_theta, L_max, output Pl_theta
 */
static void ccl_compute_legendre_polynomial(int corr_type,double theta,int ell_max,double *Pl_theta)
{
  int j;
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
                                     int *status) {
  int i;
  double *l_arr = NULL, *cl_arr = NULL, *Pl_theta = NULL;
  ccl_f1d_t *cl_spl;

  if(corr_type==CCL_CORR_LM || corr_type==CCL_CORR_LP){
    *status=CCL_ERROR_NOT_IMPLEMENTED;
    ccl_cosmology_set_status_message(cosmo,
                                     "ccl_correlation.c: ccl_tracer_corr_legendre(): "
                                     "CCL does not support full-sky xi+- calcuations.\nhttps://arxiv.org/abs/1702.05301 indicates flat-sky to be sufficient.\n");
  }

  if(*status==0) {
    l_arr=malloc(((int)(cosmo->spline_params.ELL_MAX_CORR)+1)*sizeof(double));
    if(l_arr==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
                                       "ccl_correlation.c: ccl_tracer_corr_legendre(): "
                                       "ran out of memory\n");
    }
  }

  if(*status==0) {
    cl_arr=malloc(((int)(cosmo->spline_params.ELL_MAX_CORR)+1)*sizeof(double));
    if(cl_arr==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
                                       "ccl_correlation.c: ccl_tracer_corr_legendre(): "
                                       "ran out of memory\n");
    }
  }

  if(*status==0) {
    //Interpolate input Cl into
    cl_spl=ccl_f1d_t_new(n_ell,ell,cls,cls[0],0,
			 ccl_f1d_extrap_const,
			 ccl_f1d_extrap_logx_logy, status);
    if(cl_spl==NULL) {
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
                                       "ccl_correlation.c: ccl_tracer_corr_legendre(): "
                                       "ran out of memory\n");
    }
  }

  if(*status==0) {
    for(i=0;i<=(int)(cosmo->spline_params.ELL_MAX_CORR);i++) {
      double l=(double)i;
      l_arr[i]=l;
      cl_arr[i]=ccl_f1d_t_eval(cl_spl,l);
    }
    ccl_f1d_t_free(cl_spl);

    if (do_taper_cl)
      *status=taper_cl((int)(cosmo->spline_params.ELL_MAX_CORR)+1,l_arr,cl_arr,taper_cl_limits);
  }

  int local_status, i_L;
#pragma omp parallel default(none) \
                     shared(cosmo, theta, cl_arr, wtheta, n_theta, status, corr_type) \
                     private(Pl_theta, i, i_L, local_status)
  {
    Pl_theta = NULL;
    local_status = *status;

    if (local_status == 0) {
      Pl_theta = malloc(sizeof(double)*((int)(cosmo->spline_params.ELL_MAX_CORR)+1));
      if (Pl_theta == NULL) {
        local_status = CCL_ERROR_MEMORY;
      }
    }

    #pragma omp for schedule(dynamic)
    for (int i=0; i < n_theta; i++) {
      if (local_status == 0) {
        wtheta[i] = 0;
        ccl_compute_legendre_polynomial(corr_type, theta[i], (int)(cosmo->spline_params.ELL_MAX_CORR), Pl_theta);
        for (i_L=1; i_L < (int)(cosmo->spline_params.ELL_MAX_CORR); i_L+=1)
          wtheta[i] += cl_arr[i_L]*Pl_theta[i_L];
        wtheta[i] /= (M_PI*4);
      }
    }

    if (local_status) {
      #pragma omp atomic write
      *status = local_status;
    }

    free(Pl_theta);
  }
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
                     int *status) {
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
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_correlation(): Unknown algorithm\n");
  }

}

/*--------ROUTINE: ccl_correlation_3d ------
TASK: Calculate the 3d-correlation function. Do so by using FFTLog.

INPUT: cosmology, scale factor a,
       number of r values, r values,
       key for tapering, limits of tapering

Correlation function result will be in array xi
 */

void ccl_correlation_3d(ccl_cosmology *cosmo,
                        ccl_f2d_t *psp, double a,
                        int n_r,double *r,double *xi,
                        int do_taper_pk,double *taper_pk_limits,
                        int *status) {
  int i,N_ARR;
  double *k_arr,*pk_arr,*r_arr,*xi_arr;

  //number of data points for k and pk array
  N_ARR=(int)(cosmo->spline_params.N_K_3DCOR*log10(cosmo->spline_params.K_MAX/cosmo->spline_params.K_MIN));

  k_arr=ccl_log_spacing(cosmo->spline_params.K_MIN,cosmo->spline_params.K_MAX,N_ARR);
  if(k_arr==NULL) {
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_correlation_3d(): ran out of memory\n");
    return;
  }

  pk_arr=malloc(N_ARR*sizeof(double));
  if(pk_arr==NULL) {
    free(k_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_correlation_3d(): ran out of memory\n");
    return;
  }

  for (i=0; i<N_ARR; i++){
    pk_arr[i] = ccl_f2d_t_eval(psp, log(k_arr[i]), a, cosmo, status);
  }
  if (do_taper_pk)
    taper_cl(N_ARR,k_arr,pk_arr,taper_pk_limits);

  r_arr=malloc(sizeof(double)*N_ARR);
  if(r_arr==NULL) {
    free(k_arr);
    free(pk_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_correlation_3d(): ran out of memory\n");
    return;
  }
  xi_arr=malloc(sizeof(double)*N_ARR);
  if(xi_arr==NULL) {
    free(k_arr);
    free(pk_arr);
    free(r_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_correlation_3d(): ran out of memory\n");
    return;
  }

  for(i=0;i<N_ARR;i++)
    r_arr[i]=0;

  ccl_fftlog_ComputeXi3D(0, 0, 1, N_ARR, k_arr, &pk_arr, r_arr, &xi_arr, status);

  // Interpolate to output values of r
  ccl_f1d_t *xi_spl=ccl_f1d_t_new(N_ARR,r_arr,xi_arr,xi_arr[0],0,
				  ccl_f1d_extrap_const,
				  ccl_f1d_extrap_const, status);
  if (xi_spl == NULL) {
    free(k_arr);
    free(pk_arr);
    free(r_arr);
    free(xi_arr);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_correlation.c: ccl_correlation_3d(): ran out of memory\n");
    return;
  }
  for(i=0;i<n_r;i++)
    xi[i]=ccl_f1d_t_eval(xi_spl,r[i]);
  ccl_f1d_t_free(xi_spl);

  free(k_arr);
  free(pk_arr);
  free(r_arr);
  free(xi_arr);

  return;
}

/*--------ROUTINE: ccl_correlation_multipole ------
TASK: Calculate multipole of the redshift space correlation function. Do so using FFTLog.

INPUT:  cosmology, scale factor a, beta (= growth rate / bias),
        multipole order l = 0, 2, or 4, number of s values, s values

Multipole function result will be in array xi
 */
void ccl_correlation_multipole(ccl_cosmology *cosmo, ccl_f2d_t *psp,
                               double a, double beta,
                               int l, int n_s, double *s, double *xi,
                               int *status) {
  int i, N_ARR;
  double *k_arr, *pk_arr, *s_arr, *xi_arr, *xi_arr0;

  N_ARR = (int)(cosmo->spline_params.N_K_3DCOR * log10(cosmo->spline_params.K_MAX / cosmo->spline_params.K_MIN));

  k_arr = ccl_log_spacing(cosmo->spline_params.K_MIN, cosmo->spline_params.K_MAX, N_ARR);
  if (k_arr == NULL) {
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole(): ran out of memory\n");
    return;
  }

  pk_arr = malloc(N_ARR * sizeof(double));
  if (pk_arr == NULL) {
    free(k_arr);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole(): ran out of memory\n");
    return;
  }

  for (i = 0; i < N_ARR; i++)
    pk_arr[i] = ccl_f2d_t_eval(psp, log(k_arr[i]), a, cosmo, status);

  s_arr = malloc(sizeof(double) * N_ARR);
  if (s_arr == NULL) {
    free(k_arr);
    free(pk_arr);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole(): ran out of memory\n");
    return;
  }
  xi_arr = malloc(sizeof(double) * N_ARR);
  if (xi_arr == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole(): ran out of memory\n");
    return;
  }
  xi_arr0 = malloc(sizeof(double) * N_ARR);
  if (xi_arr0 == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    free(xi_arr);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole(): ran out of memory\n");
    return;
  }

  for (i = 0; i < N_ARR; i++) s_arr[i] = 0;

  // Calculate multipoles

  if (l == 0) {
    ccl_fftlog_ComputeXi3D(0, 0, 1, N_ARR, k_arr, &pk_arr, s_arr, &xi_arr0, status);
    for (i = 0; i < N_ARR; i++)
      xi_arr[i] = (1. + 2. / 3 * beta + 1. / 5 * beta * beta) * xi_arr0[i];
  } else if (l == 2) {
    ccl_fftlog_ComputeXi3D(2, 0, 1, N_ARR, k_arr, &pk_arr, s_arr, &xi_arr0, status);
    for (i = 0; i < N_ARR; i++)
      xi_arr[i] = -(4. / 3 * beta + 4. / 7 * beta * beta) * xi_arr0[i];
  } else if (l == 4) {
    ccl_fftlog_ComputeXi3D(4, 0, 1, N_ARR, k_arr, &pk_arr, s_arr, &xi_arr0, status);
    for (i = 0; i < N_ARR; i++) xi_arr[i] = 8. / 35 * beta * beta * xi_arr0[i];
  } else {
    ccl_cosmology_set_status_message(cosmo, "unavailable value of l\n");
    return;
  }

  // Interpolate to output values of s
  ccl_f1d_t *xi_spl = ccl_f1d_t_new(N_ARR, s_arr, xi_arr, xi_arr[0], 0,
				    ccl_f1d_extrap_const,
				    ccl_f1d_extrap_const, status);
  if (xi_spl == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    free(xi_arr);
    free(xi_arr0);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole(): ran out of memory\n");
  }
  for (i = 0; i < n_s; i++) xi[i] = ccl_f1d_t_eval(xi_spl,s[i]);
  ccl_f1d_t_free(xi_spl);

  free(k_arr);
  free(pk_arr);
  free(s_arr);
  free(xi_arr);
  free(xi_arr0);

  return;
}

/*--------ROUTINE: ccl_correlation_multipole_spline ------
TASK: Store multipoles of the redshift-space correlation in global splines

INPUT:  cosmology, scale factor a

Result is stored in cosmo->data.rsd_splines[]
 */

void ccl_correlation_multipole_spline(ccl_cosmology *cosmo, ccl_f2d_t *psp,
                                      double a, int *status) {
  int i, N_ARR;
  double *k_arr, *pk_arr, *s_arr, *xi_arr, *xi_arr0, *xi_arr2, *xi_arr4;

  N_ARR = (int)(cosmo->spline_params.N_K_3DCOR * log10(cosmo->spline_params.K_MAX / cosmo->spline_params.K_MIN));

  k_arr = ccl_log_spacing(cosmo->spline_params.K_MIN, cosmo->spline_params.K_MAX, N_ARR);
  if (k_arr == NULL) {
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole_spline(): "
           "ran out of memory\n");
    return;
  }

  pk_arr = malloc(N_ARR * sizeof(double));
  if (pk_arr == NULL) {
    free(k_arr);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole_spline(): "
           "ran out of memory\n");
    return;
  }

  for (i = 0; i < N_ARR; i++)
    pk_arr[i] = ccl_f2d_t_eval(psp, log(k_arr[i]), a, cosmo, status);

  s_arr = malloc(sizeof(double) * N_ARR);
  if (s_arr == NULL) {
    free(k_arr);
    free(pk_arr);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole_spline(): "
           "ran out of memory\n");
    return;
  }
  xi_arr = malloc(sizeof(double) * N_ARR);
  if (xi_arr == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole_spline(): "
           "ran out of memory\n");
    return;
  }
  xi_arr0 = malloc(sizeof(double) * N_ARR);
  if (xi_arr0 == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    free(xi_arr);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole_spline(): "
           "ran out of memory\n");
    return;
  }
  xi_arr2 = malloc(sizeof(double) * N_ARR);
  if (xi_arr2 == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    free(xi_arr);
    free(xi_arr0);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole_spline(): "
           "ran out of memory\n");
    return;
  }
  xi_arr4 = malloc(sizeof(double) * N_ARR);
  if (xi_arr4 == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    free(xi_arr);
    free(xi_arr0);
    free(xi_arr2);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole_spline(): "
           "ran out of memory\n");
    return;
  }

  for (i = 0; i < N_ARR; i++) s_arr[i] = 0;

  // Calculate multipoles
  ccl_fftlog_ComputeXi3D(0, 0, 1, N_ARR, k_arr, &pk_arr, s_arr, &xi_arr0, status);
  ccl_fftlog_ComputeXi3D(2, 0, 1, N_ARR, k_arr, &pk_arr, s_arr, &xi_arr2, status);
  ccl_fftlog_ComputeXi3D(4, 0, 1, N_ARR, k_arr, &pk_arr, s_arr, &xi_arr4, status);

  // free any memory that may have been allocated
  ccl_f1d_t_free(cosmo->data.rsd_splines[0]);
  ccl_f1d_t_free(cosmo->data.rsd_splines[1]);
  ccl_f1d_t_free(cosmo->data.rsd_splines[2]);
  cosmo->data.rsd_splines[0] = NULL;
  cosmo->data.rsd_splines[1] = NULL;
  cosmo->data.rsd_splines[1] = NULL;

  // Interpolate to output values of s
  cosmo->data.rsd_splines[0] = ccl_f1d_t_new(N_ARR, s_arr, xi_arr0, xi_arr0[0], 0,
					     ccl_f1d_extrap_const,
					     ccl_f1d_extrap_const, status);
  if (cosmo->data.rsd_splines[0] == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    free(xi_arr);
    free(xi_arr0);
    free(xi_arr2);
    free(xi_arr4);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole_spline(): "
           "ran out of memory\n");
    return;
  }

  cosmo->data.rsd_splines[1] = ccl_f1d_t_new(N_ARR, s_arr, xi_arr2, xi_arr2[0], 0,
					     ccl_f1d_extrap_const,
					     ccl_f1d_extrap_const, status);
  if (cosmo->data.rsd_splines[1] == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    free(xi_arr);
    free(xi_arr0);
    free(xi_arr2);
    free(xi_arr4);
    ccl_f1d_t_free(cosmo->data.rsd_splines[0]);
    cosmo->data.rsd_splines[0] = NULL;
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole_spline(): "
           "ran out of memory\n");
    return;
  }

  cosmo->data.rsd_splines[2] = ccl_f1d_t_new(N_ARR, s_arr, xi_arr4, xi_arr4[0], 0,
					     ccl_f1d_extrap_const,
					     ccl_f1d_extrap_const, status);
  if (cosmo->data.rsd_splines[2] == NULL) {
    free(k_arr);
    free(pk_arr);
    free(s_arr);
    free(xi_arr);
    free(xi_arr0);
    free(xi_arr2);
    free(xi_arr4);
    ccl_f1d_t_free(cosmo->data.rsd_splines[0]);
    cosmo->data.rsd_splines[0] = NULL;
    ccl_f1d_t_free(cosmo->data.rsd_splines[1]);
    cosmo->data.rsd_splines[1] = NULL;
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_multipole_spline(): "
           "ran out of memory\n");
    return;
  }

  // set the scale factor
  cosmo->data.rsd_splines_scalefactor = a;

  free(k_arr);
  free(pk_arr);
  free(s_arr);
  free(xi_arr);
  free(xi_arr0);
  free(xi_arr2);
  free(xi_arr4);

  return;
}

/*--------ROUTINE: ccl_correlation_3dRsd ------
TASK: Calculate the redshift-space correlation function.

INPUT:  cosmology, scale factor a, number of s values, s values,
        mu = cosine of galaxy separation angle w.r.t. line of sight,
        beta (= growth rate / bias), key for using spline

Correlation function result will be in array xi
 */

void ccl_correlation_3dRsd(ccl_cosmology *cosmo, ccl_f2d_t *psp,
                           double a, int n_s, double *s,
                           double mu, double beta, double *xi, int use_spline,
                           int *status) {
  int i;
  double *xi_arr0, *xi_arr2, *xi_arr4;

  if (use_spline == 0) {
    xi_arr0 = malloc(sizeof(double) * n_s);
    if (xi_arr0 == NULL) {
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
             "ccl_correlation.c: ccl_correlation_3dRsd(): ran out of memory\n");
      return;
    }
    xi_arr2 = malloc(sizeof(double) * n_s);
    if (xi_arr2 == NULL) {
      free(xi_arr0);
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
             "ccl_correlation.c: ccl_correlation_3dRsd(): ran out of memory\n");
      return;
    }
    xi_arr4 = malloc(sizeof(double) * n_s);
    if (xi_arr4 == NULL) {
      free(xi_arr0);
      free(xi_arr2);
      *status = CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo,
             "ccl_correlation.c: ccl_correlation_3dRsd(): ran out of memory\n");
      return;
    }

    ccl_correlation_multipole(cosmo, psp, a, beta, 0, n_s, s, xi_arr0, status);
    ccl_correlation_multipole(cosmo, psp, a, beta, 2, n_s, s, xi_arr2, status);
    ccl_correlation_multipole(cosmo, psp, a, beta, 4, n_s, s, xi_arr4, status);
    for (i = 0; i < n_s; i++)
      xi[i] = xi_arr0[i] + xi_arr2[i] * gsl_sf_legendre_Pl(2, mu) +
              xi_arr4[i] * gsl_sf_legendre_Pl(4, mu);
    free(xi_arr0);
    free(xi_arr2);
    free(xi_arr4);

  } else {
    if ((cosmo->data.rsd_splines[0] == NULL) ||
        (cosmo->data.rsd_splines[1] == NULL) ||
        (cosmo->data.rsd_splines[2] == NULL) ||
        (cosmo->data.rsd_splines_scalefactor != a))
      ccl_correlation_multipole_spline(cosmo, psp, a, status);

    for (i = 0; i < n_s; i++)
      xi[i] = (1. + 2. / 3 * beta + 1. / 5 * beta * beta) *
        ccl_f1d_t_eval(cosmo->data.rsd_splines[0],s[i]) -
        (4. / 3 * beta + 4. / 7 * beta * beta) *
        ccl_f1d_t_eval(cosmo->data.rsd_splines[1],s[i]) *
        gsl_sf_legendre_Pl(2, mu) +
        8. / 35 * beta * beta * ccl_f1d_t_eval(cosmo->data.rsd_splines[2],s[i]) *
        gsl_sf_legendre_Pl(4, mu);
  }

  return;
}

/*--------ROUTINE: ccl_correlation_3dRsd_avgmu ------
TASK: Calculate the average of redshift-space correlation function xi(s,mu) over mu at constant s

INPUT:  cosmology, scale factor a, number of s values, s values, beta (= growth rate / bias)

The result will be in array xi
*/

void ccl_correlation_3dRsd_avgmu(ccl_cosmology *cosmo, ccl_f2d_t *psp,
                                 double a, int n_s, double *s,
                                 double beta, double *xi,
                                 int *status) {
// The average is just the l=0 multipole - the higher multiples inetegrate to zero.
  ccl_correlation_multipole(cosmo, psp, a, beta, 0, n_s, s, xi, status);

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

void ccl_correlation_pi_sigma(ccl_cosmology *cosmo, ccl_f2d_t *psp,
                              double a, double beta,
                              double pi, int n_sig, double *sig, double *xi,
                              int use_spline, int *status) {
  int i;
  double *mu_arr, *s_arr, *xi_arr;

  mu_arr = malloc(sizeof(double) * n_sig);
  if (mu_arr == NULL) {
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_pi_sigma(): ran out of memory\n");
    return;
  }

  s_arr = malloc(sizeof(double) * n_sig);
  if (s_arr == NULL) {
    free(mu_arr);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_pi_sigma(): ran out of memory\n");
    return;
  }

  xi_arr = malloc(sizeof(double) * n_sig);
  if (xi_arr == NULL) {
    free(mu_arr);
    free(s_arr);
    *status = CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo,
           "ccl_correlation.c: ccl_correlation_pi_sigma(): ran out of memory\n");
    return;
  }

  for (i = 0; i < n_sig; i++) {
    s_arr[i] = sqrt(pi * pi + sig[i] * sig[i]);
    mu_arr[i] = pi / s_arr[i];
  }

  for (i = 0; i < n_sig; i++) {
    ccl_correlation_3dRsd(cosmo, psp, a, n_sig, s_arr, mu_arr[i], beta, xi_arr,
                          use_spline, status);
    xi[i] = xi_arr[i];
  }

  free(mu_arr);
  free(xi_arr);
  free(s_arr);

  return;
}
