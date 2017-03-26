#include "gsl/gsl_integration.h"
#include "ccl_cls.h"
#include "gsl/gsl_roots.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_sf_bessel.h"
#include "ccl_error.h"
#include "ccl_utils.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "ccl_power.h"
#include "ccl.h"
#include "fftlog.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

typedef struct
{
  gsl_spline *cl; 
  double theta;
  int i_bessel;
} corr_int_par;

/*--------ROUTINE: ccl_corr_integrand ------
TASK: Compute the integrand of the correlation function
INPUT: ell-value and a params structure defined above.
 */

/*
To Do:
- Implement tapering function, to taper cls at both low ell and high ell regime. 
  This will reduce the ringing
- Implement a binning function
- Optional: Implement a function to use GSL implementation of hankel transform.
 */
static double ccl_corr_integrand(double l, void *params)
{
  corr_int_par *p=(corr_int_par *) params;
  double bessel_j=gsl_sf_bessel_Jn(p->i_bessel,p->theta*l);
  return l*bessel_j*gsl_spline_eval(p->cl,l,NULL);
}

/*--------ROUTINE: ccl_general_corr ------
TASK: Compute the correlation function by passing it a spline of Cl and a bessel function index
INPUT: Cl spline, theta-vector, correlation function vector, number of theta values, index of the bessel function. NB: length of theta and corr_func must match.
TODO: Check normalization of correlation function. 
*/
static void ccl_general_corr(gsl_spline *cl, double *theta, double *corr_func, int n_theta, int i_bessel)
{
  gsl_function F;
  corr_int_par cp;
  double result,eresult;

  cp.i_bessel=i_bessel;
  cp.cl=cl;

  gsl_integration_workspace *w=gsl_integration_workspace_alloc(GSL_INTEGRATION_LIMIT);
  /*Alternative QAWO integration, but omega should be revised:
  const double omega = 1;
  const double L = L_MAX_INT-L_MIN_INT;
  gsl_integration_qawo_table* wf = gsl_integration_qawo_table_alloc(omega, L, GSL_INTEG_SINE, GSL_INTEGRATION_LIMIT);*/
  /* Alternative cquag integrator 
     gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc (1000);
  */

  for (int i=0;i<n_theta;i++)
    {
      cp.theta=theta[i];
      F.function=&ccl_corr_integrand;
      F.params=&cp;
      //This is another integration option: qawo is supposedly better for oscillatory functions but it isn't working either.
      //int status = gsl_integration_qawo (&F,L_MIN_INT,0,EPSREL_CORR_FUNC,GSL_INTEGRATION_LIMIT,w,wf,&result,&eresult);
      //Original integrator
      gsl_integration_qag(&F,L_MIN_INT,L_MAX_INT,0,EPSREL_CORR_FUNC,GSL_INTEGRATION_LIMIT,GSL_INTEG_GAUSS41,w,&result,&eresult);
      /*Integrator we are using for photo-z
	gsl_integration_cquad(&F,L_MIN_INT,L_MAX_INT,0,EPSREL_CORR_FUNC,w, &result, &eresult, NULL);*/
      corr_func[i]=result/(2*M_PI);
    }
  
  //gsl_integration_qawo_table_free(wf);
  //gsl_integration_cquad_workspace_free(w);
  gsl_integration_workspace_free(w);

  return;
}

int bin_corr(int n_theta, double *theta, double *corr_func,int n_theta_bins, 
	     double *theta_bins, double *corr_func_binned)
{
  double theta_integrand_lim[2]={0,0};
  double bin_norm=0;
  int j_start=0;
  for (int i=0;i<n_theta_bins-1;i++)
    {
      bin_norm=0;
      theta_integrand_lim[0]=theta_bins[i];
      for (int j=j_start;j<n_theta;j++)
	{
	  if (theta[j]<theta_bins[i])
	      continue;
	  if (theta[j]>theta_bins[i+1])
	    {	    
	      corr_func_binned[i]/=bin_norm;
	      j_start=j;//this assumes theta is monotonically increasing
	      break;//move onto next bin
	    }
	  if (j!=n_theta-1)
	    if (theta[j+1]<theta_bins[i+1])//min(theta[j+1],theta_bins[i+1])
	      theta_integrand_lim[1]=theta[j+1];
	    else
	      theta_integrand_lim[1]=theta_bins[i+1];
	  else
	    theta_integrand_lim[1]=theta_bins[i+1];
	  corr_func_binned[i]+=theta[j]*corr_func[j]*(theta_integrand_lim[1]-
						      theta_integrand_lim[0]);
	  bin_norm+=theta[j]*(theta_integrand_lim[1]-
				  theta_integrand_lim[0]);
	  theta_integrand_lim[0]=theta_integrand_lim[1];
	  if(j==n_theta-1)
	    corr_func_binned[i]/=bin_norm;
	}
    }
  return 0;
}

int taper_cl(int n_ell,double *ell,double *cl, double *low_ell_limit,double *high_ell_limit)
{
  for (int i=0;i<n_ell;i++)
    {
      if (ell[i]<low_ell_limit[0] || ell[i]>high_ell_limit[1])
	{
	  cl[i]=0;//ell outside desirable range
	  continue;
	}
      if (ell[i]>=low_ell_limit[1] && ell[i]<=high_ell_limit[0])
	continue;//ell within good ell range
      //printf("tapering %.3e %.3e ",ell[i],cl[i]);
      if (ell[i]<low_ell_limit[1])//tapering low ell
	cl[i]*=cos((ell[i]-low_ell_limit[1])/(low_ell_limit[1]-
						low_ell_limit[0])*M_PI/2.);
  
      if (ell[i]>high_ell_limit[0])//tapering high ell
	  cl[i]*=cos((ell[i]-high_ell_limit[0])/(high_ell_limit[1]-
						   high_ell_limit[0])*M_PI/2.);
      //printf(" %.3e\n",cl[i]);
    }
  return 0;
}

/*--------ROUTINE: ccl_tracer_corr ------
TASK: For a given tracer, get the correlation function
INPUT: type of tracer, number of theta values to evaluate = NL, theta vector
 */
int ccl_tracer_corr(ccl_cosmology *cosmo, int n_theta, double **theta, CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel,double **corr_func){
  /* do we need to input i_bessel? could just be set here based on tracer..*/
  if((ct1->tracer_type==CL_TRACER_WL) && (ct2->tracer_type==CL_TRACER_WL)){
    if((i_bessel!=0) && (i_bessel!=4)) return 1;
  }
  if((ct1->tracer_type==CL_TRACER_NC) && (ct2->tracer_type==CL_TRACER_NC)){
    if(i_bessel!=0) return 1;
  }
  if(((ct1->tracer_type==CL_TRACER_WL) && (ct2->tracer_type==CL_TRACER_NC)) || ((ct1->tracer_type==CL_TRACER_NC) && (ct2->tracer_type==CL_TRACER_WL))){
    if(i_bessel!=2) return 1;
  }  

  double *l_arr,cl_arr[n_theta];
  
  // l_arr=ccl_log_spacing(L_MIN_INT,L_MAX_INT,n_theta);
  l_arr=ccl_log_spacing(.5,60000,n_theta); 
  /*Zero padding rule of thumb: Extend ell range by ~ factor of 3 (e)
    on both low and high ell end. Then use the tapering 
    function to smoothly set cl to zero outside the sensible range.
   */

  int status=0;
  for(int i=0;i<n_theta;i+=1) {
    //Re-scaling the power-spectrum due to Bessel function missing factor
    cl_arr[i]=ccl_angular_cl(cosmo,l_arr[i],ct1,ct2,&status)*sqrt(l_arr[i]); 
    //    printf("ell cl %.3e %.3e",l_arr[i],cl_arr[i]);
  }
  
  double taper_low_ell_limit[2]={1,2};
  double taper_high_ell_limit[2]={30000,50000};

  status=taper_cl(n_theta,l_arr,cl_arr, taper_low_ell_limit, taper_high_ell_limit);


  *theta=(double *)malloc(sizeof(double)*n_theta);
  *corr_func=(double *)malloc(sizeof(double)*n_theta);

  for(int i=0;i<n_theta;i++)
    {
      (*theta)[i]=1./l_arr[n_theta-i-1]; 
    }
  
  /* FFTlog uses spherical bessel functions, j_n, but for projected 
     correlations we need bessel functions of first order, J_n.
     To compensate for the difference, we use the relation
     j_n(x) = sqrt(Pi/2x)J_{n+1/2}(x)
     J_{m}(x) = sqrt(2x/Pi) j_{m-1/2}(x)
     Note that the following routine only takes integers */
  fftlog_ComputeXiLM(i_bessel-0.5, 1 , n_theta , l_arr, cl_arr, *theta,*corr_func);
    
  for(int i=0;i<n_theta;i++)
    {
      (*corr_func)[i]=M_PI*(*corr_func)[i]*sqrt(2.0*(*theta)[i]/M_PI); 
    }
  return 0;
}

