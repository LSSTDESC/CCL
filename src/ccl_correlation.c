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

//ccl_angular_cl like function for test case. Hankel tranform of 1./l is 1./theta (uto factors of 2\pi)
double angular_l_inv2(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,CCL_ClTracer *clt2, int * status)
{
  if (l==0)
    return 0;
  else
    return 1./l;
  }

/*Binning the computed correlation function*/
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

/*Applying cosine tapering to cls to reduce aliasing*/
int taper_cl(int n_ell,double *ell,double *cl, double *ell_limits)
{
  //ell_limits=[low_ell_limit_lower,low_ell_limit_upper,high_ell_limit_lower,high_ell_limit_upper ]

  for (int i=0;i<n_ell;i++)
    {
      if (ell[i]<ell_limits[0] || ell[i]>ell_limits[3])
	{
	  cl[i]=0;//ell outside desirable range
	  continue;
	}
      if (ell[i]>=ell_limits[1] && ell[i]<=ell_limits[2])
	continue;//ell within good ell range

      if (ell[i]<ell_limits[1])//tapering low ell
	cl[i]*=cos((ell[i]-ell_limits[1])/(ell_limits[1]-
						ell_limits[0])*M_PI/2.);
  
      if (ell[i]>ell_limits[2])//tapering high ell
	  cl[i]*=cos((ell[i]-ell_limits[2])/(ell_limits[3]-
						   ell_limits[2])*M_PI/2.);
    }
  return 0;
}

/*--------ROUTINE: ccl_tracer_corr ------
TASK: For a given tracer, get the correlation function
INPUT: type of tracer, number of theta values to evaluate = NL, theta vector
 */

int ccl_tracer_corr(ccl_cosmology *cosmo, int n_theta, double **theta,
                    CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel,
		    bool do_taper_cl,double *taper_cl_limits, 
		    double **corr_func)
{
  return ccl_tracer_corr2(cosmo, n_theta,theta,ct1,ct2,i_bessel,do_taper_cl,taper_cl_limits,
			  corr_func,ccl_angular_cl);
}


/*Following function takes a function to calculate angular cl as well. By default above function will call it using ccl_angular_cl*/
int ccl_tracer_corr2(ccl_cosmology *cosmo, int n_theta, double **theta, 
		    CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel,
		     bool do_taper_cl,double *taper_cl_limits,double **corr_func, 
		    double (*angular_cl)(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,
					 CCL_ClTracer *clt2, int * status) ){

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

  double *l_arr;
  double cl_arr[n_theta];
  //ccl_angular_cl expects ell to be integer... type conversion later
  
  // l_arr=ccl_log_spacing(L_MIN_INT,L_MAX_INT,n_theta);
  l_arr=ccl_log_spacing(.01,60000,n_theta); 

  int status=0,l2=0;
  for(int i=0;i<n_theta;i+=1) {
    if (l_arr[i]<1)
      {
	cl_arr[i]=0;
	//cl_arr[i]=1./l_arr[i];//works for the 1/ell analytical function.
	//cl_arr[i]=1./sqrt(l_arr[i]*l_arr[i]+1);
	//cl_arr[i]=exp(-0.5*l_arr[i]*l_arr[i]*1);
	continue;
      }
    l_arr[i]=(int)l_arr[i];//conversion since cl function require integers
    //this leads to repeated ell in the array, especially at low ell

    cl_arr[i]=angular_cl(cosmo,l_arr[i],ct1,ct2,&status); 

    //    cl_arr[i]*=sqrt(l_arr[i]);
    /*during FFTlog, we need to multiply cl with ell. 
      However, FFTlog only multiplies by m-0.5, where m is an int. 
      We set m=1 and multiply here by sqrt(ell) here to compensate.
      Whole thing can be sorted by changing m to double in FFTlog and
      then passing m=1.5.
      Update: Changed FFTlog to take in m as double.
    */
  }

  if (do_taper_cl)
    status=taper_cl(n_theta,l_arr,cl_arr, taper_cl_limits);
 
  *theta=(double *)malloc(sizeof(double)*n_theta);
  *corr_func=(double *)malloc(sizeof(double)*n_theta);

  
  for(int i=0;i<n_theta;i++)
    {
      (*theta)[i]=0;//1./l_arr[n_theta-i-1]; 
    }//theta is modified by the fftlog 
  
  /* FFTlog uses spherical bessel functions, j_n, but for projected 
     correlations we need bessel functions of first order, J_n.
     To compensate for the difference, we use the relation
     j_n(x) = sqrt(Pi/2x)J_{n+1/2}(x)
     J_{m}(x) = sqrt(2x/Pi) j_{m-1/2}(x)*/

  fftlog_ComputeXiLM(i_bessel-0.5, 1.5 , n_theta , l_arr, cl_arr, *theta,*corr_func);
    
  for(int i=0;i<n_theta;i++)
    {
      //(*corr_func)[i]=M_PI*(*corr_func)[i]*sqrt(2.0*(*theta)[i]/M_PI); 
      (*corr_func)[i]*=sqrt((*theta)[i]*2.0*M_PI);//same as above line
    }
  return 0;
}

