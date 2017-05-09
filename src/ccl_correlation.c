#include "gsl/gsl_integration.h"
#include "ccl_cls.h"
#include "gsl/gsl_roots.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_sf_bessel.h"
#include "gsl/gsl_sf_legendre.h"
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
//used to be bin_corr
static int bin_func(int n_theta, double *theta, double *corr_func,int n_theta_bins, 
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
static int taper_cl(int n_ell,int *ell,double *cl, double *ell_limits)
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
  
  /* return ccl_tracer_corr_fftlog(cosmo, n_theta,theta,ct1,ct2,i_bessel,do_taper_cl,taper_cl_limits,
			  corr_func,ccl_angular_cl);
   */
  return ccl_tracer_corr_legendre(cosmo, n_theta,theta,ct1,ct2,i_bessel,do_taper_cl,
				  taper_cl_limits,corr_func,ccl_angular_cl);   
}


/*Following function takes a function to calculate angular cl as well. 
By default above function will call it using ccl_angular_cl*/
int ccl_tracer_corr_fftlog(ccl_cosmology *cosmo, int n_theta, double **theta, 
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

  if (do_taper_cl)//also takes in int l_arr
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

int compute_legedre_polynomial(CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel,
				double **theta, int n_theta, int L_max, double **Pl_theta)
{
  double Nl2=0;//Nl**2
  double *Gp,*Gm,*Plm;
  int Plm_indx;
  if((ct1->tracer_type==CL_TRACER_WL) && (ct2->tracer_type==CL_TRACER_WL))
    {
      Gp=(double *)malloc(sizeof(double)*L_max);
      Gm=(double *)malloc(sizeof(double)*L_max);
    }
  if((ct1->tracer_type==CL_TRACER_WL) || (ct2->tracer_type==CL_TRACER_WL))
    {
      int Plm_size=gsl_sf_legendre_array_n(L_max);
      Plm=(double *)malloc(sizeof(double)*Plm_size);
    }

  for (int i=0;i<n_theta;i++)
    {
      if((ct1->tracer_type==CL_TRACER_NC) && (ct2->tracer_type==CL_TRACER_NC))
	{
	  gsl_sf_legendre_Pl_array(L_max,cos((*theta)[i]),Pl_theta[i]);
	  for (int j=0;j<L_max;j++)
	    Pl_theta[i][j]*=(2*j+1);
	}	   

      else if(((ct1->tracer_type==CL_TRACER_WL) && (ct2->tracer_type==CL_TRACER_NC))
	      ||((ct1->tracer_type==CL_TRACER_NC) && (ct2->tracer_type==CL_TRACER_WL)))
	{//https://arxiv.org/pdf/1007.4809.pdf
	  //gsl_sf_legendre_Plm_array(L_max,2,cos((*theta)[i]),Pl_theta[i]);//deprecated in gsl
	  for (int j=2;j<L_max;j++)
	    {
	      //Pl_theta[i][j]*=(2*j+1)/(j*(j+1));//https://arxiv.org/pdf/1007.4809.pdf
	     Nl2=2.0/((double)((j-1)*j*(j+1)*(j+2)));//https://arxiv.org/pdf/astro-ph/9611125v1.pdf
	      Pl_theta[i][j]*=(2*j+1)*sqrt(Nl2);
	    }
	  Pl_theta[i][0]=0;
	  Pl_theta[i][1]=0;
	}

      else if((ct1->tracer_type==CL_TRACER_WL) && (ct2->tracer_type==CL_TRACER_WL))
	{//https://arxiv.org/pdf/astro-ph/9611125v1.pdf
	  //gsl_sf_legendre_Plm_array(L_max,2,cos((*theta)[i]),Pl_theta[i]);//deprecated in gsl
	  //gsl_sf_legendre_array(GSL_SF_LEGENDRE_NONE,L_max,cos((*theta)[i]),Plm);//too slow
	  if (i_bessel==0)
	    {
	      gsl_sf_legendre_Pl_array(L_max,cos((*theta)[i]),Pl_theta[i]);
	      for (int j=0;j<L_max;j++)
		Pl_theta[i][j]*=(2*j+1);
	    }
	  else{
	    gsl_sf_legendre_Pl_array(L_max,cos((*theta)[i]),Pl_theta[i]);
	    for (int j=0;j<L_max;j++)
	      Pl_theta[i][j]*=0;
	  }
	}
    }
  return 0;
}


int ccl_tracer_corr_legendre(ccl_cosmology *cosmo, int n_theta, double **theta,
		     CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel,
                     bool do_taper_cl,double *taper_cl_limits,double **corr_func,
		     double (*angular_cl)(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,
					  CCL_ClTracer *clt2, int * status) ){
  int L_max=n_theta; //15000;
  int status=0;
  int l_arr[L_max];
  double cl_arr[L_max];
  //n_theta=100;
  l_arr[0]=0;cl_arr[0]=0;
  for(int i=1;i<L_max;i+=1) {
     l_arr[i]=i;
    cl_arr[i]=angular_cl(cosmo,l_arr[i],ct1,ct2,&status);
  }

  if (do_taper_cl)
    status=taper_cl(L_max,l_arr,cl_arr, taper_cl_limits);

  //double *theta2;//why is theta and corr_func double pointer, **theta ??
  *theta=ccl_log_spacing(0.01*M_PI/180.,10*M_PI/180.,n_theta);
  *corr_func=(double *)malloc(sizeof(double)*n_theta);

  double **Pl_theta;
  Pl_theta=(double **)malloc( n_theta*sizeof( double*));
  for (int i=0;i<n_theta;i++)
      Pl_theta[i]=(double *)malloc(sizeof(double)*L_max);
 
  status=compute_legedre_polynomial(ct1,ct2,i_bessel,theta,n_theta,L_max,Pl_theta);

  for (int i=0;i<n_theta;i++){
    (*corr_func)[i]=0;
    for(int i_L=1;i_L<L_max;i_L+=1) {
      (*corr_func)[i]+=cl_arr[i_L]*Pl_theta[i][i_L];
    }
    (*corr_func)[i]/=(M_PI*4);
  }
  return 0;
}



/*--------ROUTINE: ccl_single_tracer_corr ------
TASK: Wrap bin_func and tracer_corr to get the correlation function at a single point
INPUT: type of tracer, number of theta values to evaluate = NL, theta vector
 */

double ccl_single_tracer_corr(double theta_in,ccl_cosmology *cosmo,CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel)
{

  double *theta,corr_func_out,*corr_func;
  int n_theta=NL;
  double taper_cl_limits[4]={1,2,10000,15000};
  
  ccl_tracer_corr_legendre(cosmo, n_theta,&theta,ct1,ct2,i_bessel,true,
			   taper_cl_limits,&corr_func,ccl_angular_cl);
  //ccl_tracer_corr_fftlog(cosmo, n_theta,&theta,ct1,ct2,i_bessel,true,taper_cl_limits,
  //&corr_func,ccl_angular_cl);
  //Spline the correlation
  gsl_spline * corr_spline = gsl_spline_alloc(CORR_SPLINE_TYPE, n_theta);
  int status = gsl_spline_init(corr_spline, theta,corr_func,n_theta);
  status = gsl_spline_eval_e(corr_spline,theta_in, NULL,&corr_func_out);
  
  return corr_func_out;    
}
