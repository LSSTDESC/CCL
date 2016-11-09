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

/*--------ROUTINE: ccl_tracer_corr ------
TASK: For a given tracer, get the correlation function
INPUT: type of tracer, number of theta values to evaluate, theta vector
TODO: Check Cl spline limits and make sure it is not evaluated outside limits.
 */
int ccl_tracer_corr(ccl_cosmology *cosmo, int n_theta, double *theta, CCL_ClTracer *ct1, CCL_ClTracer *ct2, int i_bessel,double *corr_func){

  if((ct1->tracer_type==CL_TRACER_WL) && (ct2->tracer_type==CL_TRACER_WL)){
    if((i_bessel!=0) && (i_bessel!=4)) return 1;
  }
  if((ct1->tracer_type==CL_TRACER_NC) && (ct2->tracer_type==CL_TRACER_NC)){
    if(i_bessel!=0) return 1;
  }
  if(((ct1->tracer_type==CL_TRACER_WL) && (ct2->tracer_type==CL_TRACER_NC)) || ((ct1->tracer_type==CL_TRACER_NC) && (ct2->tracer_type==CL_TRACER_WL))){
    if(i_bessel!=2) return 1;
  }  

  double *l_arr,cl_arr[NL];

  l_arr=ccl_log_spacing(L_MIN_INT,L_MAX_INT,NL);
  for(int i=0;i<NL;i+=1) {
    cl_arr[i]=ccl_angular_cl(cosmo,l_arr[i],ct1,ct2); 
  }

  //Now spline
  gsl_spline * clspl = gsl_spline_alloc(L_SPLINE_TYPE, NL);
  int status = gsl_spline_init(clspl, l_arr, cl_arr, NL);
  if (status){
    gsl_spline_free(clspl);
    cosmo->status = CCL_ERROR_SPLINE;
    strcpy(cosmo->status_message,"ccl_power.c: ccl_cosmology_compute_power_class(): Error creating Cl spline\n");
  } else {
    //And compute correlation function
    //ccl_general_corr(clspl,theta,corr_func,n_theta,i_bessel);
    //gsl_spline_free(clspl);
  }
 
  fftlog_ComputeXiLM(i_bessel , 2 , NL , l_arr, cl_arr, theta,corr_func);

  return status;

}

