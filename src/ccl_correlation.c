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

typedef struct
{
  SplPar *cl; 
  double theta;
  int i_bessel;
}corr_int_par;

//params: cls, theta, i_bessel, *l*
static double corr_integrand(double l, void *params)
{
  corr_int_par *p=(corr_int_par *) params;
  double bessel_j=gsl_sf_bessel_Jn(p->i_bessel,p->theta*l);
  return l*bessel_j*gsl_spline_eval(p->cl,l,NULL);
}
// Check Spline limits and make sure it is not evaluated outside limits.
void get_corr(SplPar *cl, double *theta,double *corr_func, int n_theta,int l_min,int l_max, int i_bessel)
{//n_theta=length of theta and corr_func.. they must match
  gsl_function F;
  gsl_integration_workspace *w=gsl_integration_workspace_alloc(GSL_INTEGRATION_LIMIT);
  corr_int_par cp;
  cp.i_bessel=i_bessel;
  cp.cl=cl;
  double result,eresult;
  for (int i=0;i<n_theta;i++)
    {
      cp.theta=theta[i];
      F.function=&corr_integrand;
      F.params=&cp;
      gsl_integration_qag(&F,l_min,l_max,0,EPSREL_CORR_FUNC,GSL_INTEGRATION_LIMIT,GSL_INTEG_GAUSS41,w,&result,&eresult);
      corr_func[i]=result/(2*M_PI);
    }
  return;
}
