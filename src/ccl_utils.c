#include "ccl_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ccl_params.h"
#include "ccl_error.h"
#include <gsl/gsl_errno.h>

/* ------- ROUTINE: ccl_linear spacing ------
INPUTS: [xmin,xmax] of the interval to be divided in N bins
OUTPUT: bin edges in range [xmin,xmax]
*/

double * ccl_linear_spacing(double xmin, double xmax, int N)
{
  double dx = (xmax-xmin)/(N -1.);
  
  double * x = malloc(sizeof(double)*N);
  if (x==NULL) {
    fprintf(stderr, "ERROR: Could not allocate memory for linear-spaced array (N=%d)\n", N);
    return x;
  }
  
  for (int i=0; i<N; i++) {
    x[i] = xmin + dx*i;
  }
  x[0]=xmin; //Make sure roundoff errors don't spoil edges
  x[N-1]=xmax; //Make sure roundoff errors don't spoil edges
  
  return x;
}

/* ------- ROUTINE: ccl_linlog spacing ------
 * INPUTS: [xminlog,xmax] of the interval to be divided in bins
 *         xmin when linear spacing starts
 *         Nlog number of logarithmically spaced bins 
 *         Nlin number of linearly spaced bins 
 * OUTPUT: bin edges in range [xminlog,xmax]
 * */

double * ccl_linlog_spacing(double xminlog, double xmin, double xmax, int Nlog, int Nlin)
{
  if (Nlog<2) {
    fprintf(stderr, "ERROR: Cannot make log-spaced array with %d points - need at least 2\n", Nlog);
    return NULL;
  }

  if (!(xminlog>0 && xmin>0)) {
    fprintf(stderr, "ERROR: Cannot make log-spaced array xminlog or xmin  non-positive (had %le, %le)\n", xminlog, xmin);
    return NULL;
  }

  if (xminlog>xmin){
    fprintf(stderr, "ERROR: xminlog must be smaller as xmin");
    return NULL;
  }

  if (xmin>xmax){
    fprintf(stderr, "ERROR: xmin must be smaller as xmax");
    return NULL;
  }

  double * x = malloc(sizeof(double)*(Nlin+Nlog-1));
  if (x==NULL) {
    fprintf(stderr, "ERROR: Could not allocate memory for array of size (Nlin+Nlog-1)=%d)\n", (Nlin+Nlog-1));
    return x;
  }

  double dx = (xmax-xmin)/(Nlin -1.);
  double log_xchange = log(xmin);
  double log_xmin = log(xminlog);
  double dlog_x = (log_xchange - log_xmin) /  (Nlog-1.);

  for (int i=0; i<Nlin+Nlog-1; i++) {
    if (i<Nlog)
        x[i] = exp(log_xmin + dlog_x*i);
    if (i>=Nlog)
        x[i] = xmin + dx*(i-Nlog+1);
  }

  x[0]=xminlog; //Make sure roundoff errors don't spoil edges
  x[Nlog-1]=xmin; //Make sure roundoff errors don't spoil edges
  x[Nlin+Nlog-2]=xmax; //Make sure roundoff errors don't spoil edges
  
  return x;
}

/* ------- ROUTINE: ccl_log spacing ------
INPUTS: [xmin,xmax] of the interval to be divided logarithmically in N bins
TASK: divide an interval in N logarithmic bins
OUTPUT: bin edges in range [xmin,xmax]
*/

double * ccl_log_spacing(double xmin, double xmax, int N)
{
  if (N<2) {
    fprintf(stderr, "ERROR: Cannot make log-spaced array with %d points - need at least 2\n", N);
    return NULL;
  }
  
  if (!(xmin>0 && xmax>0)) {
    fprintf(stderr, "ERROR: Cannot make log-spaced array xmax or xmax non-positive (had %le, %le)\n", xmin, xmax);
    return NULL;
  }
  
  double log_xmax = log(xmax);
  double log_xmin = log(xmin);
  double dlog_x = (log_xmax - log_xmin) /  (N-1.);
  
  double * x = malloc(sizeof(double)*N);
  if (x==NULL) {
    fprintf(stderr, "ERROR: Could not allocate memory for log-spaced array (N=%d)\n", N);
    return x;
  }
  
  for (int i=0; i<N; i++) {
    x[i] = exp(log_xmin + dlog_x*i);
  }
  x[0]=xmin; //Make sure roundoff errors don't spoil edges
  x[N-1]=xmax; //Make sure roundoff errors don't spoil edges
  
  return x;
}


//Spline creator
//n     -> number of points
//x     -> x-axis
//y     -> f(x)-axis
//y0,yf -> values of f(x) to use beyond the interpolation range
SplPar *ccl_spline_init(int n,double *x,double *y,double y0,double yf)
{
  SplPar *spl=malloc(sizeof(SplPar));
  if(spl==NULL)
    return NULL;
  
  spl->intacc=gsl_interp_accel_alloc();
  spl->spline=gsl_spline_alloc(gsl_interp_cspline,n);
  int parstatus=gsl_spline_init(spl->spline,x,y,n);
  if(parstatus) {
    gsl_interp_accel_free(spl->intacc);
    gsl_spline_free(spl->spline);
    return NULL;
  }

  spl->x0=x[0];
  spl->xf=x[n-1];
  spl->y0=y0;
  spl->yf=yf;

  return spl;
}

//Evaluates spline at x checking for bound errors
double ccl_spline_eval(double x,SplPar *spl)
{
  if(x<=spl->x0)
    return spl->y0;
  else if(x>=spl->xf) 
    return spl->yf;
  else {
    double y;
    int stat=gsl_spline_eval_e(spl->spline,x,spl->intacc,&y);
    if (stat!=GSL_SUCCESS) {
      ccl_raise_gsl_warning(stat, "ccl_utils.c: ccl_splin_eval():");
      return NAN;
    }
    return y;
  }
}

#define CCL_GAMMA1 2.6789385347077476336556 //Gamma(1/3)
#define CCL_GAMMA2 1.3541179394264004169452 //Gamma(2/3)
#define CCL_ROOTPI12 21.269446210866192327578 //12*sqrt(pi)
double ccl_j_bessel(int l,double x)
{
  double jl;
  double ax=fabs(x);
  double ax2=x*x;
  if(l<0) {
    fprintf(stderr,"CosmoMas: l>0 for Bessel function");
    exit(1);
  }

  if(l<7) {
    if(l==0) {
      if(ax<0.1) jl=1-ax2*(1-ax2/20.)/6.;
      else jl=sin(x)/x;
    }
    else if(l==1) {
      if(ax<0.2) jl=ax*(1-ax2*(1-ax2/28)/10)/3;
      else jl=(sin(x)/ax-cos(x))/ax;
    }
    else if(l==2) {
      if(ax<0.3) jl=ax2*(1-ax2*(1-ax2/36)/14)/15;
      else jl=(-3*cos(x)/ax-sin(x)*(1-3/ax2))/ax;
    }
    else if(l==3) {
      if(ax<0.4)
	jl=ax*ax2*(1-ax2*(1-ax2/44)/18)/105;
      else 
	jl=(cos(x)*(1-15/ax2)-sin(x)*(6-15/ax2)/ax)/ax;
    }
    else if(l==4) {
      if(ax<0.6)
	jl=ax2*ax2*(1-ax2*(1-ax2/52)/22)/945;
      else
	jl=(sin(x)*(1-(45-105/ax2)/ax2)+cos(x)*(10-105/ax2)/ax)/ax;
    }
    else if(l==5) {
      if(ax<1.0)
	jl=ax2*ax2*ax*(1-ax2*(1-ax2/60)/26)/10395;
      else {
	jl=(sin(x)*(15-(420-945/ax2)/ax2)/ax-
	    cos(x)*(1-(105-945/ax2)/ax2))/ax;
      }
    }
    else {
      if(ax<1.0)
	jl=ax2*ax2*ax2*(1-ax2*(1-ax2/68)/30)/135135;
      else {
	jl=(sin(x)*(-1+(210-(4725-10395/ax2)/ax2)/ax2)+
	    cos(x)*(-21+(1260-10395/ax2)/ax2)/ax)/ax;
      }
    }
  }
  else {
    double nu=l+0.5;
    double nu2=nu*nu;
    
    if(ax<1.0E-40) jl=0;
    else if((ax2/l)<0.5) {
      jl=(exp(l*log(ax/nu)-M_LN2+nu*(1-M_LN2)-(1-(1-3.5/nu2)/(30*nu2))/(12*nu))/nu)*
	(1-ax2/(4*nu+4)*(1-ax2/(8*nu+16)*(1-ax2/(12*nu+36))));
    }
    else if((l*l/ax)<0.5) {
      double beta=ax-0.5*M_PI*(l+1);
      jl=(cos(beta)*(1-(nu2-0.25)*(nu2-2.25)/(8*ax2)*(1-(nu2-6.25)*(nu2-12.25)/(48*ax2)))-
	  sin(beta)*(nu2-0.25)/(2*ax)*(1-(nu2-2.25)*(nu2-6.25)/(24*ax2)*
				       (1-(nu2-12.25)*(nu2-20.25)/(80*ax2))))/ax;
    }
    else {
      double l3=pow(nu,0.325);
      if(ax<nu-1.31*l3) {
	double cosb=nu/ax;
	double sx=sqrt(nu2-ax2);
	double cotb=nu/sx;
	double secb=ax/nu;
	double beta=log(cosb+sx/ax);
	double cot3b=cotb*cotb*cotb;
	double cot6b=cot3b*cot3b;
	double sec2b=secb*secb;
	double expterm=((2+3*sec2b)*cot3b/24
			-((4+sec2b)*sec2b*cot6b/16
			  +((16-(1512+(3654+375*sec2b)*sec2b)*sec2b)*cot3b/5760
			    +(32+(288+(232+13*sec2b)*sec2b)*sec2b)*sec2b*cot6b/(128*nu))*
			  cot6b/nu)/nu)/nu;
	jl=sqrt(cotb*cosb)/(2*nu)*exp(-nu*beta+nu/cotb-expterm);
      }
      else if(ax>nu+1.48*l3) {
	double cosb=nu/ax;
	double sx=sqrt(ax2-nu2);
	double cotb=nu/sx;
	double secb=ax/nu;
	double beta=acos(cosb);
	double cot3b=cotb*cotb*cotb;
	double cot6b=cot3b*cot3b;
	double sec2b=secb*secb;
	double trigarg=nu/cotb-nu*beta-0.25*M_PI-
	  ((2+3*sec2b)*cot3b/24+(16-(1512+(3654+375*sec2b)*sec2b)*sec2b)*
	   cot3b*cot6b/(5760*nu2))/nu;
	double expterm=((4+sec2b)*sec2b*cot6b/16-
			(32+(288+(232+13*sec2b)*sec2b)*sec2b)*
			sec2b*cot6b*cot6b/(128*nu2))/nu2;
	jl=sqrt(cotb*cosb)/nu*exp(-expterm)*cos(trigarg);
      }
      else {
	double beta=ax-nu;
	double beta2=beta*beta;
	double sx=6/ax;
	double sx2=sx*sx;
	double secb=pow(sx,0.3333333333333333333333);
	double sec2b=secb*secb;
	
	jl=(CCL_GAMMA1*secb+beta*CCL_GAMMA2*sec2b
	    -(beta2/18-1.0/45.0)*beta*sx*secb*CCL_GAMMA1
	    -((beta2-1)*beta2/36+1.0/420.0)*sx*sec2b*CCL_GAMMA2
	    +(((beta2/1620-7.0/3240.0)*beta2+1.0/648.0)*beta2-1.0/8100.0)*sx2*secb*CCL_GAMMA1
	    +(((beta2/4536-1.0/810.0)*beta2+19.0/11340.0)*beta2-13.0/28350.0)*beta*sx2*sec2b*CCL_GAMMA2
	    -((((beta2/349920-1.0/29160.0)*beta2+71.0/583200.0)*beta2-121.0/874800.0)*
	      beta2+7939.0/224532000.0)*beta*sx2*sx*secb*CCL_GAMMA1)*sqrt(sx)/CCL_ROOTPI12;
      }
    }
  }
  if((x<0)&&(l%2!=0)) jl=-jl;
  
  return jl;
}

//Spline destructor
void ccl_spline_free(SplPar *spl)
{
  gsl_spline_free(spl->spline);
  gsl_interp_accel_free(spl->intacc);
  free(spl);
}
