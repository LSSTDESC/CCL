#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_expint.h>
#include "ccl.h"

static double einasto_norm_integrand(double x, void *params)
{
  double alpha = *((double *)(params));
  return x*x*exp(-2*(pow(x,alpha)-1)/alpha);
}

void ccl_einasto_norm_integral(int n_m, double *r_s, double *r_delta, double *alpha,
			       double *norm_out,int *status)
{
#pragma omp parallel default(none)			\
  shared(n_m, r_s, r_delta, alpha, norm_out, status)
  {
    int ii;
    gsl_function F;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    
    if (w == NULL)
      *status = CCL_ERROR_MEMORY;
    
    if(*status == 0) {
      for(ii=0;ii<n_m;ii++) {
	int qagstatus;
	double result, eresult;
	double x_max = r_delta[ii]/r_s[ii];
	F.function = &einasto_norm_integrand;
	F.params = &(alpha[ii]);
	qagstatus = gsl_integration_qag(&F, 0, x_max, 0, 1E-4,
					1000, GSL_INTEG_GAUSS31,
					w, &result, &eresult);
	if(qagstatus != GSL_SUCCESS) {
	  ccl_raise_gsl_warning(qagstatus, "ccl_haloprofile.c: ccl_einasto_norm_integral():");
	  *status = CCL_ERROR_INTEG;
	  result = NAN;
	  break;
	}
	norm_out[ii] = 4 * M_PI * r_s[ii] * r_s[ii] * r_s[ii] * result;
      }
    } //end omp for
  
    gsl_integration_workspace_free(w);
  } //end omp parallel
}

static double hernquist_norm_integrand(double x, void *params)
{
  double opx=1+x;
  return x*x/(x*opx*opx*opx);
}

void ccl_hernquist_norm_integral(int n_m, double *r_s, double *r_delta,
			       double *norm_out,int *status)
{
#pragma omp parallel default(none)		\
  shared(n_m, r_s, r_delta, norm_out, status)
  {
    int ii;
    gsl_function F;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    
    if (w == NULL)
      *status = CCL_ERROR_MEMORY;
    
    if(*status == 0) {
      for(ii=0;ii<n_m;ii++) {
	int qagstatus;
	double result, eresult;
	double x_max = r_delta[ii]/r_s[ii];
	F.function = &hernquist_norm_integrand;
	F.params = NULL;
	qagstatus = gsl_integration_qag(&F, 0, x_max, 0, 1E-4,
					1000, GSL_INTEG_GAUSS31,
					w, &result, &eresult);
	if(qagstatus != GSL_SUCCESS) {
	  ccl_raise_gsl_warning(qagstatus, "ccl_haloprofile.c: ccl_hernquist_norm_integral():");
	  *status = CCL_ERROR_INTEG;
	  result = NAN;
	  break;
	}
	norm_out[ii] = 4 * M_PI * r_s[ii] * r_s[ii] * r_s[ii] * result;
      }
    } //end omp for
  
    gsl_integration_workspace_free(w);
  } //end omp parallel
}

//cosmo: ccl cosmology object containing cosmological parameters
//c: halo concentration, needs to be consistent with halo size definition
//halomass: halo mass
//massdef_delta: mass definition, overdensity relative to mean matter density
//a: scale factor
//rp: radius at which to calculate output
//nr: number of radii for calculation
//sigma_r: stores surface mass density (integrated along line of sight) at given projected rp
//returns void
void ccl_projected_halo_profile_nfw(ccl_cosmology *cosmo, double c,
                                    double halomass, double massdef_delta_m,
                                    double a, double *rp, int nr, double *sigma_r,
                                    int *status) {

    //haloradius: halo radius for mass definition
    //rs: scale radius
    double haloradius, rs;

    haloradius = r_delta(cosmo, halomass, a, massdef_delta_m, status);
    rs = haloradius/c;

    //rhos: NFW density parameter
    double rhos;

    rhos = halomass/(4.*M_PI*(rs*rs*rs)*(log(1.+c)-c/(1.+c)));

    double x;

    int i;
    for(i=0; i < nr; i++){

        x = rp[i]/rs;
        if (x==1.){
            sigma_r[i] = 2.*rs*rhos/3.;
        }
        else if (x<1.){
            sigma_r[i] = 2.*rs*rhos*(1.-2.*atanh(sqrt(fabs((1.-x)/(1.+x))))/sqrt(fabs(1.-x*x)))/(x*x-1.);
        }
        else {
            sigma_r[i] = 2.*rs*rhos*(1.-2.*atan(sqrt(fabs((1.-x)/(1.+x))))/sqrt(fabs(1.-x*x)))/(x*x-1.);
        }
    }
    return;

}
