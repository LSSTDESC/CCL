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

//maths helper function for NFW profile
static double helper_fx(double x){
    double f;
    f = log(1.+x)-x/(1.+x);
    return f;
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

// Structure to hold parameters of integrand_hernquist
typedef struct{
  ccl_cosmology *cosmo;
  double rs;
} Int_Hernquist_Par;

static double integrand_hernquist(double r, void *params){

    Int_Hernquist_Par *p = (Int_Hernquist_Par *)params;
    double rs = p->rs;

    return 4.*M_PI*r*r/((r/rs)*(1.+r/rs)*(1.+r/rs)*(1.+r/rs));
}

//integrate hernquist profile to get normalization of rhos
static double integrate_hernquist(ccl_cosmology *cosmo, double R, double rs, int *status){
    int qagstatus;
    double result = 0, eresult;
    Int_Hernquist_Par ipar;
    gsl_function F;
    gsl_integration_workspace *w = NULL;

    w = gsl_integration_workspace_alloc(1000);

    if (w == NULL) {
      *status = CCL_ERROR_MEMORY;
    }

    if (*status == 0) {
      // Structure required for the gsl integration
      ipar.cosmo = cosmo;
      ipar.rs = rs;
      F.function = &integrand_hernquist;
      F.params = &ipar;

      // Actually does the integration
      qagstatus = gsl_integration_qag(
        &F, 0, R, 0, 0.0001, 1000, 3, w,
        &result, &eresult);

      // Check for errors
      if (qagstatus != GSL_SUCCESS) {
        ccl_raise_gsl_warning(qagstatus, "ccl_haloprofile.c: integrate_hernquist():");
        *status = CCL_ERROR_PROFILE_INT;
        ccl_cosmology_set_status_message(cosmo, "ccl_haloprofile.c: integrate_hernquist(): Integration failure\n");
        result = NAN;
      }
    }

    // Clean up
    gsl_integration_workspace_free(w);

    return result;
}


//cosmo: ccl cosmology object containing cosmological parameters
//c: halo concentration, needs to be consistent with halo size definition
//halomass: halo mass
//massdef_delta: mass definition, overdensity relative to mean matter density
//a: scale factor
//r: radii at which to calculate output
//nr: number of radii for calculation
//rho_r: stores densities at r
//returns void
void ccl_halo_profile_hernquist(ccl_cosmology *cosmo, double c, double halomass,
                                double massdef_delta_m, double a, double *r, int nr,
                                double *rho_r, int *status) {

    //haloradius: halo radius for mass definition
    //rs: scale radius
    double haloradius, rs;

    haloradius = r_delta(cosmo, halomass, a, massdef_delta_m, status);
    rs = haloradius/c;

    //rhos: scale density
    double rhos;

    rhos = halomass/integrate_hernquist(cosmo, haloradius, rs, status); //normalize

    int i;
    double x;
    for(i=0; i < nr; i++) {
        x = r[i]/rs;
        rho_r[i] = rhos/(x*pow((1.+x),3));
    }

    return;

}
