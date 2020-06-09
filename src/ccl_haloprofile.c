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
    int status_this=0;
    gsl_function F;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    
    if (w == NULL)
      status_this = CCL_ERROR_MEMORY;
    
    if(status_this == 0) {
#pragma omp for
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
	  status_this = CCL_ERROR_INTEG;
	  result = NAN;
	}
	norm_out[ii] = 4 * M_PI * r_s[ii] * r_s[ii] * r_s[ii] * result;
      }
    } //end omp for
  
    gsl_integration_workspace_free(w);
    if(status_this) {
      #pragma omp atomic write
      *status = status_this;
    }
  } //end omp parallel
}

<<<<<<< HEAD

//cosmo: ccl cosmology object containing cosmological parameters
//c: halo concentration, needs to be consistent with halo size definition
//halomass: halo mass
//massdef_delta: mass definition, overdensity relative to mean matter density
//a: scale factor
//r: radii at which to calculate output
//nr: number of radii for calculation
//rho_r: stores densities at r
//returns void
void ccl_halo_profile_einasto(ccl_cosmology *cosmo, double c, double halomass,
                              double massdef_delta_m, double a, double *r, int nr,
                              double *rho_r, int *status) {

    //haloradius: halo radius for mass definition
    //rs: scale radius
    double haloradius, rs;

    haloradius = r_delta(cosmo, halomass, a, massdef_delta_m, status);
    rs = haloradius/c;


    //nu: peak height, https://arxiv.org/pdf/1401.1216.pdf eqn1
    //alpha: Einasto parameter, https://arxiv.org/pdf/1401.1216.pdf eqn5
    double nu;
    double alpha;  //calibrated relation with nu, with virial mass
    double Mvir;
    double Delta_v;

    Delta_v = Dv_BryanNorman(cosmo, a, status); //virial definition, if odelta is this, the definition is virial.

    if (massdef_delta_m<Delta_v+1 && massdef_delta_m>Delta_v-1){   //allow rounding of virial definition
        Mvir = halomass;
    }
    else{
        double rhs; //NFW equation for cvir: f(Rvir/Rs)/f(c)=(Rvir^3*Delta_v)/(R^3*massdef) -> f(cvir)/cvir^3 = rhs
        rhs = (helper_fx(c)*(rs*rs*rs)*Delta_v)/((haloradius*haloradius*haloradius)*massdef_delta_m);
        Mvir = halomass*helper_fx(solve_cvir(cosmo, rhs, c, status))/helper_fx(c);
    }

    nu = 1.686/ccl_sigmaM(cosmo, log10(Mvir), a, status); //delta_c_Tinker
    alpha = 0.155 + 0.0095*nu*nu;

    //rhos: scale density
    double rhos;

    rhos = halomass/integrate_einasto(cosmo, haloradius, alpha, rs, status); //normalize

    int i;
    for(i=0; i < nr; i++) {
        rho_r[i] = rhos*exp(-2.*(pow(r[i]/rs,alpha)-1.)/alpha);
    }

    return;

=======
static double hernquist_norm_integrand(double x, void *params)
{
  double opx=1+x;
  return x*x/(x*opx*opx*opx);
>>>>>>> master
}

void ccl_hernquist_norm_integral(int n_m, double *r_s, double *r_delta,
			       double *norm_out,int *status)
{
#pragma omp parallel default(none)		\
  shared(n_m, r_s, r_delta, norm_out, status)
  {
    int ii;
    int status_this=0;
    gsl_function F;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    
    if (w == NULL)
      status_this = CCL_ERROR_MEMORY;
    
    if(status_this == 0) {
#pragma omp for
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
	  status_this = CCL_ERROR_INTEG;
	  result = NAN;
	}
	norm_out[ii] = 4 * M_PI * r_s[ii] * r_s[ii] * r_s[ii] * result;
      }
    } //end omp for
  
    gsl_integration_workspace_free(w);
    if(status_this) {
      #pragma omp atomic write
      *status = status_this;
    }
  } //end omp parallel
}
