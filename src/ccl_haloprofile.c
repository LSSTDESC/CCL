#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_expint.h>
#include "ccl.h"

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
//r: radii at which to calculate output
//nr: number of radii for calculation
//rho_r: stores densities at r
//returns void
void ccl_halo_profile_nfw(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double *r, int nr, double *rho_r, int *status){

    //haloradius: halo radius for mass definition
    //rs: scale radius
    double haloradius, rs;

    haloradius = r_delta(cosmo, halomass, a, massdef_delta_m, status);
    rs = haloradius/c;

    //rhos: NFW density parameter
    double rhos;

    rhos = halomass/(4.*M_PI*(rs*rs*rs)*helper_fx(c));

    int i;
    double x;
    for(i=0; i < nr; i++) {
        x = r[i]/rs;
        rho_r[i] = rhos/(x*(1.+x)*(1.+x));
    }
    return;
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
void ccl_projected_halo_profile_nfw(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double *rp, int nr, double *sigma_r, int *status){

    //haloradius: halo radius for mass definition
    //rs: scale radius
    double haloradius, rs;

    haloradius = r_delta(cosmo, halomass, a, massdef_delta_m, status);
    rs = haloradius/c;

    //rhos: NFW density parameter
    double rhos;

    rhos = halomass/(4.*M_PI*(rs*rs*rs)*helper_fx(c));

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

//maths helper function assuming NFW approximation
static double helper_solve_cvir(double c, void *rhs_pointer){
    double rhs = *(double*)rhs_pointer;
    return (log(1.+c)-c/(1.+c))/(c*c*c) - rhs;
}

//solve for cvir from different mass definition iteratively, assuming NFW profile.
static double solve_cvir(ccl_cosmology *cosmo, double rhs, double initial_guess, int *status){

    int rootstatus;
    int iter = 0, max_iter = 100;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double cvir;
    double c_lo = 0.1, c_hi = initial_guess*100.;
    gsl_function F;

    F.function = &helper_solve_cvir;
    F.params = &rhs;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc (T);
    gsl_root_fsolver_set (s, &F, c_lo, c_hi);

    do
      {
        iter++;
        gsl_root_fsolver_iterate (s);
        cvir = gsl_root_fsolver_root (s);
        c_lo = gsl_root_fsolver_x_lower (s);
        c_hi = gsl_root_fsolver_x_upper (s);
        rootstatus = gsl_root_test_interval (c_lo, c_hi, 0, 0.0001);
      }
    while (rootstatus == GSL_CONTINUE && iter < max_iter);

    gsl_root_fsolver_free (s);


    // Check for errors
    if (rootstatus != GSL_SUCCESS) {
        ccl_raise_gsl_warning(rootstatus, "ccl_haloprofile.c: solve_cvir():");
        *status = CCL_ERROR_PROFILE_ROOT;
        ccl_cosmology_set_status_message(cosmo, "ccl_haloprofile.c: solve_cvir(): Root finding failure\n");
        return NAN;
    } else {
        return cvir;
    }
}

// Structure to hold parameters of integrand_einasto
typedef struct{
  ccl_cosmology *cosmo;
  double alpha, rs;
} Int_Einasto_Par;

static double integrand_einasto(double r, void *params){

    Int_Einasto_Par *p = (Int_Einasto_Par *)params;
    double alpha = p->alpha;
    double rs = p->rs;

    return 4.*M_PI*r*r*exp(-2.*(pow(r/rs,alpha)-1)/alpha);
}

//integrate einasto profile to get normalization of rhos
static double integrate_einasto(ccl_cosmology *cosmo, double R, double alpha, double rs, int *status){
    int qagstatus;
    double result = 0, eresult;
    Int_Einasto_Par ipar;
    gsl_function F;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);

    // Structure required for the gsl integration
    ipar.cosmo = cosmo;
    ipar.alpha = alpha;
    ipar.rs = rs;
    F.function = &integrand_einasto;
    F.params = &ipar;

    // Actually does the integration
    qagstatus = gsl_integration_qag(
      &F, 0, R, 0, 0.0001, 1000, 3, w,
      &result, &eresult);

    // Clean up
    gsl_integration_workspace_free(w);

    // Check for errors
    if (qagstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(qagstatus, "ccl_haloprofile.c: integrate_einasto():");
      *status = CCL_ERROR_PROFILE_INT;
      ccl_cosmology_set_status_message(cosmo, "ccl_haloprofile.c: integrate_einasto(): Integration failure\n");
      return NAN;
    } else {
      return result;
    }
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
void ccl_halo_profile_einasto(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double *r, int nr, double *rho_r, int *status){

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

    nu = 1.686/ccl_sigmaM(cosmo, Mvir, a, status); //delta_c_Tinker
    alpha = 0.155 + 0.0095*nu*nu;

    //rhos: scale density
    double rhos;

    rhos = halomass/integrate_einasto(cosmo, haloradius, alpha, rs, status); //normalize

    int i;
    for(i=0; i < nr; i++) {
        rho_r[i] = rhos*exp(-2.*(pow(r[i]/rs,alpha)-1.)/alpha);
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
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);

    // Structure required for the gsl integration
    ipar.cosmo = cosmo;
    ipar.rs = rs;
    F.function = &integrand_hernquist;
    F.params = &ipar;

    // Actually does the integration
    qagstatus = gsl_integration_qag(
      &F, 0, R, 0, 0.0001, 1000, 3, w,
      &result, &eresult);

    // Clean up
    gsl_integration_workspace_free(w);

    // Check for errors
    if (qagstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(qagstatus, "ccl_haloprofile.c: integrate_hernquist():");
      *status = CCL_ERROR_PROFILE_INT;
      ccl_cosmology_set_status_message(cosmo, "ccl_haloprofile.c: integrate_hernquist(): Integration failure\n");
      return NAN;
    } else {
      return result;
    }
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
void ccl_halo_profile_hernquist(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double *r, int nr, double *rho_r, int *status){

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
