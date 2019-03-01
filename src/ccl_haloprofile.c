#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_expint.h>
#include "ccl.h"
#include "ccl_correlation.h"
#include "ccl_massfunc.h"
#include "ccl_halomod.h"
#include "ccl_haloprofile.h"


double r_delta(ccl_cosmology *cosmo, double halomass, double a, double odelta, int *status); //in Mpc comving. halomass in Msun, odelta integer.
double ccl_sigmaM(ccl_cosmology *cosmo, double halomass, double a, int *status); //returns sigma from the sigmaM interpolation.
double Dv_BryanNorman(ccl_cosmology *cosmo, double a, int *status); //Computes the virial collapse density contrast with respect to the matter density assuming LCDM.

//maths helper function for NFW profile
static double helper_fx(double x){
    double f;
    f = log(1.+x)-x/(1.+x);
    return f;
}


//take vectorized r.
//implement inner and outer separately and let user choose combining method.
//take flexible arguments.
//For h&w:
/*----- ROUTINE: ccl_halob1 -----
INPUT: ccl_cosmology * cosmo, double halo mass in units of Msun, double scale factor
TASK: returns dimensionless linear halo bias
*/
//double ccl_halo_bias(ccl_cosmology *cosmo, double halomass, double a, double odelta, int *status)
/*--------ROUTINE: ccl_correlation_3d ------
TASK: Calculate the 3d-correlation function. Do so by using FFTLog.

INPUT: cosmology, scale factor a,
       number of r values, r values,
       key for tapering, limits of tapering

Correlation function result will be in array xi
 */
//void ccl_correlation_3d(ccl_cosmology *cosmo, double a,
//			int n_r,double *r,double *xi,
//			int do_taper_pk,double *taper_pk_limits,
//			int *status)


//cosmo: ccl cosmology object containing cosmological parameters
//c: halo concentration, needs to be consistent with halo size definition
//halomass: halo mass
//massdef_delta: mass definition, overdensity relative to matter density
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

    rhos = halomass/(4.*M_PI*pow(rs,3)*helper_fx(c));

    //M_enc: mass enclosed in r
    //double Menc;

    //Menc = halomass*helper_fx(r/rs)/helper_fx(c);

    int i;
    for(i=0; i < nr; i++) {
        rho_r[i] = rhos/((r[i]/rs)*pow(1.+r[i]/rs,2));
    }
    return;
}

//cosmo: ccl cosmology object containing cosmological parameters
//c: halo concentration, needs to be consistent with halo size definition
//halomass: halo mass
//massdef_delta: mass definition, overdensity relative to matter density
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

    rhos = halomass/(4.*M_PI*pow(rs,3)*helper_fx(c));

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

//solve for cvir from different mass definition iteratively, assuming NFW profile.
static double solve_cvir(double rhs, double initial_guess){    //10^-5 accuracy
    double c_small, c_large;
    if (helper_fx(initial_guess)/pow(initial_guess,3)<rhs){
        c_large = initial_guess;
        c_small = c_large/2.;
        while (helper_fx(c_small)/pow(c_small,3)<rhs) {
            c_large = c_small;
            c_small /= 2.;
        }
    }
    else {
        c_small = initial_guess;
        c_large = c_small*2.;
        while (helper_fx(c_large)/pow(c_large,3)>rhs) {
            c_small = c_large;
            c_large *= 2.;
        }
    }
    while (c_large-c_small>0.00001) {
        if (helper_fx((c_small+c_large)/2.)/pow((c_small+c_large)/2.,3)<rhs) {
            c_large = (c_small+c_large)/2.;
        }
        else {
            c_small = (c_small+c_large)/2.;
        }
    }

    return c_small;
}

//integrate einasto profile to get normalization of rhos
static double integrate_einasto(double R, double alpha, double rs){
    double i;
    double integral;
    integral = 0;
    for (i = 1.; i < 1001.; i+=1.) {
        integral += 4.*M_PI*pow(R*i/1000.,2)*exp(-2.*(pow((R*i/(1000.*rs)),alpha)-1)/alpha)*R/1000.;
    }
    return integral;
}


//cosmo: ccl cosmology object containing cosmological parameters
//c: halo concentration, needs to be consistent with halo size definition
//halomass: halo mass
//massdef_delta: mass definition, overdensity relative to matter density
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
        rhs = (helper_fx(c)*pow(rs,3)*Delta_v)/(pow(haloradius,3)*massdef_delta_m);
        Mvir = halomass*helper_fx(solve_cvir(rhs, haloradius))/helper_fx(c);
    }


    nu = 1.686/ccl_sigmaM(cosmo, Mvir, a, status); //delta_c_Tinker
    alpha = 0.155 + 0.0095*nu*nu;

    //rhos: scale density
    double rhos;

    rhos = halomass/integrate_einasto(haloradius, alpha, rs); //normalize

    int i;
    for(i=0; i < nr; i++) {
        rho_r[i] = rhos*exp(-2.*(pow(r[i]/rs,alpha)-1.)/alpha);
    }

    return;

}


double integrate_hernquist(double R, double rs){
        double i;
        double integral;
        double r;
        integral = 0;
        for (i = 1.; i < 1001.; i+=1.) {
            r = R*i/1000.;
            integral += 4.*M_PI*pow(r,2)*(R/1000.)/((r/rs)*pow((1.+r/rs),3));
        }
        return integral;
}

//cosmo: ccl cosmology object containing cosmological parameters
//c: halo concentration, needs to be consistent with halo size definition
//halomass: halo mass
//massdef_delta: mass definition, overdensity relative to matter density
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

    rhos = halomass/integrate_hernquist(haloradius, rs); //normalize

    int i;
    for(i=0; i < nr; i++) {
        rho_r[i] = rhos/((r[i]/rs)*pow((1.+r[i]/rs),3));
    }

    return;

}


//cosmo: ccl cosmology object containing cosmological parameters
//c: halo concentration, needs to be consistent with halo size definition
//halomass: halo mass
//massdef_delta: mass definition, overdensity relative to matter density
//a: scale factor
//r: radius at which to calculate output
//returns mass density at given r
/*double ccl_halo_profile_diemer(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double r, int *status){}*/
