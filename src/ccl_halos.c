#include "ccl_utils.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_sf_expint.h"
#include "ccl_background.h"
#include "ccl_power.h"
#include "ccl_massfunc.h"
#include "ccl_error.h"
#include "ccl_halomod.h"

//maths helper function for NFW profile
static double helper_fx(double x){
    double f;
    f = log(1.+x)-x/(1.+x);
    return f;
}
    
//cosmo: ccl cosmology object containing cosmological parameters
//c: halo concentration, needs to be consistent with halo size definition
//halomass: halo mass
//massdef_delta: mass definition, overdensity relative to matter density
//a: scale factor
//r: radius at which to calculate output
//returns mass density at given r
double ccl_halo_profile_nfw(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double r, int *status){

    //haloradius: halo radius for mass definition
    //rs: scale radius
    double haloradius, rs;
    
    haloradius = r_delta(cosmo, halomass, a, massdef_delta_m, status);
    rs = haloradius/c;
    
    //rho0: NFW density parameter
    double rho0;
    
    rho0 = halomass/(4.*M_PI*pow(rs,3)*helper_fx(c));
    
    //M_enc: mass enclosed in r
    double Menc;
    
    Menc = halomass*helper_fx(r/rs)/helper_fx(c);
    
    //rho_r: density at r
    double rho_r;
    
    rho_r = rho0/((r/rs)*pow(1.+r/rs,2));
    
    return rho_r;
}
    
//cosmo: ccl cosmology object containing cosmological parameters
//c: halo concentration, needs to be consistent with halo size definition
//halomass: halo mass
//massdef_delta: mass definition, overdensity relative to matter density
//a: scale factor
//rp: radius at which to calculate output
//returns surface mass density (integrated along line of sight) at given projected r
double ccl_projected_halo_profile_nfw(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double rp, int *status){
    
    //haloradius: halo radius for mass definition
    //rs: scale radius
    double haloradius, rs;
    
    haloradius = r_delta(cosmo, halomass, a, massdef_delta_m, status);
    rs = haloradius/c;
    
    //rho0: NFW density parameter
    double rho0;
    
    rho0 = halomass/(4.*M_PI*pow(rs,3)*helper_fx(c));
    
    double x;
    double sigma;
    
    x = rp/rs;
    
    if (x==1.){
        sigma = 2.*rs*rho0;
    }
    else {
        sigma = 2.*rs*rho0*(1.-2.*atanh(abs((1.-x)/(1.+x)))/sqrt(abs(1.-x*x)));
    }
    
    return sigma;
    
}


//cosmo: ccl cosmology object containing cosmological parameters
//c: halo concentration, needs to be consistent with halo size definition
//halomass: halo mass
//massdef_delta: mass definition, overdensity relative to matter density
//a: scale factor
//rp: radius at which to calculate output
//returns mass density at given r
double ccl_halo_profile_einasto(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double r, int *status){
    
    //haloradius: halo radius for mass definition
    //rs: scale radius
    double haloradius, rs;
    
    haloradius = r_delta(cosmo, halomass, a, massdef_delta_m, status);
    rs = haloradius/c; //assuming same formula for scale radius as in NFW(?)
    
    //nu: peak height, https://arxiv.org/pdf/1401.1216.pdf eqn1
    //alpha: Einasto parameter, https://arxiv.org/pdf/1401.1216.pdf eqn5
    double nu;
    double alpha;
    
    nu = 1.686/ccl_sigmaM(cosmo, halomass, a, status);
    alpha = 0.155 + 0.0095*nu*nu;
    
    //rhos: scale density
    double rhos;
    
    rhos = halomass/(4.*M_PI*pow(rs,3)*helper_fx(c)); //assuming same formula as rho0 in NFW(?)
    
    //rho_r: density at r
    double rho_r;
    
    rho_r = rhos*exp(-2.*(pow(r/rs,alpha)-1.)/alpha)
    
    return rho_r;

}
