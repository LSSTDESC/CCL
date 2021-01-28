#include <math.h>
#include "ccl.h"

/*
 * Allocate a new struct for storing EH98 data
 * @param params Cosmological parameters
 * @param int, include BAO wiggles if not 0, smooth otherwise
 */
eh_struct* ccl_eh_struct_new(ccl_parameters *params, int wiggled) {
  //////
  // Computes Eisenstein & Hu parameters for
  // P_k and r_sound
  // see astro-ph/9709112 for the relevant equations
  double OMh2,OBh2;
  double th2p7;
  eh_struct *eh=malloc(sizeof(eh_struct));
  if(eh==NULL)
    return NULL;

  OMh2=params->Omega_m*params->h*params->h; //Cosmo params scaled by h^2
  OBh2=params->Omega_b*params->h*params->h;
  th2p7=params->T_CMB/2.7; //This is Theta_{2.7} in E&Hu notation
  eh->th2p7=th2p7; //This is Theta_{2.7} in E&Hu notation
  eh->zeq=2.5E4*OMh2/pow(th2p7,4); //Eq. 2
  eh->keq=0.0746*OMh2/(params->h*th2p7*th2p7); //Eq. 3

  //This group corresponds to Eq. 4
  double b1,b2;
  b1=0.313*pow(OMh2,-0.419)*(1+0.607*pow(OMh2,0.674));
  b2=0.238*pow(OMh2,0.223);
  eh->zdrag=1291*pow(OMh2,0.251)*(1+b1*pow(OBh2,b2))/(1+0.659*pow(OMh2,0.828));

  //These are the baryon-to-photon ratios
  //at equality (Req) and drag (Rd) epochs
  //Eq. 5
  double Req,Rd;
  Req=31.5*OBh2*1000./(eh->zeq*pow(th2p7,4));
  Rd=31.5*OBh2*1000./((1+eh->zdrag)*pow(th2p7,4));
  eh->rsound=2/(3*eh->keq)*sqrt(6/Req)*
    log((sqrt(1+Rd)+sqrt(Rd+Req))/(1+sqrt(Req)));

  //This is Eq. 7 (but in h/Mpc)
  eh->kSilk=1.6*pow(OBh2,0.52)*pow(OMh2,0.73)*(1+pow(10.4*OMh2,-0.95))/params->h;

  //These are Eqs. 11
  double a1,a2,b_frac;
  a1=pow(46.9*OMh2,0.670)*(1+pow(32.1*OMh2,-0.532));
  a2=pow(12.0*OMh2,0.424)*(1+pow(45.0*OMh2,-0.582));
  b_frac=OBh2/OMh2;
  eh->alphac=pow(a1,-b_frac)*pow(a2,-b_frac*b_frac*b_frac);

  //These are Eqs. 12
  double bb1,bb2;
  bb1=0.944/(1+pow(458*OMh2,-0.708));
  bb2=pow(0.395*OMh2,-0.0266);
  eh->betac=1/(1+bb1*(pow(1-b_frac,bb2)-1));

  double y=eh->zeq/(1+eh->zdrag);
  double sqy=sqrt(1+y);
  double gy=y*(-6*sqy+(2+3*y)*log((sqy+1)/(sqy-1))); //Eq 15
  //Baryon suppression Eq. 14
  eh->alphab=2.07*eh->keq*eh->rsound*pow(1+Rd,-0.75)*gy;

  //Baryon envelope shift Eq. 24
  eh->betab=0.5+b_frac+(3-2*b_frac)*sqrt(pow(17.2*OMh2,2)+1);

  //Node shift parameter Eq. 23
  eh->bnode=8.41*pow(OMh2,0.435);

  //Approx for the sound horizon, Eq. 26
  eh->rsound_approx=params->h*44.5*log(9.83/OMh2)/
    sqrt(1+10*pow(OBh2,0.75));

  eh->wiggled=wiggled;

  return eh;
}

static double tkEH_0(double keq,double k,double a,double b)
{
  //////
  // Eisentstein & Hu's Tk_0
  // see astro-ph/9709112 for the relevant equations
  double q=k/(13.41*keq); //Eq 10
  double c=14.2/a+386./(1+69.9*pow(q,1.08)); //Eq 20
  double l=log(M_E+1.8*b*q); //Change of var for Eq 19
  return l/(l+c*q*q); //Returns Eq 19
}

static double tkEH_c(eh_struct *eh,double k)
{
  //////
  // Eisenstein & Hu's Tk_c
  // see astro-ph/9709112 for the relevant equations
  double f=1/(1+pow(k*eh->rsound/5.4,4)); //Eq 18
  return f*tkEH_0(eh->keq,k,1,eh->betac)+
    (1-f)*tkEH_0(eh->keq,k,eh->alphac,eh->betac); //Returns Eq 17
}

static double jbes0(double x)
{
  double jl;
  double ax2=x*x;

  if(ax2<1e-4) jl=1-ax2*(1-ax2/20.)/6.;
  else jl=sin(x)/x;

  return jl;
}

static double tkEH_b(eh_struct *eh,double k)
{
  //////
  // Eisenstein & Hu's Tk_b (Eq 21)
  // see astro-ph/9709112 for the relevant equations
  double x_bessel,part1,part2;
  double x=k*eh->rsound;

  //First term of Eq. 21
  if(k==0) x_bessel=0;
  else {
    x_bessel=x*pow(1+eh->bnode*eh->bnode*eh->bnode/(x*x*x),-1./3.);
  }

  part1=tkEH_0(eh->keq,k,1,1)/(1+pow(x/5.2,2));

  //Second term of Eq. 21
  if(k==0)
    part2=0;
  else
    part2=eh->alphab/(1+pow(eh->betab/x,3))*exp(-pow(k/eh->kSilk,1.4));

  return jbes0(x_bessel)*(part1+part2);
}

static double tsqr_EH(ccl_parameters *params,eh_struct *eh,double k)
{
  //////
  // Eisenstein & Hu's Tk_c
  // see astro-ph/9709112 for the relevant equations
  // Notice the last parameter in eh_power controls
  // whether to introduce wiggles (BAO) in the power spectrum.
  // We do this by default when obtaining the power spectrum.
  double tk;
  double b_frac=params->Omega_b/params->Omega_m;
  if(eh->wiggled)
    //Case with baryons (Eq 8)
    tk=b_frac*tkEH_b(eh,k)+(1-b_frac)*tkEH_c(eh,k);
  else {
    //Zero baryon case (sec 4.2)
    double OMh2=params->Omega_m*params->h*params->h;
    // Compute Eq. 31
    double alpha_gamma=1-0.328*log(431*OMh2)*b_frac+0.38*log(22.3*OMh2)*b_frac*b_frac;
    // Compute Eq. 30
    double gamma_eff=params->Omega_m*params->h*(alpha_gamma+(1-alpha_gamma)/
						(1+pow(0.43*k*eh->rsound_approx,4)));
    // Compute Eq. 28 (assume k in h/Mpc)
    double q=k*eh->th2p7*eh->th2p7/gamma_eff;
    // Compute Eqs. 29
    double l0=log(2*M_E+1.8*q);
    double c0=14.2+731/(1+62.5*q);
    tk=l0/(l0+c0*q*q);  //T_0 of Eq. 29
  }

  return tk*tk; //Return T_0^2
}

/*
 * Compute the Eisenstein & Hu (1998) unnormalize power spectrum
 * @param params Cosmological parameters
 * @param p, an eh_struct instance
 * @param k, wavenumber in Mpc^-1
 */
double ccl_eh_power(ccl_parameters *params, eh_struct* eh, double k) {
  //Wavenumber in units of Mpc^-1
  double kinvh = k/params->h;  //Changed to h/Mpc
  return pow(k, params->n_s) * tsqr_EH(params, eh, kinvh);
}
