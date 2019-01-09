#include "common.h"

#ifdef _NOLIM_CMBL
#define N_CHI 1000
static double transfer_cmblens_nolim(int l,double k,RunParams *par)
{
  int i;
  double ret=0;
  double dchi=par->chi_LSS/N_CHI;
  for(i=0;i<N_CHI;i++) {
    double chi=(i+0.5)*dchi;
    double w=1-chi/par->chi_LSS;
    double a=spline_eval(chi,par->aofchi);
    double gf=spline_eval(chi,par->gfofchi);
    double jl=csm_j_bessel(l,k*chi);

    ret+=gf*w*jl/(chi*a);
  }

  return ret*(par->prefac_lensing)*dchi*l*(l+1)/k*sqrt((2*l+1.)/M_PI);
}
#endif //_NOLIM_CMBL

static double transfer_dens(int l,double k,RunParams *par,int ibin)
{
  double chi=(l+0.5)/k;
  double gf=spline_eval(chi,par->gfofchi);
  double h=spline_eval(chi,par->hofchi);
  double z=spline_eval(chi,par->zofchi);
  double pz=spline_eval(z,par->wind_0[ibin]);
  double bz=spline_eval(z,par->bias);

  return pz*bz*gf*h;
}

static double transfer_rsd(int l,double k,RunParams *par,int ibin)
{
  double chi0=(l+0.5)/k;
  double chi1=(l+1.5)/k;
  double z0=spline_eval(chi0,par->zofchi);
  double z1=spline_eval(chi1,par->zofchi);
  double gf0=spline_eval(chi0,par->gfofchi);
  double gf1=spline_eval(chi1,par->gfofchi);
  double fg0=spline_eval(chi0,par->fgofchi);
  double fg1=spline_eval(chi1,par->fgofchi);
  double h0=spline_eval(chi0,par->hofchi);
  double h1=spline_eval(chi1,par->hofchi);
  double pz0=spline_eval(z0,par->wind_0[ibin]);
  double pz1=spline_eval(z1,par->wind_0[ibin]);
  double term0=pz0*fg0*gf0*h0*(1+8.*l)/((2*l+1.)*(2*l+1.));
  double term1=pz1*fg1*gf1*h1*sqrt((l+0.5)/(l+1.5))*4./(2*l+3);

  return term0-term1;
}

static double transfer_isw(int l,double k,RunParams *par)
{
  double chi=(l+0.5)/k;
  double gf=spline_eval(chi,par->gfofchi);
  double h=spline_eval(chi,par->hofchi);
  double fg=spline_eval(chi,par->fgofchi);

  return 2*par->prefac_lensing*h*gf*(1-fg)/(k*k);
}

static double transfer_magnification(int l,double k,RunParams *par,int ibin)
{
  double chi=(l+0.5)/k;
  double gf=spline_eval(chi,par->gfofchi);
  double a=spline_eval(chi,par->aofchi);
  double w=spline_eval(chi,par->wind_M[ibin]);

  if(w<=0)
    return 0;
  else
    return -2*par->prefac_lensing*l*(l+1)*gf*w/(a*chi*k*k);
}

static double transfer_lensing(int l,double k,RunParams *par,int ibin)
{
  double chi=(l+0.5)/k;
  double gf=spline_eval(chi,par->gfofchi);
  double a=spline_eval(chi,par->aofchi);
  double w=spline_eval(chi,par->wind_L[ibin]);

  if(w<=0)
    return 0;
  else
    return par->prefac_lensing*l*(l+1)*gf*w/(a*chi*k*k);
}

static double transfer_cmblens(int l,double k,RunParams *par)
{
  double chi=(l+0.5)/k;

  if(chi>=par->chi_LSS)
    return 0;
  else {
    double gf=spline_eval(chi,par->gfofchi);
    double a=spline_eval(chi,par->aofchi);
    double w=1-chi/par->chi_LSS;

    return par->prefac_lensing*l*(l+1)*gf*w/(a*chi*k*k);
  }
}

double transfer_wrap(int l,double k,RunParams *par,char *trtype,int ibin)
{
  if(!strcmp(trtype,"nc")) {
    if(par->do_nc!=1)
      dam_report_error(1,"Asked to calculate NC transfer function, but can't!\n");
    else {
      double tr=0;
      if(par->has_dens)
	tr+=transfer_dens(l,k,par,ibin);
      if(par->has_rsd)
	tr+=transfer_rsd(l,k,par,ibin);
      if(par->has_lensing)
	tr+=transfer_magnification(l,k,par,ibin);
      return tr;
    }
  }
  if(!strcmp(trtype,"isw")) {
    if(par->do_isw!=1)
      dam_report_error(1,"Asked to calculate ISW transfer function, but can't!\n");
    else
      return transfer_isw(l,k,par);
  }
  if(!strcmp(trtype,"shear")) {
    if(par->do_shear!=1)
      dam_report_error(1,"Asked to calculate shear transfer function, but can't!\n");
    else
      return transfer_lensing(l,k,par,ibin);
  }
  if(!strcmp(trtype,"cmblens")) {
    if(par->do_cmblens!=1)
      dam_report_error(1,"Asked to calculate shear transfer function, but can't!\n");
    else
#ifdef _NOLIM_CMBL
      return transfer_cmblens_nolim(l,k,par);
#else //_NOLIM_CMBL
      return transfer_cmblens(l,k,par);
#endif //_NOLIM_CMBL
  }
  else
    dam_report_error(1,"Unknown transfer type %s\n",trtype);
  return -1.;
}
