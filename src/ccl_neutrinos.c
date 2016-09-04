#include "ccl_neutrinos.h"
#include "ccl_utils.h"
#include "ccl_constants.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_integration.h"

static double ccl_nu_integrand(double x, void *r)
{
  double rat=*((double*)(r));
  return sqrt(x*x+rat*rat)/(exp(x)+1.0)*x*x;
}


void ccl_calculate_nu_phasespace_spline(gsl_spline* spl) {
  int N=CCL_NU_MNUT_N;
  double *mnut = ccl_linear_spacing(log(CCL_NU_MNUT_MIN),
				    log(CCL_NU_MNUT_MAX),
				    (log(CCL_NU_MNUT_MAX)-log(CCL_NU_MNUT_MIN))/(CCL_NU_MNUT_N-1),
				    &N);
  double *y=malloc(sizeof(double)*CCL_NU_MNUT_N);
  
  int status=0;
  gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc (1000);
  gsl_function F;
  F.function = &ccl_nu_integrand;
  for (int i=0; i<CCL_NU_MNUT_N; i++) {
    double mnut_=exp(mnut[i]);
    F.params = &(mnut_);
    status |= gsl_integration_cquad(&F, 0, 1000.0, 1e-6, 0.0, workspace,&y[i], NULL, NULL); 
  }
  gsl_integration_cquad_workspace_free(workspace);
  double renorm=1./y[0];
  for (int i=0; i<CCL_NU_MNUT_N; i++) y[i]*=renorm;
  spl = gsl_spline_alloc(A_SPLINE_TYPE, CCL_NU_MNUT_N);
  status = gsl_spline_init(spl, mnut, y, CCL_NU_MNUT_N);
  // Check for errors in creating the spline
  if (status){
    free(mnut);
    free(y);
    gsl_spline_free(spl);
    fprintf(stderr, "Error creating mnu/T neutrino spline\n");
    return;
  }
  free(mnut);
}

// returns density if one neutrino species at a scale factor a, given this particular
// species' Neff and sum_mnu
// work out which units do you want
double Omeganuh2 (double a, double Neff, double mnu, double TCMB, gsl_spline* psi) {
  if (Neff==0) return 0.0;
  double Tnu=TCMB*pow(4./11.,1./3.);
  // effective neutrino temperature, very academic, but required for consistency with CLASS
  double Tnu_eff = Tnu*pow(3.046/3.,0.25);
  double a4=a*a*a*a;
  double prefix = 4.48130979e-7*Tnu*Tnu*Tnu*Tnu;
  if (mnu==0) return Neff*prefix*7./8./a4;
  // mass of one
  double mnuone=mnu/Neff;
  // mass over T
  // This returns the density at a normalized so that
  // we get nuh2 at a=0
  // (1 eV) / (Boltzmann constant * 1 kelvin) =
  // 11 604.5193
  double mnuOT=mnuone/(Tnu_eff/a)*11604.519;
  double intval;
  if (mnuOT<CCL_NU_MNUT_MIN) intval=7./8.;
  else if (mnuOT>CCL_NU_MNUT_MAX) intval=0.2776566337*mnuOT; //evalf(45*Zeta(3)/(2*Pi^4));
  else intval=gsl_spline_eval(psi, mnuOT,NULL)*7./8.;
  return Neff*intval*prefix/a4;
}

