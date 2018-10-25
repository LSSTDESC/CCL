#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>

#include "ccl.h"
#include "ccl_params.h"

/* --------- ROUTINE: h_over_h0 ---------
INPUT: scale factor, cosmology
TASK: Compute E(a)=H(a)/H0
*/
static double h_over_h0(double a, ccl_cosmology * cosmo, int *status)
{
  // Check if massive neutrinos are present - if not, we don't need to
  // compute their contribution
  double Om_mass_nu;
  if ((cosmo->params.N_nu_mass)>1e-12) {
    Om_mass_nu = ccl_Omeganuh2(
      a, cosmo->params.N_nu_mass, cosmo->params.mnu, cosmo->params.T_CMB,
      cosmo->data.accelerator, status) / (cosmo->params.h) / (cosmo->params.h);
    ccl_check_status(cosmo, status);
  }
  else {
    Om_mass_nu = 0;
  }

  /* Calculate h^2 using the formula (eqn 2 in the CCL paper):
    E(a)^2 = Omega_m a^-3 +
             Omega_l a^(-3*(1+w0+wa)) exp(3*wa*(a-1)) +
             Omega_k a^-2 +
             (Omega_g + Omega_n_rel) a^-4 +
             Om_mass_nu
  */
  return sqrt(
    (cosmo->params.Omega_m +
     cosmo->params.Omega_l *
       pow(a,-3*(cosmo->params.w0+cosmo->params.wa)) *
       exp(3*cosmo->params.wa*(a-1)) +
     cosmo->params.Omega_k * a +
     (cosmo->params.Omega_g + cosmo->params.Omega_n_rel) / a +
     Om_mass_nu * a*a*a) / (a*a*a));
}

/* --------- ROUTINE: ccl_omega_x ---------
INPUT: cosmology object, scale factor, species label
TASK: Compute the density relative to critical, Omega(a) for a given species.
Possible values for "label":
ccl_species_crit_label <- critical (physical)
ccl_species_m_label <- matter
ccl_species_l_label <- DE
ccl_species_g_label <- radiation
ccl_species_k_label <- curvature
ccl_species_ur_label <- massless neutrinos
ccl_species_nu_label <- massive neutrinos
*/
double ccl_omega_x(ccl_cosmology * cosmo, double a, ccl_species_x_label label, int *status)
{
  // If massive neutrinos are present, compute the phase-space integral and
  // get OmegaNuh2. If not, set OmegaNuh2 to zero.
  double OmNuh2;
  if ((cosmo->params.N_nu_mass) > 0.0001) {
    // Call the massive neutrino density function just once at this redshift.
    OmNuh2 = ccl_Omeganuh2(a, cosmo->params.N_nu_mass, cosmo->params.mnu,
		       cosmo->params.T_CMB, cosmo->data.accelerator, status);
    ccl_check_status(cosmo, status);
  }
  else {
    OmNuh2 = 0.;
  }

  double hnorm = h_over_h0(a, cosmo, status);

  switch(label) {
    case ccl_species_crit_label :
      return 1.;
    case ccl_species_m_label :
      return cosmo->params.Omega_m / (a*a*a) / hnorm / hnorm;
    case ccl_species_l_label :
      return
        cosmo->params.Omega_l *
        pow(a,-3 * (1 + cosmo->params.w0 + cosmo->params.wa)) *
        exp(3 * cosmo->params.wa * (a-1)) / hnorm / hnorm;
    case ccl_species_g_label :
      return cosmo->params.Omega_g / (a*a*a*a) / hnorm / hnorm;
    case ccl_species_k_label :
      return cosmo->params.Omega_k / (a*a) / hnorm / hnorm;
    case ccl_species_ur_label :
      return cosmo->params.Omega_n_rel / (a*a*a*a) / hnorm / hnorm;
    case ccl_species_nu_label :
      return OmNuh2 / (cosmo->params.h) / (cosmo->params.h) / hnorm / hnorm;
    default:
      *status = CCL_ERROR_PARAMETERS;
      ccl_cosmology_set_status_message(
        cosmo, "ccl_background.c: ccl_omega_x(): Species %d not supported\n", label);
      return NAN;
  }
}

/* --------- ROUTINE: ccl_rho_x ---------
INPUT: cosmology object, scale factor, species label
TASK: Compute rho_x(a), with x defined by species label.
Possible values for "label":
ccl_species_crit_label <- critical (physical)
ccl_species_m_label <- matter (physical)
ccl_species_l_label <- DE (physical)
ccl_species_g_label <- radiation (physical)
ccl_species_k_label <- curvature (physical)
ccl_species_ur_label <- massless neutrinos (physical)
ccl_species_nu_label <- massive neutrinos (physical)
*/
double ccl_rho_x(ccl_cosmology * cosmo, double a, ccl_species_x_label label, int is_comoving, int *status)
{
  double comfac;
  if (is_comoving) {
     comfac = a*a*a;
  } else {
      comfac = 1.0;
  }
  double hnorm = h_over_h0(a, cosmo, status);
  double rhocrit =
    RHO_CRITICAL *
    (cosmo->params.h) *
    (cosmo->params.h) * hnorm * hnorm * comfac;

  return rhocrit * ccl_omega_x(cosmo, a, label, status);
}

// Structure to hold parameters of chi_integrand
typedef struct {
  ccl_cosmology *cosmo;
  int * status;
} chipar;

/* --------- ROUTINE: chi_integrand ---------
INPUT: scale factor
TASK: compute the integrand of the comoving distance
*/
static double chi_integrand(double a, void * params_void)
{
  ccl_cosmology * cosmo = ((chipar *)params_void)->cosmo;
  int *status = ((chipar *)params_void)->status;

  return CLIGHT_HMPC/(a*a*h_over_h0(a, cosmo, status));
}

/* --------- ROUTINE: growth_ode_system ---------
INPUT: scale factor
TASK: Define the ODE system to be solved in order to compute the growth (of the density)
*/
static int growth_ode_system(double a,const double y[],double dydt[],void *params)
{
  int status = 0;
  ccl_cosmology * cosmo = params;

  double hnorm=h_over_h0(a,cosmo, &status);
  double om=ccl_omega_x(cosmo, a, ccl_species_m_label, &status);

  dydt[0]=y[1]/(a*a*a*hnorm);
  dydt[1]=1.5*hnorm*a*om*y[0];

  return status;
}

/* --------- ROUTINE: df_integrand ---------
INPUT: scale factor, spline object
TASK: Compute integrand from modified growth function
*/
static double df_integrand(double a,void * spline_void)
{
  if(a<=0)
    return 0;
  else {
    gsl_spline *df_a_spline=(gsl_spline *)spline_void;

    return gsl_spline_eval(df_a_spline,a,NULL)/a;
  }
}

/* --------- ROUTINE: growth_factor_and_growth_rate ---------
INPUT: scale factor, cosmology
TASK: compute the growth (D(z)) and the growth rate, logarithmic derivative (f?)
*/

static int  growth_factor_and_growth_rate(double a,double *gf,double *fg,ccl_cosmology *cosmo, int *stat)
{
  if(a<EPS_SCALEFAC_GROWTH) {
    *gf=a;
    *fg=1;
    return 0;
  }
  else {
    int gslstatus;
    double y[2];
    double ainit=EPS_SCALEFAC_GROWTH;
    gsl_odeiv2_system sys={growth_ode_system,NULL,2,cosmo};
    gsl_odeiv2_driver *d=
      gsl_odeiv2_driver_alloc_y_new(&sys,gsl_odeiv2_step_rkck,0.1*EPS_SCALEFAC_GROWTH,0,ccl_gsl->ODE_GROWTH_EPSREL);

    y[0]=EPS_SCALEFAC_GROWTH;
    y[1]=EPS_SCALEFAC_GROWTH*EPS_SCALEFAC_GROWTH*EPS_SCALEFAC_GROWTH*
      h_over_h0(EPS_SCALEFAC_GROWTH,cosmo, stat);

    gslstatus=gsl_odeiv2_driver_apply(d,&ainit,a,y);
    gsl_odeiv2_driver_free(d);

    if(gslstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_background.c: growth_factor_and_growth_rate():");
      return NAN;
    }

    *gf=y[0];
    *fg=y[1]/(a*a*h_over_h0(a,cosmo, stat)*y[0]);
    return 0;
  }
}


/* --------- ROUTINE: compute_chi ---------
INPUT: scale factor, cosmology
OUTPUT: chi -> radial comoving distance
TASK: compute radial comoving distance at a
*/
static void compute_chi(double a, ccl_cosmology *cosmo, double * chi, int * stat)
{
  int gslstatus;
  double result;
  chipar p;

  p.cosmo=cosmo;
  p.status=stat;

  gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc(ccl_gsl->N_ITERATION);
  gsl_function F;
  F.function = &chi_integrand;
  F.params = &p;
  //TODO: CQUAD is great, but slower than other methods. This could be sped up if it becomes an issue.
  gslstatus=gsl_integration_cquad(
    &F, a, 1.0, 0.0, ccl_gsl->INTEGRATION_DISTANCE_EPSREL, workspace, &result, NULL, NULL);
  *chi=result/cosmo->params.h;
  gsl_integration_cquad_workspace_free(workspace);

  if (gslstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_background.c: compute_chi():");
    *stat = CCL_ERROR_COMPUTECHI;
  }
}


//Root finding for a(chi)
typedef struct {
  double chi;
  ccl_cosmology *cosmo;
  int * status;
} Fpar;

static double fzero(double a,void *params)
{
  double chi,chia,a_use=a;

  chi=((Fpar *)params)->chi;
  compute_chi(a_use,((Fpar *)params)->cosmo,&chia, ((Fpar *)params)->status);

  return chi-chia;
}

static double dfzero(double a,void *params)
{
  ccl_cosmology *cosmo=((Fpar *)params)->cosmo;
  int *stat = ((Fpar *)params)->status;

  chipar p;
  p.cosmo=cosmo;
  p.status=stat;

  return chi_integrand(a,&p)/cosmo->params.h;
}

static void fdfzero(double a,void *params,double *f,double *df)
{
  *f=fzero(a,params);
  *df=dfzero(a,params);
}

/* --------- ROUTINE: a_of_chi ---------
INPUT: comoving distance chi, cosmology, stat, a_old, gsl_root_fdfsolver
OUTPUT: scale factor
TASK: compute the scale factor that corresponds to a given comoving distance chi
Note: This routine uses a root solver to find an a such that compute_chi(a) = chi.
The root solver uses the derivative of compute_chi (which is chi_integrand) and
the value itself.
*/
static void a_of_chi(double chi, ccl_cosmology *cosmo, int* stat, double *a_old, gsl_root_fdfsolver *s)
{
  if(chi==0) {
    *a_old=1;
  }
  else {
    Fpar p;
    gsl_function_fdf FDF;
    double a_previous,a_current=*a_old;

    p.cosmo=cosmo;
    p.chi=chi;
    p.status=stat;
    FDF.f=&fzero;
    FDF.df=&dfzero;
    FDF.fdf=&fdfzero;
    FDF.params=&p;
    gsl_root_fdfsolver_set(s,&FDF,a_current);

    int iter=0, gslstatus;
    do {
      iter++;
      gslstatus=gsl_root_fdfsolver_iterate(s);
      if(gslstatus!=GSL_SUCCESS) ccl_raise_gsl_warning(gslstatus, "ccl_background.c: a_of_chi():");
      a_previous=a_current;
      a_current=gsl_root_fdfsolver_root(s);
      gslstatus=gsl_root_test_delta(a_current, a_previous, 0, ccl_gsl->ROOT_EPSREL);
    } while(gslstatus==GSL_CONTINUE && iter <= ccl_gsl->ROOT_N_ITERATION);

    *a_old=a_current;

    // Allows us to pass a status to h_over_h0 for the neutrino integral calculation.
    if(gslstatus==GSL_SUCCESS) {
      *stat = *(p.status);
    }
    else {
      ccl_raise_gsl_warning(gslstatus, "ccl_background.c: a_of_chi():");
      *stat = CCL_ERROR_COMPUTECHI;
    }
  }
}

/* ----- ROUTINE: ccl_cosmology_compute_distances ------
INPUT: cosmology
TASK: if not already there, make a table of comoving distances and of E(a)
*/

void ccl_cosmology_compute_distances(ccl_cosmology * cosmo, int *status)
{

  if(cosmo->computed_distances)
    return;

  if(ccl_splines->A_SPLINE_MAX>1.) {
    *status = CCL_ERROR_COMPUTECHI;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: scale factor cannot be larger than 1.\n");
    return;
  }

  // Create logarithmically and then linearly-spaced values of the scale factor
  int na = ccl_splines->A_SPLINE_NA+ccl_splines->A_SPLINE_NLOG-1;
  double * a = ccl_linlog_spacing(ccl_splines->A_SPLINE_MINLOG, ccl_splines->A_SPLINE_MIN, ccl_splines->A_SPLINE_MAX, ccl_splines->A_SPLINE_NLOG, ccl_splines->A_SPLINE_NA);

  if (a==NULL ||
      (fabs(a[0]-ccl_splines->A_SPLINE_MINLOG)>1e-5) ||
      (fabs(a[na-1]-ccl_splines->A_SPLINE_MAX)>1e-5) ||
      (a[na-1]>1.0)) {
      // old:    cosmo->status = CCL_ERROR_LINSPACE;
      *status = CCL_ERROR_LINSPACE;
      ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_distances(): Error creating first logarithmic and then linear spacing in a\n");
      return;
  }

  // allocate space for y, which will be all three
  // of E(a), chi(a), D(a) and f(a) in turn.
  double *y = malloc(sizeof(double)*na);
  if(y==NULL) {
    free(a);
    // old:    cosmo->status=CCL_ERROR_MEMORY;
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_distances(): ran out of memory\n");
    return;
  }

  // Fill in E(a)
  for (int i=0; i<na; i++)
    y[i] = h_over_h0(a[i], cosmo, status);

  // Allocate and fill E spline with values we just got
  gsl_spline * E = gsl_spline_alloc(A_SPLINE_TYPE, na);

  int splstatus = gsl_spline_init(E, a, y, na);
  // Check for errors in creating the spline
  if (splstatus) {
    free(a);
    free(y);
    gsl_spline_free(E);
    *status = CCL_ERROR_SPLINE;
    //    cosmo->status = CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_distances(): Error creating  E(a) spline\n");
    return;
  }

  for (int i=0; i<na; i++) {
    compute_chi(a[i],cosmo,&(y[i]), status);
   }

  if (*status) {
    free(a);
    free(y);
    gsl_spline_free(E);
    *status = CCL_ERROR_INTEG;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_distances(): chi(a) integration error \n");
    return;
  }

  gsl_spline * chi = gsl_spline_alloc(A_SPLINE_TYPE, na);
  splstatus = gsl_spline_init(chi, a, y, na); //in Mpc

  if (splstatus || *status) {
    free(a);
    free(y);
    gsl_spline_free(E);
    gsl_spline_free(chi);
    *status = CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_distances(): Error creating  chi(a) spline\n");
    return;
  }

  //Spline for a(chi)
  double dchi=5.,chi0=y[na-1],chif=y[0],a0=a[na-1],af=a[0];
  //TODO: The interval in chi (5. Mpc) should be made a macro
  free(y); free(a);
  na=(int)((chif-chi0)/dchi);
  y=ccl_linear_spacing(chi0,chif,na);
  dchi=(chif-chi0)/na;
  if(y==NULL || (fabs(y[0]-chi0)>1E-5) || (fabs(y[na-1]-chif)>1e-5)) {
    free(y);
    gsl_spline_free(E);
    gsl_spline_free(chi);
    *status = CCL_ERROR_LINSPACE;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_distances(): Error creating linear spacing in chi\n");
    return;
  }

  a=malloc(sizeof(double)*na);
  if(a==NULL) {
    free(y);
    gsl_spline_free(E);
    gsl_spline_free(chi);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_distances(): ran out of memory\n");
    return;
  }

  a[0]=a0; a[na-1]=af;
  const gsl_root_fdfsolver_type *T=gsl_root_fdfsolver_newton;
  gsl_root_fdfsolver *s=gsl_root_fdfsolver_alloc(T);
  for(int i=1;i<na-1;i++) {
    // we are using the previous value as a guess here to help the root finder
    // as long as we use small steps in a this should be fine
    a_of_chi(y[i],cosmo, status, &a0,s);
    a[i]=a0;
  }

  gsl_root_fdfsolver_free(s);
  if(*status) {
    free(a);
    free(y);
    gsl_spline_free(E);
    gsl_spline_free(chi);
    *status = CCL_ERROR_ROOT;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_distances(): a(chi) root-finding error \n");
    return;
  }

  gsl_spline * achi=gsl_spline_alloc(A_SPLINE_TYPE,na);
  splstatus=gsl_spline_init(achi,y,a,na);

  if (splstatus) {
    free(a);
    free(y);
    gsl_spline_free(E);
    gsl_spline_free(chi);
    gsl_spline_free(achi);
    *status = CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_distances(): Error creating  a(chi) spline\n");
    return;
  }

  if(cosmo->data.accelerator==NULL)
    cosmo->data.accelerator=gsl_interp_accel_alloc();
  cosmo->data.E = E;
  cosmo->data.chi = chi;
  cosmo->data.achi=achi;
  cosmo->computed_distances = true;

  free(a);
  free(y);

}


/* ----- ROUTINE: ccl_cosmology_compute_growth ------
INPUT: cosmology
TASK: if not already there, make a table of growth function and growth rate
      normalize growth to input parameter growth0
*/

void ccl_cosmology_compute_growth(ccl_cosmology * cosmo, int * status)
{

  // This is not valid for massive neutrinos; if we have massive neutrinos, exit.
  if (cosmo->params.N_nu_mass>0){
	  *status = CCL_ERROR_NOT_IMPLEMENTED;
	  ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_growth(): Support for the growth rate in cosmologies with massive neutrinos is not yet implemented.\n");
	  return;
  }

  if(cosmo->computed_growth)
    return;

  // Create logarithmically and then linearly-spaced values of the scale factor
  int  chistatus = 0, na = ccl_splines->A_SPLINE_NA+ccl_splines->A_SPLINE_NLOG-1;
  double * a = ccl_linlog_spacing(ccl_splines->A_SPLINE_MINLOG, ccl_splines->A_SPLINE_MIN, ccl_splines->A_SPLINE_MAX, ccl_splines->A_SPLINE_NLOG, ccl_splines->A_SPLINE_NA);
  if (a==NULL ||
      (fabs(a[0]-ccl_splines->A_SPLINE_MINLOG)>1e-5) ||
      (fabs(a[na-1]-ccl_splines->A_SPLINE_MAX)>1e-5) ||
      (a[na-1]>1.0)
      ) {
    free(a);
    *status = CCL_ERROR_LINSPACE;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_growth(): Error creating logarithmically and then linear spacing in a\n");
    return;
  }

  gsl_integration_cquad_workspace * workspace=NULL;
  gsl_function F;
  gsl_spline *df_a_spline=NULL;
  if(cosmo->params.has_mgrowth) {
    double *df_arr=malloc(na*sizeof(double));
    if(df_arr==NULL) {
      free(a);
      *status=CCL_ERROR_MEMORY;
      ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_distances(): ran out of memory\n");
      return;
    }
    //Generate spline for Delta f(z) that we will then interpolate into an array of a
    gsl_spline *df_z_spline=gsl_spline_alloc(A_SPLINE_TYPE,cosmo->params.nz_mgrowth);
    chistatus=gsl_spline_init(df_z_spline,cosmo->params.z_mgrowth,cosmo->params.df_mgrowth,
			      cosmo->params.nz_mgrowth);

    if(chistatus) {
      free(a);
      free(df_arr);
      gsl_spline_free(df_z_spline);
      *status = CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_growth(): Error creating Delta f(z) spline\n");
      return;
    }
    for (int i=0; i<na; i++) {
      if(a[i]>0) {
        double z=1./a[i]-1.;

        if(z<=cosmo->params.z_mgrowth[0])
          df_arr[i]=cosmo->params.df_mgrowth[0];
        else if(z>cosmo->params.z_mgrowth[cosmo->params.nz_mgrowth-1])
          df_arr[i]=cosmo->params.df_mgrowth[cosmo->params.nz_mgrowth-1];
        else
	       chistatus |= gsl_spline_eval_e(df_z_spline,z,NULL,&df_arr[i]);
      } else {
	     df_arr[i]=0;
      }
    }
    if(chistatus) {
      free(a);
      free(df_arr);
      gsl_spline_free(df_z_spline);
      *status = CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_growth(): Error evaluating Delta f(z) spline\n");
      return;
    }
    gsl_spline_free(df_z_spline);

    //Generate Delta(f) spline
    df_a_spline=gsl_spline_alloc(A_SPLINE_TYPE,na);
    chistatus=gsl_spline_init(df_a_spline,a,df_arr,na);
    free(df_arr);
    if (chistatus) {
      free(a);
      gsl_spline_free(df_a_spline);
      *status = CCL_ERROR_SPLINE;
      ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_growth(): Error creating Delta f(a) spline\n");
      return;
    }

    workspace=gsl_integration_cquad_workspace_alloc(ccl_gsl->N_ITERATION);
    F.function=&df_integrand;
    F.params=df_a_spline;
  }

  // allocate space for y, which will be all three
  // of E(a), chi(a), D(a) and f(a) in turn.
  int  status_mg=0, gslstatus;
  double growth0,fgrowth0;
  double *y = malloc(sizeof(double)*na);
  if(y==NULL) {
    free(a);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_distances(): ran out of memory\n");
    return;
  }
  double *y2 = malloc(sizeof(double)*na);
  if(y2==NULL) {
    free(a);
    free(y);
    *status=CCL_ERROR_MEMORY;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_distances(): ran out of memory\n");
    return;
  }

  chistatus|=growth_factor_and_growth_rate(1.,&growth0,&fgrowth0,cosmo, status);
  for(int i=0; i<na; i++) {
    chistatus|=growth_factor_and_growth_rate(a[i],&(y[i]),&(y2[i]),cosmo, status);
    if(cosmo->params.has_mgrowth) {
      if(a[i]>0) {
	double df,integ;
	//Add modification to f
	gslstatus = gsl_spline_eval_e(df_a_spline,a[i],NULL,&df);
  if(gslstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_background.c: ccl_cosmology_compute_growth():");
    status_mg |= gslstatus;
  }
	y2[i]+=df;
	//Multiply D by exp(-int(df))
	gslstatus = gsl_integration_cquad(&F,a[i],1.0,0.0,ccl_gsl->INTEGRATION_DISTANCE_EPSREL,workspace,&integ,NULL,NULL);
	if(gslstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_background.c: ccl_cosmology_compute_growth():");
    status_mg |= gslstatus;
  }
  y[i]*=exp(-integ);
      }
    }
    y[i]/=growth0;
  }
  if(chistatus || status_mg || *status) {
    free(a);
    free(y);
    free(y2);
    if(df_a_spline!=NULL)
      gsl_spline_free(df_a_spline);
    if(workspace!=NULL)
      gsl_integration_cquad_workspace_free(workspace);
    if (chistatus) {
      *status = CCL_ERROR_INTEG;
      ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_growth(): integral for linear growth factor didn't converge\n");
    }
    if(status_mg) {
      *status = CCL_ERROR_INTEG;
      ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_growth(): integral for MG growth factor didn't converge\n");
    }
    return;
  }

  if(cosmo->params.has_mgrowth) {
    gsl_spline_free(df_a_spline);
    gsl_integration_cquad_workspace_free(workspace);
  }

  gsl_spline * growth = gsl_spline_alloc(A_SPLINE_TYPE, na);
  chistatus = gsl_spline_init(growth, a, y, na);
  if(chistatus) {
    free(a);
    free(y);
    free(y2);
    gsl_spline_free(growth);
    *status = CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_growth(): Error creating D(a) spline\n");
    return;
  }

  gsl_spline * fgrowth = gsl_spline_alloc(A_SPLINE_TYPE, na);
  chistatus = gsl_spline_init(fgrowth, a, y2, na);
  if(chistatus) {
    free(a);
    free(y);
    free(y2);
    gsl_spline_free(growth);
    gsl_spline_free(fgrowth);
    *status = CCL_ERROR_SPLINE;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_cosmology_compute_growth(): Error creating f(a) spline\n");
    return;
  }

  // Initialize the accelerator which speeds the splines and
  // assign all the splines we've just made to the structure.
  if(cosmo->data.accelerator==NULL)
    cosmo->data.accelerator=gsl_interp_accel_alloc();
  cosmo->data.growth = growth;
  cosmo->data.fgrowth = fgrowth;
  cosmo->data.growth0 = growth0;
  cosmo->computed_growth = true;

  free(a);
  free(y);
  free(y2);

  return;
}

//Expansion rate normalized to 1 today

double ccl_h_over_h0(ccl_cosmology * cosmo, double a, int* status)
{

  if(!cosmo->computed_distances) {
    ccl_cosmology_compute_distances(cosmo,status);
    ccl_check_status(cosmo, status);
  }

  double h_over_h0;
  int gslstatus = gsl_spline_eval_e(cosmo->data.E, a, cosmo->data.accelerator,&h_over_h0);
  if(gslstatus != GSL_SUCCESS) {
    ccl_raise_gsl_warning(gslstatus, "ccl_background.c: ccl_h_over_h0():");
    *status = gslstatus;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_h_over_h0(): Scale factor outside interpolation range.\n");
    return NAN;
  }

  return h_over_h0;
}


void ccl_h_over_h0s(ccl_cosmology * cosmo, int na, double a[], double output[], int * status)
{
  int _status;

  for (int i=0; i<na; i++) {
    _status = 0;
    output[i] = ccl_h_over_h0(cosmo, a[i], &_status);
    *status |= _status;
  }
}

// Distance-like function examples, all in Mpc
double ccl_comoving_radial_distance(ccl_cosmology * cosmo, double a, int * status)
{
  if((a > (1.0 - 1.e-8)) && (a<=1.0)) {
    return 0.;
  }
  else if(a>1.) {
    *status = CCL_ERROR_COMPUTECHI;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: scale factor cannot be larger than 1.\n");
    ccl_check_status(cosmo,status);
    return NAN;
  }
  else {
    if(!cosmo->computed_distances) {
      ccl_cosmology_compute_distances(cosmo, status);
      ccl_check_status(cosmo,status);
    }

    double crd;
    int gslstatus = gsl_spline_eval_e(cosmo->data.chi, a, cosmo->data.accelerator, &crd);
    if(gslstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_background.c: ccl_comoving_radial_distance():");
      *status = gslstatus;
      ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_comoving_radial_distance(): Scale factor outside interpolation range.\n");
      return NAN;
    }
    return crd;
  }
}

void ccl_comoving_radial_distances(ccl_cosmology * cosmo, int na, double a[], double output[], int* status)
{
  int _status;

  for (int i=0; i<na; i++) {
    _status = 0;
    output[i] = ccl_comoving_radial_distance(cosmo, a[i], &_status);
    *status |= _status;
  }
}

double ccl_sinn(ccl_cosmology *cosmo, double chi, int *status)
{
  //////
  //         { sin(x)  , if k==1
  // sinn(x)={  x      , if k==0
  //         { sinh(x) , if k==-1
  switch(cosmo->params.k_sign) {
  case -1:
    return sinh(cosmo->params.sqrtk * chi) / cosmo->params.sqrtk;
  case 1:
    return sin(cosmo->params.sqrtk*chi) / cosmo->params.sqrtk;
  case 0:
    return chi;
  default:
    *status = CCL_ERROR_PARAMETERS;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_sinn: ill-defined cosmo->params.k_sign = %d",
	    cosmo->params.k_sign);
    return NAN;
  }
}

double ccl_comoving_angular_distance(ccl_cosmology * cosmo, double a, int* status)
{
  if((a > (1.0 - 1.e-8)) && (a<=1.0)) {
    return 0.;
  }
  else if(a>1.) {
    *status = CCL_ERROR_COMPUTECHI;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: scale factor cannot be larger than 1.\n");
    ccl_check_status(cosmo,status);
    return NAN;
  }
  else {
    if (!cosmo->computed_distances) {
      ccl_cosmology_compute_distances(cosmo, status);
      ccl_check_status(cosmo, status);
    }

    double chi;
    int gslstatus = gsl_spline_eval_e(cosmo->data.chi, a,
                                      cosmo->data.accelerator,&chi);
    if(gslstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_background.c: ccl_comoving_angular_distance():");
      *status |= gslstatus;
      ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_comoving_angular_distance(): Scale factor outside interpolation range.\n");
      return NAN;
    }
    return ccl_sinn(cosmo,chi,status);
  }
}

void ccl_comoving_angular_distances(ccl_cosmology * cosmo, int na, double a[],
                                    double output[], int* status)
{
  int _status;

  for (int i=0; i < na; i++) {
    _status = 0;
    output[i] = ccl_comoving_angular_distance(cosmo, a[i], &_status);
    *status |= _status;
  }
}

double ccl_luminosity_distance(ccl_cosmology * cosmo, double a, int* status)
{
  return ccl_comoving_angular_distance(cosmo, a, status) / a;
}

void ccl_luminosity_distances(ccl_cosmology * cosmo, int na, double a[], double output[], int * status)
{
  int _status;

  for (int i=0; i<na; i++) {
    _status = 0;
    output[i] = ccl_luminosity_distance(cosmo, a[i], &_status);
    *status |= _status;
  }
}

double ccl_distance_modulus(ccl_cosmology * cosmo, double a, int* status)
{
  if((a > (1.0 - 1.e-8)) && (a<=1.0)) {
    *status = CCL_ERROR_COMPUTECHI;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: distance_modulus undefined for a=1.\n");
    ccl_check_status(cosmo,status);
    return NAN;
  } else if(a>1.) {
    *status = CCL_ERROR_COMPUTECHI;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: scale factor cannot be larger than 1.\n");
    ccl_check_status(cosmo,status);
    return NAN;
  } else {
    if (!cosmo->computed_distances) {
      ccl_cosmology_compute_distances(cosmo, status);
      ccl_check_status(cosmo, status);
    }
    /* distance modulus = 5 * log10(d) - 5
       Since d in CCL is in Mpc, you get
         5*log10(10^6) - 5 = 30 - 5 = 25
      for the constant.
    */
    return 5 * log10(ccl_luminosity_distance(cosmo, a, status)) + 25;
  }
}


void ccl_distance_moduli(ccl_cosmology * cosmo, int na, double a[], double output[], int * status)
{
  int _status;

  for (int i=0; i<na; i++) {
    _status = 0;
    output[i] = ccl_distance_modulus(cosmo, a[i], &_status);
    *status |= _status;
  }
}

//Scale factor for a given distance
double ccl_scale_factor_of_chi(ccl_cosmology * cosmo, double chi, int * status)
{
  if((chi < 1.e-8) && (chi>=0.)) {
    return 1.;
  }
  else if(chi<0.) {
    *status = CCL_ERROR_COMPUTECHI;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: distance cannot be smaller than 0.\n");
    ccl_check_status(cosmo,status);
    return NAN;
  }
  else {
    if (!cosmo->computed_distances) {
      ccl_cosmology_compute_distances(cosmo,status);
      ccl_check_status(cosmo,status);
    }
    double a;
    int gslstatus = gsl_spline_eval_e(cosmo->data.achi, chi,cosmo->data.accelerator_achi, &a);
    if(gslstatus != GSL_SUCCESS) {
      ccl_raise_gsl_warning(gslstatus, "ccl_background.c: ccl_scale_factor_of_chi():");
      *status |= gslstatus;
    }
    return a;
  }
}

//
void ccl_scale_factor_of_chis(ccl_cosmology * cosmo, int nchi, double chi[], double output[], int * status)
{
  int _status;

  for (int i=0; i<nchi; i++) {
    _status = 0;
    output[i] = ccl_scale_factor_of_chi(cosmo, chi[i], &_status);
    *status |= _status;
  }
}

double ccl_growth_factor(ccl_cosmology * cosmo, double a, int * status)
{
  if(a==1.){
    return 1.;
  }
  else if(a>1.) {
    *status = CCL_ERROR_COMPUTECHI;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: scale factor cannot be larger than 1.\n");
    ccl_check_status(cosmo,status);
    return NAN;
  }
  else {
    if (!cosmo->computed_growth) {
      ccl_cosmology_compute_growth(cosmo, status);
      ccl_check_status(cosmo, status);
    }
    if (*status!= CCL_ERROR_NOT_IMPLEMENTED) {
      double D;
      int gslstatus = gsl_spline_eval_e(cosmo->data.growth, a, cosmo->data.accelerator,&D);
      if(gslstatus != GSL_SUCCESS) {
        ccl_raise_gsl_warning(gslstatus, "ccl_background.c: ccl_growth_factor():");
        *status |= gslstatus;
        ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_growth_factor(): Scale factor outside interpolation range.\n");
        return NAN;
      }
      return D;
    }
    else {
      return NAN;
    }
  }
}

void ccl_growth_factors(ccl_cosmology * cosmo, int na, double a[], double output[], int * status)
{
  int _status;

  for (int i=0; i<na; i++) {
    _status = 0;
    output[i] = ccl_growth_factor(cosmo, a[i], &_status);
    *status |= _status;
  }
}

double ccl_growth_factor_unnorm(ccl_cosmology * cosmo, double a, int * status)
{

  if (!cosmo->computed_growth) {
    ccl_cosmology_compute_growth(cosmo, status);
    ccl_check_status(cosmo, status);
  }

  if (*status != CCL_ERROR_NOT_IMPLEMENTED) {
      return cosmo->data.growth0 * ccl_growth_factor(cosmo, a, status);
  } else {
    return NAN;
  }
}

void ccl_growth_factors_unnorm(ccl_cosmology * cosmo, int na, double a[], double output[], int * status)
{
  int _status;

  for (int i=0; i<na; i++) {
    _status = 0;
    output[i] = ccl_growth_factor_unnorm(cosmo, a[i], &_status);
    *status |= _status;
  }
}

double ccl_growth_rate(ccl_cosmology * cosmo, double a, int * status)
{
  if(a>1.) {
    *status = CCL_ERROR_COMPUTECHI;
    ccl_cosmology_set_status_message(cosmo, "ccl_background.c: scale factor cannot be larger than 1.\n");
    ccl_check_status(cosmo,status);
    return NAN;
  } else {
    if (!cosmo->computed_growth) {
      ccl_cosmology_compute_growth(cosmo, status);
      ccl_check_status(cosmo, status);
    }
    if(*status != CCL_ERROR_NOT_IMPLEMENTED) {
      double g;
      int gslstatus = gsl_spline_eval_e(cosmo->data.fgrowth, a, cosmo->data.accelerator,&g);
      if(gslstatus != GSL_SUCCESS) {
        ccl_raise_gsl_warning(gslstatus, "ccl_background.c: ccl_growth_rate():");
        *status |= gslstatus;
        ccl_cosmology_set_status_message(cosmo, "ccl_background.c: ccl_growth_rate(): Scale factor outside interpolation range.\n");
        return NAN;
      }
      return g;
    } else {
	    return NAN;
	  }
  }
}

void ccl_growth_rates(ccl_cosmology * cosmo, int na, double a[], double output[], int * status)
{
  int _status;

  for (int i=0; i<na; i++) {
    _status = 0;
    output[i] = ccl_growth_rate(cosmo, a[i], &_status);
    *status |= _status;
  }
}
