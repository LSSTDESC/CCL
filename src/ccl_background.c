#include "ccl_background.h"
#include "ccl_utils.h"
#include "ccl_error.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "gsl/gsl_errno.h"
//#include "gsl/gsl_odeiv.h"
#include "gsl/gsl_odeiv2.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_integration.h"

//TODO: is it worth separating between cases for speed purposes?
//E.g. flat vs non-flat, LDCM vs wCDM
//CHANGED: modified this to include non-flat cosmologies

/* --------- ROUTINE: h_over_h0 ---------
INPUT: scale factor, cosmological parameters
TASK: Compute E(z)=H(z)/H0
*/
static double h_over_h0(double a, ccl_parameters * params)
{
  return sqrt((params->Omega_m+params->Omega_l*pow(a,-3*(params->w0+params->wa))*
	       exp(3*params->wa*(a-1))+params->Omega_k*a+params->Omega_g/a)/(a*a*a));
}

/* --------- ROUTINE: omega_m_z ---------
INPUT: scale factor, cosmological parameters
TASK: Compute Omega_m(z)
*/
static double omega_m_z(double a,ccl_parameters * params)
{
  return params->Omega_m/(params->Omega_m+params->Omega_l*pow(a,-3*(params->w0+params->wa))*
			  exp(3*params->wa*(a-1))+params->Omega_g/a+params->Omega_k*a);
}

/* --------- ROUTINE: chi_integrand ---------
INPUT: scale factor
TASK: compute the integrand of the comoving distance
*/
static double chi_integrand(double a, void * cosmo_void)
{
  ccl_cosmology * cosmo = cosmo_void;
  return CLIGHT_HMPC/(a*a*h_over_h0(a, &(cosmo->params)));
}

/* --------- ROUTINE: growth_ode_system ---------
INPUT: scale factor
TASK: Define the ODE system to be solved in order to compute the growth (of the density)
*/
static int growth_ode_system(double a,const double y[],double dydt[],void *params)
{
  ccl_cosmology * cosmo = params;
  double hnorm=h_over_h0(a,&(cosmo->params));
  double om=omega_m_z(a,&(cosmo->params));

  dydt[0]=y[1]/(a*a*a*hnorm);
  dydt[1]=1.5*hnorm*a*om*y[0];

  return GSL_SUCCESS;
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
static void growth_factor_and_growth_rate(double a,double *gf,double *fg,ccl_cosmology *cosmo)
{
  if(a<EPS_SCALEFAC_GROWTH) {
    *gf=a;
    *fg=1;
    return;
  }
  else {
    double y[2];
    double ainit=EPS_SCALEFAC_GROWTH;
    gsl_odeiv2_system sys={growth_ode_system,NULL,2,cosmo};
    gsl_odeiv2_driver *d=
      gsl_odeiv2_driver_alloc_y_new(&sys,gsl_odeiv2_step_rkck,0.1*EPS_SCALEFAC_GROWTH,0,EPSREL_GROWTH);

    y[0]=EPS_SCALEFAC_GROWTH;
    y[1]=EPS_SCALEFAC_GROWTH*EPS_SCALEFAC_GROWTH*EPS_SCALEFAC_GROWTH*
      h_over_h0(EPS_SCALEFAC_GROWTH,&(cosmo->params));


    int status=gsl_odeiv2_driver_apply(d,&ainit,a,y);
    gsl_odeiv2_driver_free(d);
    
    if(status!=GSL_SUCCESS) {
      cosmo->status = 1;
      strcpy(cosmo->status_message ,"ccl_background.c: growth_factor_and_growth_rate(): ODE for growth factor didn't converge\n");
    }
    
    *gf=y[0];
    *fg=y[1]/(a*a*h_over_h0(a,&(cosmo->params))*y[0]);
  }
}

/* ----- ROUTINE: ccl_cosmology_compute_distances ------
INPUT: cosmology
TASK: if not already there, make a table of comoving distances and of E(a)
*/

void ccl_cosmology_compute_distances(ccl_cosmology * cosmo)
{
  if(cosmo->computed_distances)
    return;

  // Create linearly-spaced values of the scale factor
  int na = A_SPLINE_NA;
  double * a = ccl_linear_spacing(A_SPLINE_MIN, A_SPLINE_MAX, na);
  if (a==NULL || 
      (fabs(a[0]-A_SPLINE_MIN)>1e-5) || 
      (fabs(a[na-1]-A_SPLINE_MAX)>1e-5) || 
      (a[na-1]>1.0)
      ) {
    cosmo->status = 2;
    strcpy(cosmo->status_message,"ccl_background.c: ccl_cosmology_compute_distances(): Error creating linear spacing in a\n");
    return;
  }

  // allocate space for y, which will be all three
  // of E(a), chi(a), D(a) and f(a) in turn.
  double *y = malloc(sizeof(double)*na);

  // Fill in E(a)
  for (int i=0; i<na; i++){
    y[i] = h_over_h0(a[i], &cosmo->params);
  }

  // Allocate and fill E spline with values we just got
  gsl_spline * E = gsl_spline_alloc(A_SPLINE_TYPE, na);
  int status = gsl_spline_init(E, a, y, na);
  // Check for errors in creating the spline
  if (status){
    free(a);
    free(y);
    gsl_spline_free(E);
    cosmo->status = 4;
    strcpy(cosmo->status_message,"ccl_background.c: ccl_cosmology_compute_distances(): Error creating  E(a) spline\n");
    return;
  }

  //Fill in chi(a)
  //TODO: CQUAD is great, but slower than other methods. This could be sped up if it becomes an issue.
  gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc (1000);
  gsl_function F;
  F.function = &chi_integrand;
  F.params = cosmo;
  for (int i=0; i<na; i++){
    status |= gsl_integration_cquad(&F, a[i], 1.0, 0.0,EPSREL_DIST,workspace,&y[i], NULL, NULL); 
    y[i]/=cosmo->params.h;
  }
  gsl_integration_cquad_workspace_free(workspace);
  if (status){
    free(a);
    free(y);
    gsl_spline_free(E);        
    cosmo->status = 5;
    strcpy(cosmo->status_message,"ccl_background.c: ccl_cosmology_compute_distances(): chi(a) integration error \n");
    return;
  }

  gsl_spline * chi = gsl_spline_alloc(A_SPLINE_TYPE, na);
  status = gsl_spline_init(chi, a, y, na); //in Mpc
  if (status){
    free(a);
    free(y);
    gsl_spline_free(E);
    gsl_spline_free(chi);
    cosmo->status = 4;
    strcpy(cosmo->status_message,"ccl_background.c: ccl_cosmology_compute_distances(): Error creating  chi(a) spline\n");
    return;
  }

  if(cosmo->data.accelerator==NULL)
    cosmo->data.accelerator=gsl_interp_accel_alloc();
  cosmo->data.E = E;
  cosmo->data.chi = chi;
  cosmo->computed_distances = true;
  
  free(a);
  free(y);
}


/* ----- ROUTINE: ccl_cosmology_compute_growth ------
INPUT: cosmology
TASK: if not already there, make a table of growth function and growth rate
      normalize growth to input parameter growth0
*/

void ccl_cosmology_compute_growth(ccl_cosmology * cosmo)
{
  if(cosmo->computed_growth)
    return;

  // Create linearly-spaced values of the scale factor
  int status = 0, na = A_SPLINE_NA;
  double * a = ccl_linear_spacing(A_SPLINE_MIN, A_SPLINE_MAX, na);
  if (a==NULL || 
      (fabs(a[0]-A_SPLINE_MIN)>1e-5) || 
      (fabs(a[na-1]-A_SPLINE_MAX)>1e-5) || 
      (a[na-1]>1.0)
      ) {
    cosmo->status = 2;
    strcpy(cosmo->status_message,"ccl_background.c: ccl_cosmology_compute_growth(): Error creating linear spacing in a\n");
    fprintf(stderr, "Error creating linear spacing.\n");        
    return;
  }

  gsl_integration_cquad_workspace * workspace=NULL;
  gsl_function F;
  gsl_spline *df_a_spline=NULL;
  if(cosmo->params.has_mgrowth) {
    double *df_arr=malloc(na*sizeof(double));
    //Generate spline for Delta f(z) that we will then interpolate into an array of a
    gsl_spline *df_z_spline=gsl_spline_alloc(A_SPLINE_TYPE,cosmo->params.nz_mgrowth);
    status=gsl_spline_init(df_z_spline,cosmo->params.z_mgrowth,cosmo->params.df_mgrowth,
			    cosmo->params.nz_mgrowth);
    if(status) {
      free(a);
      free(df_arr);
      gsl_spline_free(df_z_spline);
      cosmo->status = 4;
      strcpy(cosmo->status_message,"ccl_background.c: ccl_cosmology_compute_growth(): Error creating Delta f(z) spline\n");
      return;
    }
    for (int i=0; i<na; i++){
      if(a[i]>0) {
	       double z=1./a[i]-1.;
	       if(z<=cosmo->params.z_mgrowth[0]) 
	          df_arr[i]=cosmo->params.df_mgrowth[0];
	       else if(z>cosmo->params.z_mgrowth[cosmo->params.nz_mgrowth-1]) 
	          df_arr[i]=cosmo->params.df_mgrowth[cosmo->params.nz_mgrowth-1];
	       else
	          df_arr[i]=gsl_spline_eval(df_z_spline,z,NULL);
      }
      else
	       df_arr[i]=0;
    }
    gsl_spline_free(df_z_spline);

    //Generate Delta(f) spline
    df_a_spline=gsl_spline_alloc(A_SPLINE_TYPE,na);
    status=gsl_spline_init(df_a_spline,a,df_arr,na);
    if (status){
      free(a);
      free(df_arr);
      gsl_spline_free(df_a_spline);
      cosmo->status = 4;
      strcpy(cosmo->status_message,"ccl_background.c: ccl_cosmology_compute_growth(): Error creating Delta f(a) spline\n");
      return;
    }
    free(df_arr);

    workspace=gsl_integration_cquad_workspace_alloc(1000);
    F.function=&df_integrand;
    F.params=df_a_spline;
  }

  // allocate space for y, which will be all three
  // of E(a), chi(a), D(a) and f(a) in turn.
  double growth0,fgrowth0;
  double *y = malloc(sizeof(double)*na);
  double *y2 = malloc(sizeof(double)*na);
  growth_factor_and_growth_rate(1.,&growth0,&fgrowth0,cosmo);
  for (int i=0; i<na; i++){
    growth_factor_and_growth_rate(a[i],&(y[i]),&(y2[i]),cosmo);
    if(cosmo->params.has_mgrowth) {
      if(a[i]>0) {
	       double df,integ;
	       //Add modification to f
	       df=gsl_spline_eval(df_a_spline,a[i],NULL);
	       y2[i]+=df;
	       //Multiply D by exp(-int(df))
	       status |= gsl_integration_cquad(&F,a[i],1.0,0.0,EPSREL_DIST,workspace,&integ,NULL,NULL);
	       y[i]*=exp(-integ);
      }
    }
    y[i]/=growth0;
  }
  if (status || cosmo->status){
    free(a);
    free(y);
    free(y2);
    if(df_a_spline!=NULL)
      gsl_spline_free(df_a_spline);
    if(workspace!=NULL)
      gsl_integration_cquad_workspace_free(workspace);
    if (status){
      cosmo->status = 5;
      strcpy(cosmo->status_message ,"ccl_background.c: ccl_cosmology_compute_growth(): integral for MG growth factor didn't converge\n");
    }
    return;
  }
   
  if(cosmo->params.has_mgrowth) {
    gsl_spline_free(df_a_spline);
    gsl_integration_cquad_workspace_free(workspace);
  }

  gsl_spline * growth = gsl_spline_alloc(A_SPLINE_TYPE, na);
  status = gsl_spline_init(growth, a, y, na);
  if (status){
    free(a);
    free(y);
    free(y2);
    gsl_spline_free(growth);
    cosmo->status = 4;
    strcpy(cosmo->status_message,"ccl_background.c: ccl_cosmology_compute_growth(): Error creating D(a) spline\n");
    return;
  }
  gsl_spline * fgrowth = gsl_spline_alloc(A_SPLINE_TYPE, na);
  status = gsl_spline_init(fgrowth, a, y2, na);
  if (status){
    free(a);
    free(y);
    free(y2);
    gsl_spline_free(growth);
    gsl_spline_free(fgrowth);
    cosmo->status = 4;
    strcpy(cosmo->status_message,"ccl_background.c: ccl_cosmology_compute_growth(): Error creating f(a) spline\n");
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

// Distance-like function examples, all in Mpc

double ccl_comoving_radial_distance(ccl_cosmology * cosmo, double a)
{
  if (!cosmo->computed_distances){
    ccl_cosmology_compute_distances(cosmo);
    ccl_check_status(cosmo);    
  }
   return gsl_spline_eval(cosmo->data.chi, a, cosmo->data.accelerator);
}

void ccl_comoving_radial_distances(ccl_cosmology * cosmo, int na, double a[na], double output[na])
{  
  if (!cosmo->computed_distances){
    ccl_cosmology_compute_distances(cosmo);
    ccl_check_status(cosmo);    
  }
  for (int i=0; i<na; i++){
    output[i]=gsl_spline_eval(cosmo->data.chi,a[i],cosmo->data.accelerator);
  }
}
//TODO: do this

double ccl_luminosity_distance(ccl_cosmology * cosmo, double a)
{
  if (!cosmo->computed_distances){
    ccl_cosmology_compute_distances(cosmo);
    ccl_check_status(cosmo);    
  }
    return ccl_comoving_radial_distance(cosmo, a) / a;
}
//TODO: this is not valid for curved cosmologies

void ccl_luminosity_distances(ccl_cosmology * cosmo, int na, double a[na], double output[na])
{
  if (!cosmo->computed_distances){
    ccl_cosmology_compute_distances(cosmo);
    ccl_check_status(cosmo);    
  }
  for (int i=0; i<na; i++){
    output[i]=gsl_spline_eval(cosmo->data.chi,a[i],cosmo->data.accelerator)/a[i];
  }
}
//TODO: do this

double ccl_growth_factor(ccl_cosmology * cosmo, double a)
{
  if (!cosmo->computed_growth){
    ccl_cosmology_compute_growth(cosmo);
    ccl_check_status(cosmo);    
  }
    return gsl_spline_eval(cosmo->data.growth, a, cosmo->data.accelerator);
}

void ccl_growth_factors(ccl_cosmology * cosmo, int na, double a[na], double output[na])
{
  if (!cosmo->computed_growth){
    ccl_cosmology_compute_growth(cosmo);
    ccl_check_status(cosmo);    
  }

  for (int i=0; i<na; i++){
    output[i]=gsl_spline_eval(cosmo->data.growth,a[i],cosmo->data.accelerator);
  }
}

double ccl_growth_factor_unnorm(ccl_cosmology * cosmo, double a)
{
  if (!cosmo->computed_growth){
    ccl_cosmology_compute_growth(cosmo);
    ccl_check_status(cosmo);    
  }
    return cosmo->data.growth0*gsl_spline_eval(cosmo->data.growth, a, cosmo->data.accelerator);
}

void ccl_growth_factors_unnorm(ccl_cosmology * cosmo, int na, double a[na], double output[na])
{
  if (!cosmo->computed_growth){
    ccl_cosmology_compute_growth(cosmo);
    ccl_check_status(cosmo);    
  }
  for (int i=0; i<na; i++){
    output[i]=cosmo->data.growth0*gsl_spline_eval(cosmo->data.growth,a[i],cosmo->data.accelerator);
  }
}
//TODO: do this

double ccl_growth_rate(ccl_cosmology * cosmo, double a)
{
  if (!cosmo->computed_growth){
    ccl_cosmology_compute_growth(cosmo);
    ccl_check_status(cosmo);    
  }
    return gsl_spline_eval(cosmo->data.fgrowth, a, cosmo->data.accelerator);
}

void ccl_growth_rates(ccl_cosmology * cosmo, int na, double a[na], double output[na])
{
  if (!cosmo->computed_growth){
    ccl_cosmology_compute_growth(cosmo);
    ccl_check_status(cosmo);    
  }
  for (int i=0; i<na; i++){
    output[i]=gsl_spline_eval(cosmo->data.fgrowth,a[i],cosmo->data.accelerator);
  }
}
