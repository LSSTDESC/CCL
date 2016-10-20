#include "ccl_background.h"
#include "ccl_utils.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "gsl/gsl_errno.h"
//#include "gsl/gsl_odeiv.h"
#include "gsl/gsl_odeiv2.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_roots.h"

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
	       exp(3*params->wa*(a-1))+params->Omega_k*a)/(a*a*a));
}

/* --------- ROUTINE: omega_m_z ---------
INPUT: scale factor, cosmological parameters
TASK: Compute Omega_m(z)
*/
static double omega_m_z(double a,ccl_parameters * params)
{
  return params->Omega_m/(params->Omega_m+params->Omega_l*pow(a,-3*(params->w0+params->wa))*
			  exp(3*params->wa*(a-1))+params->Omega_k*a);
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
static int growth_factor_and_growth_rate(double a,double *gf,double *fg,ccl_cosmology *cosmo)
{
  if(a<EPS_SCALEFAC_GROWTH) {
    *gf=a;
    *fg=1;
    return 0;
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
      fprintf(stderr,"ODE didn't converge when computing growth\n");
      return 1;
    }
    
    *gf=y[0];
    *fg=y[1]/(a*a*h_over_h0(a,&(cosmo->params))*y[0]);

    return 0;
  }
}

/* --------- ROUTINE: compute_chi ---------
INPUT: scale factor, cosmology
OUTPUT: chi -> radial comoving distance
TASK: compute radial comoving distance at a
*/
static int compute_chi(double a,ccl_cosmology *cosmo,double *chi)
{
  int status;
  double result;
  gsl_integration_cquad_workspace * workspace = gsl_integration_cquad_workspace_alloc (1000);
  gsl_function F;
  F.function = &chi_integrand;
  F.params = cosmo;
  //TODO: CQUAD is great, but slower than other methods. This could be sped up if it becomes an issue.
  status=gsl_integration_cquad(&F, a, 1.0, 0.0,EPSREL_DIST,workspace,&result, NULL, NULL); 
  *chi=result/cosmo->params.h;
  gsl_integration_cquad_workspace_free(workspace);
  return status;
}

//Root finding for a(chi)
typedef struct {
  double chi;
  ccl_cosmology *cosmo;
} Fpar;

static double fzero(double a,void *params)
{
  double chi,chia,a_use=a;
  chi=((Fpar *)params)->chi;
  compute_chi(a_use,((Fpar *)params)->cosmo,&chia);

  return chi-chia;
}

static double dfzero(double a,void *params)
{
  ccl_cosmology *cosmo=((Fpar *)params)->cosmo;
  
  return chi_integrand(a,cosmo)/cosmo->params.h;
}

static void fdfzero(double a,void *params,double *f,double *df)
{
  *f=fzero(a,params);
  *df=dfzero(a,params);
}

static double a_of_chi(double chi,ccl_cosmology *cosmo,double *a_old,gsl_root_fdfsolver *s)
{
  if(chi==0)
    return 1.;
  else {
    Fpar p;
    gsl_function_fdf FDF;
    double a_previous,a_current=*a_old;

    p.cosmo=cosmo;
    p.chi=chi;
    FDF.f=&fzero;
    FDF.df=&dfzero;
    FDF.fdf=&fdfzero;
    FDF.params=&p;
    gsl_root_fdfsolver_set(s,&FDF,a_current);

    int iter=0,status;
    do {
      iter++;
      status=gsl_root_fdfsolver_iterate(s);
      a_previous=a_current;
      a_current=gsl_root_fdfsolver_root(s);
      status=gsl_root_test_delta(a_current,a_previous,1E-6,0);
    } while(status==GSL_CONTINUE);

    *a_old=a_current;
    return a_current;
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

  // Create linearly-spaced values of the scale factor
  int na=0;
  double * a = ccl_linear_spacing(A_SPLINE_MIN, A_SPLINE_MAX, A_SPLINE_DELTA, &na);
  if (a==NULL || 
      (fabs(a[0]-A_SPLINE_MIN)>1e-5) || 
      (fabs(a[na-1]-A_SPLINE_MAX)>1e-5) || 
      (a[na-1]>1.0)
      ) {
    fprintf(stderr, "Error creating linear spacing.\n");        
    *status = 1;
    return;
  }

  // allocate space for y, which will be all three
  // of E(a), chi(a), D(a) and f(a) in turn.
  double *y = malloc(sizeof(double)*na); //TODO: check for bad allocation

  // Fill in E(a)
  for (int i=0; i<na; i++){
    y[i] = h_over_h0(a[i], &cosmo->params);
  }

  // Allocate and fill E spline with values we just got
  gsl_spline * E = gsl_spline_alloc(A_SPLINE_TYPE, na);
  *status = gsl_spline_init(E, a, y, na);
  // Check for errors in creating the spline
  if (*status){
    free(a);
    free(y);
    gsl_spline_free(E);
    fprintf(stderr, "Error creating E(a) spline\n");
    return;
  }

  //Fill in chi(a)
  for (int i=0; i<na; i++)
    *status |= compute_chi(a[i],cosmo,&(y[i]));
  if (*status){
    free(a);
    free(y);
    gsl_spline_free(E);        
    fprintf(stderr, "Error integrating to get chi(a)\n");
    return;
  }

  gsl_spline * chi = gsl_spline_alloc(A_SPLINE_TYPE, na);
  *status = gsl_spline_init(chi, a, y, na); //in Mpc
  if (*status){
    free(a);
    free(y);
    gsl_spline_free(E);
    gsl_spline_free(chi);
    fprintf(stderr, "Error creating chi(a) spline\n");
    return;
  }

  //Spline for a(chi)
  double dchi=3.,chi0=y[na-1],chif=y[0],a0=a[na-1],af=a[0];
  //TODO: The interval in chi (3. Mpc) should probably be made a macro
  free(y); free(a);
  na=(int)((chif-chi0)/dchi); dchi=(chif-chi0)/na; na=0;
  y=ccl_linear_spacing(chi0,chif,dchi,&na);
  if(y==NULL || (fabs(y[0]-chi0)>1E-5) || (fabs(y[na-1]-chif)>1e-5)) {
    fprintf(stderr,"Error creating linear spacing\n");
    *status=1;
    gsl_spline_free(E);
    gsl_spline_free(chi);
    return;
  }
  a=malloc(sizeof(double)*na);
  if(a==NULL) {
    fprintf(stderr,"Out of memory\n");
    *status=1;
    free(y);
    gsl_spline_free(E);
    gsl_spline_free(chi);
    return;
  }
  a[0]=a0; a[na-1]=af;
  const gsl_root_fdfsolver_type *T=gsl_root_fdfsolver_newton;
  gsl_root_fdfsolver *s=gsl_root_fdfsolver_alloc(T);
  for(int i=1;i<na-1;i++) {
    a[i]=a_of_chi(y[i],cosmo,&a0,s);
  }
  gsl_root_fdfsolver_free(s);
  //TODO: check for errors in solver
  gsl_spline * achi=gsl_spline_alloc(A_SPLINE_TYPE,na);
  *status=gsl_spline_init(achi,y,a,na);
  if(*status) {
    fprintf(stderr,"Out of memory\n");
    *status=1;
    free(y);
    free(a);
    gsl_spline_free(E);
    gsl_spline_free(chi);
    gsl_spline_free(achi);
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

void ccl_cosmology_compute_growth(ccl_cosmology * cosmo, int *status)
{
  if(cosmo->computed_growth)
    return;

  // Create linearly-spaced values of the scale factor
  int na=0;
  double * a = ccl_linear_spacing(A_SPLINE_MIN, A_SPLINE_MAX, A_SPLINE_DELTA, &na);
  if (a==NULL || 
      (fabs(a[0]-A_SPLINE_MIN)>1e-5) || 
      (fabs(a[na-1]-A_SPLINE_MAX)>1e-5) || 
      (a[na-1]>1.0)
      ) {
    fprintf(stderr, "Error creating linear spacing.\n");        
    *status = 1;
    return;
  }

  gsl_integration_cquad_workspace * workspace=NULL;
  gsl_function F;
  gsl_spline *df_a_spline=NULL;
  if(cosmo->params.has_mgrowth) {
    double *df_arr=malloc(na*sizeof(double));
    //Generate spline for Delta f(z) that we will then interpolate into an array of a
    gsl_spline *df_z_spline=gsl_spline_alloc(A_SPLINE_TYPE,cosmo->params.nz_mgrowth);
    *status=gsl_spline_init(df_z_spline,cosmo->params.z_mgrowth,cosmo->params.df_mgrowth,
			    cosmo->params.nz_mgrowth);
    if(*status) {
      free(a);
      free(df_arr);
      gsl_spline_free(df_z_spline);
      fprintf(stderr,"Error creating MG spline\n");
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
    *status=gsl_spline_init(df_a_spline,a,df_arr,na);
    if (*status){
      free(a);
      free(df_arr);
      gsl_spline_free(df_a_spline);
      fprintf(stderr, "Error creating MG spline\n");
      return;
    }
    free(df_arr);

    workspace=gsl_integration_cquad_workspace_alloc(1000);
    F.function=&df_integrand;
    F.params=df_a_spline;
  }

  // allocate space for y, which will be all three
  // of D(a) and f(a) in turn.
  double growth0,fgrowth0;
  double *y = malloc(sizeof(double)*na);
  double *y2 = malloc(sizeof(double)*na);
  growth_factor_and_growth_rate(1.,&growth0,&fgrowth0,cosmo);
  for (int i=0; i<na; i++){
    *status |= growth_factor_and_growth_rate(a[i],&(y[i]),&(y2[i]),cosmo);
    if(cosmo->params.has_mgrowth) {
      if(a[i]>0) {
	double df,integ;
	//Add modification to f
	df=gsl_spline_eval(df_a_spline,a[i],NULL);
	y2[i]+=df;
	//Multiply D by exp(-int(df))
	*status |= gsl_integration_cquad(&F,a[i],1.0,0.0,EPSREL_DIST,workspace,&integ,NULL,NULL);
	y[i]*=exp(-integ);
      }
    }
    y[i]/=growth0;
  }
  if (*status){
    free(a);
    free(y);
    free(y2);
    if(df_a_spline!=NULL)
      gsl_spline_free(df_a_spline);
    if(workspace!=NULL)
      gsl_integration_cquad_workspace_free(workspace);
    fprintf(stderr, "Error creating growth array\n");
    return;
  }
  if(cosmo->params.has_mgrowth) {
    gsl_spline_free(df_a_spline);
    gsl_integration_cquad_workspace_free(workspace);
  }

  gsl_spline * growth = gsl_spline_alloc(A_SPLINE_TYPE, na);
  *status = gsl_spline_init(growth, a, y, na);
  if (*status){
    free(a);
    free(y);
    free(y2);
    gsl_spline_free(growth);
    fprintf(stderr, "Error creating growth spline\n");
    return;
  }
  gsl_spline * fgrowth = gsl_spline_alloc(A_SPLINE_TYPE, na);
  *status = gsl_spline_init(fgrowth, a, y2, na);
  if (*status){
    free(a);
    free(y);
    free(y2);
    gsl_spline_free(growth);
    gsl_spline_free(fgrowth);
    fprintf(stderr, "Error creating growth spline\n");
    return;
  }

  // Initialize the accelerator which speeds the splines and 
  // assign all the splines we've just made to the structure.
  if(cosmo->data.accelerator==NULL)
    cosmo->data.accelerator=gsl_interp_accel_alloc();
  if(cosmo->data.accelerator_achi==NULL)
    cosmo->data.accelerator_achi=gsl_interp_accel_alloc();
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

double ccl_h_over_h0(ccl_cosmology * cosmo, double a)
{
   return gsl_spline_eval(cosmo->data.E, a, cosmo->data.accelerator);
}

int ccl_h_over_h0s(ccl_cosmology * cosmo, int na, double a[na], double output[na])
{
  for (int i=0; i<na; i++){
    output[i]=gsl_spline_eval(cosmo->data.E,a[i],cosmo->data.accelerator);
  }

  return 0;
}

// Distance-like function examples, all in Mpc

double ccl_comoving_radial_distance(ccl_cosmology * cosmo, double a)
{
   return gsl_spline_eval(cosmo->data.chi, a, cosmo->data.accelerator);
}

int ccl_comoving_radial_distances(ccl_cosmology * cosmo, int na, double a[na], double output[na])
{
  for (int i=0; i<na; i++){
    output[i]=gsl_spline_eval(cosmo->data.chi,a[i],cosmo->data.accelerator);
  }

  return 0;
}
//TODO: do this

double ccl_luminosity_distance(ccl_cosmology * cosmo, double a)
{
    return ccl_comoving_radial_distance(cosmo, a) / a;
}
//TODO: this is not valid for curved cosmologies

int ccl_luminosity_distances(ccl_cosmology * cosmo, int na, double a[na], double output[na])
{
  for (int i=0; i<na; i++){
    output[i]=gsl_spline_eval(cosmo->data.chi,a[i],cosmo->data.accelerator)/a[i];
  }

  return 0;
}
//TODO: do this

double ccl_growth_factor(ccl_cosmology * cosmo, double a, int * status)
{
    return gsl_spline_eval(cosmo->data.growth, a, cosmo->data.accelerator);
}

int ccl_growth_factors(ccl_cosmology * cosmo, int na, double a[na], double output[na])
{
  for (int i=0; i<na; i++){
    output[i]=gsl_spline_eval(cosmo->data.growth,a[i],cosmo->data.accelerator);
  }

  return 0;
}

double ccl_growth_factor_unnorm(ccl_cosmology * cosmo, double a, int * status)
{
    return cosmo->data.growth0*gsl_spline_eval(cosmo->data.growth, a, cosmo->data.accelerator);
}

int ccl_growth_factors_unnorm(ccl_cosmology * cosmo, int na, double a[na], double output[na])
{
  for (int i=0; i<na; i++){
    output[i]=cosmo->data.growth0*gsl_spline_eval(cosmo->data.growth,a[i],cosmo->data.accelerator);
  }

  return 0;
}
//TODO: do this

double ccl_growth_rate(ccl_cosmology * cosmo, double a, int * status)
{
    return gsl_spline_eval(cosmo->data.fgrowth, a, cosmo->data.accelerator);
}

int ccl_growth_rates(ccl_cosmology * cosmo, int na, double a[na], double output[na])
{
  for (int i=0; i<na; i++){
    output[i]=gsl_spline_eval(cosmo->data.fgrowth,a[i],cosmo->data.accelerator);
  }

  return 0;
}
//TODO: do this

//Scale factor for a given distance
double ccl_scale_factor_of_chi(ccl_cosmology * cosmo, double chi)
{
  return gsl_spline_eval(cosmo->data.achi, chi,cosmo->data.accelerator_achi);
}

int ccl_scale_factor_of_chis(ccl_cosmology * cosmo, int nchi, double chi[nchi], double output[nchi])
{
  for (int i=0; i<nchi; i++) {
    output[i]=gsl_spline_eval(cosmo->data.achi,chi[i],cosmo->data.accelerator_achi);
  }

  return 0;
}
