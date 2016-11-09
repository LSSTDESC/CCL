#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"
#include "ccl_correlation.h"

#define OC 0.25
#define OB 0.05
#define OL 0.70
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define AS 2.1E-9
#define ZD 0.5
#define NZ 128
#define Z0_GC 0.50
#define SZ_GC 0.05
#define Z0_SH 0.65
#define SZ_SH 0.05


int main(int argc,char **argv)
{
  ccl_parameters params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,AS,NS,-1,NULL,NULL);
  ccl_cosmology *cosmo=ccl_cosmology_create(params,default_config);

  //Create tracers for angular power spectra
  double z_arr_gc[NZ],z_arr_sh[NZ],nz_arr_gc[NZ],nz_arr_sh[NZ],bz_arr[NZ];
  for(int i=0;i<NZ;i++) {
    z_arr_gc[i]=Z0_GC-5*SZ_GC+10*SZ_GC*(i+0.5)/NZ;
    nz_arr_gc[i]=exp(-0.5*pow((z_arr_gc[i]-Z0_GC)/SZ_GC,2));
    bz_arr[i]=1+z_arr_gc[i];
    z_arr_sh[i]=Z0_SH-5*SZ_SH+10*SZ_SH*(i+0.5)/NZ;
    nz_arr_sh[i]=exp(-0.5*pow((z_arr_sh[i]-Z0_SH)/SZ_SH,2));
  }
  static int n_theta=10;
  double *theta=(double *)malloc(sizeof(double)*n_theta);

  double d_theta=1;
  double theta_min=1;
  for(int i=0;i<n_theta;i++)
    {
      theta[i]=theta_min;
      theta_min+=d_theta;
    }

  //Galaxy clustering tracer
  CCL_ClTracer *ct_gc=ccl_cl_tracer_new(cosmo,CL_TRACER_NC,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr);
  //Cosmic shear tracer
  CCL_ClTracer *ct_wl=ccl_cl_tracer_new(cosmo,CL_TRACER_WL,NZ,z_arr_sh,nz_arr_sh,-1,NULL,NULL);
  printf("ell C_ell(g,g) C_ell(g,s) C_ell(s,s) | r(g,s)\n");

  double *clustering_corr=(double *)malloc(sizeof(double)*n_theta);
  int i_bessel=0;
  ccl_tracer_corr(cosmo,n_theta,theta,ct_gc,ct_gc,i_bessel,clustering_corr); //clustering

  //prinf("theta,corr",theta,clustering_corr);

  //Free up tracers
  ccl_cl_tracer_free(ct_gc);
  ccl_cl_tracer_free(ct_wl);

  //Free up cosmology
  ccl_cosmology_free(cosmo);
  return 0;
}
