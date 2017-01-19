#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"

#define OC 0.25
#define OB 0.05
#define OL 0.70
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define S8 0.80
#define AS 2.1E-9
#define ZD 0.5
#define NZ 128
#define Z0_GC 0.50
#define SZ_GC 0.05
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 500

int main(int argc,char **argv)
{
  //Initialize cosmological parameters
  ccl_configuration config=default_config;
  config.transfer_function_method=ccl_bbks;
  ccl_parameters params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,AS,NS,-1,NULL,NULL);
  params.sigma_8=S8;
  //Initialize cosmology object given cosmo params
  ccl_cosmology *cosmo=ccl_cosmology_create(params,config);
  
  //Compute radial distance (see include/ccl_background.h for more routines)
  printf("Comoving distance to z=%.3lf is chi=%.3lf Mpc\n",ZD,ccl_comoving_radial_distance(cosmo,1./(1+ZD)));
  //Consistency check
  printf("Scale factor at chi=%.3lf Mpc is a=%.3lf Mpc\n",
	 ccl_comoving_radial_distance(cosmo,1./(1+ZD)),
	 ccl_scale_factor_of_chi(cosmo,ccl_comoving_radial_distance(cosmo,1./(1+ZD))));
  //Compute growth (see include/ccl_background.h for more routines)
  printf("Growth factor and growth rate at z=%.3lf are D=%.3lf and f=%.3lf\n",ZD,
	 ccl_growth_factor(cosmo,1./(1+ZD)),ccl_growth_rate(cosmo,1./(1+ZD)));

  //Compute sigma_8
  printf("sigma_8 = %.3lf, (consistency check: %.3lf)\n",
	 ccl_sigma8(cosmo),ccl_sigmaR(cosmo,8/HH));

  //Create tracers for angular power spectra
  double z_arr_gc[NZ],z_arr_sh[NZ],nz_arr_gc[NZ],nz_arr_sh[NZ],bz_arr[NZ];
  for(int i=0;i<NZ;i++) {
    z_arr_gc[i]=Z0_GC-5*SZ_GC+10*SZ_GC*(i+0.5)/NZ;
    nz_arr_gc[i]=exp(-0.5*pow((z_arr_gc[i]-Z0_GC)/SZ_GC,2));
    bz_arr[i]=1+z_arr_gc[i];
    z_arr_sh[i]=Z0_SH-5*SZ_SH+10*SZ_SH*(i+0.5)/NZ;
    nz_arr_sh[i]=exp(-0.5*pow((z_arr_sh[i]-Z0_SH)/SZ_SH,2));
  }
  
  //Galaxy clustering tracer
  CCL_ClTracer *ct_gc=ccl_cl_tracer_new(cosmo,CL_TRACER_NC,0,0,0,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr,0,NULL,NULL,0,NULL,NULL,0,NULL,NULL);

  //Cosmic shear tracer
  CCL_ClTracer *ct_wl=ccl_cl_tracer_new(cosmo,CL_TRACER_WL,0,0,0,NZ,z_arr_sh,nz_arr_sh,0,NULL,NULL,0,NULL,NULL,0,NULL,NULL,0,NULL,NULL);
  
  printf("ell C_ell(g,g) C_ell(g,s) C_ell(s,s) r(g,s)\n");
  for(int l=2;l<NL;l+=10) {
    double cl_gg=ccl_angular_cl(cosmo,l,ct_gc,ct_gc); //Galaxy-galaxy
    double cl_gs=ccl_angular_cl(cosmo,l,ct_gc,ct_wl); //Galaxy-lensing
    double cl_ss=ccl_angular_cl(cosmo,l,ct_wl,ct_wl); //Lensing-lensing
    printf("%d %.3lE %.3lE %.3lE | %.3lE\n",l,cl_gg,cl_gs,cl_ss,cl_gs/sqrt(cl_gg*cl_ss));
  }
  //Free up tracers
  ccl_cl_tracer_free(ct_gc);
  ccl_cl_tracer_free(ct_wl);

  //Free up cosmology
  ccl_cosmology_free(cosmo);

  return 0;
}
