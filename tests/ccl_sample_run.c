#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"
#include "ccl_lsst_specs.h"

#define OC 0.25
#define OB 0.05
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define NORMPS 0.80
#define ZD 0.5
#define NZ 128
#define Z0_GC 0.50
#define SZ_GC 0.05
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 512
#define PS 0.1 
#define NREL 3.046
#define NMAS 0
#define MNU 0.0



// The user defines a structure of parameters to the user-defined function for the photo-z probability 
struct user_func_params
{
  double (* sigma_z) (double);
};

// The user defines a function of the form double function ( z_ph, z_spec, void * user_pz_params) where user_pz_params is a pointer to the parameters of the user-defined function. This returns the probabilty of obtaining a given photo-z given a particular spec_z.
double user_pz_probability(double z_ph, double z_s, void * user_par, int * status)
{
  double sigma_z = ((struct user_func_params *) user_par)->sigma_z(z_s);
  return exp(- (z_ph-z_s)*(z_ph-z_s) / (2.*sigma_z*sigma_z)) / (pow(2.*M_PI,0.5)*sigma_z);
}

int main(int argc,char **argv)
{
  //status flag
  int status =0;
  // Initialize cosmological parameters
  ccl_configuration config=default_config;
  config.transfer_function_method=ccl_boltzmann_class;
  //ccl_parameters params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,NAN,NS,-1,NULL,NULL);
  ccl_parameters params = ccl_parameters_create(OC, OB, OK, NREL, NMAS, MNU, W0, WA, HH, NORMPS, NS,0,NULL,NULL, &status);
  //printf("in sample run w0=%1.12f, wa=%1.12f\n", W0, WA);
  
  // Initialize cosmology object given cosmo params
  ccl_cosmology *cosmo=ccl_cosmology_create(params,config);
  // Compute radial distances (see include/ccl_background.h for more routines)
  printf("Comoving distance to z = %.3lf is chi = %.3lf Mpc\n",
	 ZD,ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status));
  printf("Luminosity distance to z = %.3lf is chi = %.3lf Mpc\n",
	 ZD,ccl_luminosity_distance(cosmo,1./(1+ZD), &status));
  printf("Distance modulus to z = %.3lf is mu = %.3lf Mpc\n",
	 ZD,ccl_distance_modulus(cosmo,1./(1+ZD), &status));
  
  
  //Consistency check
  printf("Scale factor is a=%.3lf \n",1./(1+ZD));
  printf("Consistency: Scale factor at chi=%.3lf Mpc is a=%.3lf\n",
	 ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status),
	 ccl_scale_factor_of_chi(cosmo,ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status), &status));
  
  // Compute growth factor and growth rate (see include/ccl_background.h for more routines)
  printf("Growth factor and growth rate at z = %.3lf are D = %.3lf and f = %.3lf\n",
	 ZD, ccl_growth_factor(cosmo,1./(1+ZD), &status),ccl_growth_rate(cosmo,1./(1+ZD), &status));
  
  // Compute sigma_8
  printf("Initializing power spectrum...\n");
  printf("sigma_8 = %.3lf\n\n", ccl_sigma8(cosmo, &status));
  
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
  CCL_ClTracer *ct_gc=ccl_cl_tracer_number_counts_simple_new(cosmo,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr, &status);
  
  //Cosmic shear tracer
  CCL_ClTracer *ct_wl=ccl_cl_tracer_lensing_simple_new(cosmo,NZ,z_arr_sh,nz_arr_sh, &status);
  printf("ell C_ell(g,g) C_ell(g,s) C_ell(s,s) | r(g,s)\n");
  for(int l=2;l<=NL;l*=2) {
    double cl_gg=ccl_angular_cl(cosmo,l,ct_gc,ct_gc, &status); //Galaxy-galaxy
    double cl_gs=ccl_angular_cl(cosmo,l,ct_gc,ct_wl, &status); //Galaxy-lensing
    double cl_ss=ccl_angular_cl(cosmo,l,ct_wl,ct_wl, &status); //Lensing-lensing
    printf("%d %.3lE %.3lE %.3lE | %.3lE\n",l,cl_gg,cl_gs,cl_ss,cl_gs/sqrt(cl_gg*cl_ss));
  }
  printf("\n");
  
  //Free up tracers
  ccl_cl_tracer_free(ct_gc);
  ccl_cl_tracer_free(ct_wl);
  
  //Halo mass function
  printf("M\tdN/dlog10M(z = 0, 0.5, 1))\n");
  for(int logM=9;logM<=15;logM+=1) {
    printf("%.1e\t",pow(10,logM));
    for(double z=0; z<=1; z+=0.5) {
      printf("%e\t", ccl_massfunc(cosmo, pow(10,logM),1.0/(1.0+z), 200., &status));
    }
    printf("\n");
  }
  printf("\n");
  
  //Halo bias
  printf("Halo bias: z, M, b1(M,z)\n");
  for(int logM=9;logM<=15;logM+=1) {
    for(double z=0; z<=1; z+=0.5) {
      printf("%.1e %.1e %.2e\n",1.0/(1.0+z),pow(10,logM),ccl_halo_bias(cosmo,pow(10,logM),1.0/(1.0+z), 200., &status));
    }
  }
  printf("\n");
  
  // LSST Specification
  // The user declares and sets an instance of parameters to their photo_z function:
  struct user_func_params my_params_example;
  my_params_example.sigma_z = ccl_specs_sigmaz_sources;
  
  // Declare a variable of the type of user_pz_info to hold the struct to be created.
  user_pz_info * pz_info_example;
  
  // Create the struct to hold the user information about photo_z's.
  pz_info_example = ccl_specs_create_photoz_info(&my_params_example, &user_pz_probability); 
  
  double z_test;
  double dNdz_tomo;
  int z;
  FILE * output;
  
  //Try splitting dNdz (lensing) into 5 redshift bins
  double tmp1,tmp2,tmp3,tmp4,tmp5;
  printf("Trying splitting dNdz (lensing) into 5 redshift bins. Output written into file tests/specs_example_tomo_lens.out\n");
  output = fopen("./tests/specs_example_tomo_lens.out", "w"); 
  
  if(!output) {
    fprintf(stderr, "Could not write to 'tests' subdirectory - please run this program from the main CCL directory\n");
    exit(1);
  }
  status = 0;
  for (z=0; z<100; z=z+1) {
    z_test = 0.035*z;
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,6., pz_info_example,&dNdz_tomo,&status); 
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,0.6, pz_info_example,&tmp1,&status); 
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.6,1.2, pz_info_example,&tmp2,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.2,1.8, pz_info_example,&tmp3,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.8,2.4, pz_info_example,&tmp4,&status); 
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 2.4,3.0, pz_info_example,&tmp5,&status);
    fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
  }
  
  fclose(output);
  
  //Try splitting dNdz (clustering) into 5 redshift bins
  printf("Trying splitting dNdz (clustering) into 5 redshift bins. Output written into file tests/specs_example_tomo_clu.out\n");
  output = fopen("./tests/specs_example_tomo_clu.out", "w");     
  for (z=0; z<100; z=z+1) {
    z_test = 0.035*z;
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.,6., pz_info_example,&dNdz_tomo,&status); 
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.,0.6, pz_info_example,&tmp1,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.6,1.2, pz_info_example,&tmp2,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC,1.2,1.8, pz_info_example,&tmp3,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC,1.8,2.4, pz_info_example,&tmp4,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC,2.4,3.0, pz_info_example,&tmp5,&status);
    fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
  }
  printf("ccl_sample_run completed, status = %d\n",status);
  fclose(output);
  
  //Free up photo-z info
  ccl_specs_free_photoz_info(pz_info_example);
  
  //Always clean up!!
  ccl_cosmology_free(cosmo);
  
  return 0;
}
