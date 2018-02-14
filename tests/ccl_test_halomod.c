#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"
#include "ccl_lsst_specs.h"
#include "ccl_halomod.h"

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



int main(void)
{
  //status flag
  int status =0;

  // Initialize cosmological parameters
  ccl_configuration config=default_config;
  config.transfer_function_method=ccl_boltzmann_class;
  ccl_parameters params = ccl_parameters_create(OC, OB, OK, NREL, NMAS, MNU, W0, WA, HH, NORMPS, NS,-1,-1,-1,-1,NULL,NULL, &status);
  //printf("in sample run w0=%1.12f, wa=%1.12f\n", W0, WA);
  
  // Initialize cosmology object given cosmo params
  ccl_cosmology *cosmo=ccl_cosmology_create(params,config);

  // Compute radial distances (see include/ccl_background.h for more routines)
  printf("Comoving distance to z = %.3lf is chi = %.3lf Mpc\n",
	 ZD,ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status));
  
  printf("\nk P_NL P_halo\n");
  for (double k = 0.001; k <= 100.;k*=sqrt(10.)){
    double p_nl = ccl_nonlin_matter_power(cosmo,k, ZD,&status);
    double p_halo = p_1h(cosmo,k,1./(1+ZD),&status);
    printf("%e %e %e\n",k,p_nl,p_halo);
  } 

  
  //Always clean up!!
  ccl_cosmology_free(cosmo);
  
  return 0;
}
