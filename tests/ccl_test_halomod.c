#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"
#include "ccl_lsst_specs.h"
#include "ccl_halomod.h"

//Test with a 'boring' cosmological model/
#define OC 0.25
#define OB 0.05
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define NORMPS 0.80
#define ZD 0.
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

int main(void){
  
  int status = 0; // status flag

  double a = 1./(1.+ZD); // scale factor

  FILE *fp; // File pointer

  int test_distance = 0;
  int test_basics = 0;
  int test_massfunc = 1;
  int test_nfw_wk = 0;
  int test_power = 0;

  // Initial white space
  printf("\n"); 

  // Initialize cosmological parameters
  ccl_configuration config=default_config;
  config.transfer_function_method=ccl_boltzmann_class;
  ccl_parameters params = ccl_parameters_create(OC, OB, OK, NREL, NMAS, MNU, W0, WA, HH, NORMPS, NS,-1,-1,-1,-1,NULL,NULL, &status);
  //printf("in sample run w0=%1.12f, wa=%1.12f\n", W0, WA);
  
  // Initialize the cosmology object given the cosmological parameters
  ccl_cosmology *cosmo=ccl_cosmology_create(params,config);

  //
  //Now to the tests
  //
  
  //Test the distance calculation
  if(test_distance==1){

    printf("Testing distance calculation\n");
    printf("\n");
    
    // Compute radial distances (see include/ccl_background.h for more routines)
    printf("Comoving distance to z = %.3lf is chi = %.3lf Mpc\n",
	   ZD,ccl_comoving_radial_distance(cosmo, a, &status));
    printf("\n");
  
  }

  //Test the basic halo-model function
  if(test_basics==1){

    //double dc, Dv;

    printf("Testing basics\n");
    printf("\n");

    //dc=delta_c();
    //Dv=delta_v();
    //print("delta_c: %f\n", dc);
    //print("delta_c: %f\n", Dv);
    //print("\n");
    
  }

  //Test the mass function
  if(test_massfunc==1){

    double m_min=1e10;
    double m_max=1e16;
    int nm=101;

    printf("Testing mass function\n");
    printf("\n");

    printf("M / Msun\t nu\t\t f(nu)\t\n");
    printf("=========================================\n");
    for (int i = 1; i <= nm; i++){
      double m = exp(log(m_min)+log(m_max/m_min)*((i-1.)/(nm-1.)));
      double n = nu(cosmo, m, a, &status); 
      //double gnu = massfunc_st(n);
      printf("%e\t %f\t %f\n", m, n, n);
    }
    printf("=========================================\n");
    
  }

  //Test the halo Fourier Transform
  if(test_nfw_wk==1){

    double kmin = 1e-3;
    double kmax = 1e2;
    int nk = 101;

    double c = 4.; //Halo concentration
    double m = 1e15; //Halo mass

    printf("Testing halo Fourier Transform\n");
    printf("\n");

    fp = fopen("Mead/CCL_Wk.dat", "w");
    
    for (int i = 1; i <= nk; i++){

      double k = exp(log(kmin)+log(kmax/kmin)*(i-1.)/(nk-1.));
      double wk = u_nfw_c(cosmo, c, m, k, a, &status);

      printf("%d\t %e\t %e\n", i, k, wk);
      fprintf(fp, "%e\t %e\n", k, wk);
      
    }

    fclose(fp);
    
  }  

  //Test the power spectrum calculation
  if(test_power==1){

    double kmin=1e-3;
    double kmax=100;
    int nk=101;

    printf("Testing power spectrum calculation");
    printf("\n");

    fp = fopen("Mead/CCL_power.dat", "w");
    
    printf("k\t\t P_lin\t\t P_NL\t\t P_halo\t\n");
    printf("=============================================================\n");    
    for (int i = 1; i <= nk; i++){

      double k = exp(log(kmin)+log(kmax/kmin)*(i-1.)/(nk-1.));
    
      double p_lin = ccl_linear_matter_power(cosmo, k, a, &status); // Linear spectrum
      double p_nl = ccl_nonlin_matter_power(cosmo, k, a, &status); // Non-linear spectrum (HALOFIT I think...)
      //double p_halo = p_1h(cosmo, k, a, &status); // Halo-model spectrum

      printf("%e\t %e\t %e\t %e\n", k, p_lin, p_nl, p_lin);
      fprintf(fp, "%e\t %e\t %e\t %e\n", k, p_lin, p_nl, p_lin);

    }
    printf("=============================================================\n");
    printf("\n");
    fclose(fp);
  
  }  
  
  // Always clean up the cosmology object!!
  ccl_cosmology_free(cosmo);

  // 0 is a successful return
  return 0;
}
