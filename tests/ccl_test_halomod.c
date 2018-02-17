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
#define ZD 0.0
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

  int test_distance = 1;
  int test_basics = 1;
  //int test_mass_function = 0;
  int test_halo_properties = 1;
  int test_nfw_wk = 1;
  int test_power = 1;

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

    printf("Testing basics\n");
    printf("\n");

    double dc=delta_c();
    double Dv=Delta_v();
    printf("delta_c: %f\n", dc);
    printf("Delta_v: %f\n", Dv);
    printf("\n");
    
  }

  /*
  //Test mass function
  if(test_mass_function==1){

  double m_min=1e10;
  double m_max=1e16;
  int nm=101;

  printf("Testing mass function\n");
  printf("\n");
  
  printf("M / Msun\t nu\t\t g(nu)\t\n");
  printf("=========================================\n");
  for (int i = 1; i <= nm; i++){
    double m = exp(log(m_min)+log(m_max/m_min)*((i-1.)/(nm-1.)));
    double n = nu(cosmo, m, a, &status);
    double gnu = massfunc_st(n);
    printf("%e\t %f\t %f\n", m, n, gnu);
  }
  printf("=========================================\n");
  printf("\n");
  
  }
  */

  //Test halo properties
  if(test_halo_properties==1){

    double m_min=1e10;
    double m_max=1e16;
    int nm=101;

    printf("Testing halo properties\n");
    printf("\n");

    printf("M / Msun\t nu\t\t r_vir / Mpc\t r_Lag / Mpc\t conc\t\n");
    printf("==========================================================================\n");
    for (int i = 1; i <= nm; i++){
      double m = exp(log(m_min)+log(m_max/m_min)*((i-1.)/(nm-1.)));
      double n = nu(cosmo, m, a, &status); 
      double r_vir = r_delta(cosmo, m, a, &status);
      double r_lag = r_Lagrangian(cosmo, m, a, &status);
      double conc = ccl_halo_concentration(cosmo, m, a, &status);
      printf("%e\t %f\t %f\t %f\t %f\n", m, n, r_vir, r_lag, conc);
    }
    printf("==========================================================================\n");
    printf("\n");
    
  }

  //Test the halo Fourier Transform
  if(test_nfw_wk==1){

    //k range and number of points in k
    double kmin = 1e-3;
    double kmax = 1e2;
    int nk = 101;

    //double c = 4.; //Halo concentration
    double m = 1e15; //Halo mass in Msun

    printf("Testing halo Fourier Transform\n");
    printf("\n");
    printf("Halo mass [Msun]: %e\n", m);
    double c=ccl_halo_concentration(cosmo, m, a, &status);
    printf("Halo concentration: %f\n", c);
    printf("\n");

    fp = fopen("Mead/CCL_Wk.dat", "w");

    printf("k / Mpc^-1\t Wk\t\n");
    printf("=============================\n");
    for (int i = 1; i <= nk; i++){

      double k = exp(log(kmin)+log(kmax/kmin)*(i-1.)/(nk-1.));
      double wk = u_nfw_c(cosmo, c, m, k, a, &status);

      printf("%e\t %e\n", k, wk);
      fprintf(fp, "%e\t %e\n", k, wk);
      
    }
    printf("=============================\n");

    fclose(fp);
    printf("\n");
    
    
  }  

  //Test the power spectrum calculation
  if(test_power==1){

    double kmin=1e-3;
    double kmax=100;
    int nk=101;

    printf("Testing power spectrum calculation\n");
    printf("\n");

    fp = fopen("Mead/CCL_power.dat", "w");
  
    printf("k\t\t P_lin\t\t P_NL\t\t P_2h\t\t P_1h\t\t P_halo\t\n");
    printf("=============================================================================================\n");    
    for (int i = 1; i <= nk; i++){

      double k = exp(log(kmin)+log(kmax/kmin)*(i-1.)/(nk-1.));
    
      double p_lin = ccl_linear_matter_power(cosmo, k, a, &status); // Linear spectrum
      double p_nl = ccl_nonlin_matter_power(cosmo, k, a, &status); // Non-linear spectrum (HALOFIT I think...)
      double p_twohalo = p_2h(cosmo, k, a, &status); // Two-halo power
      double p_onehalo = p_1h(cosmo, k, a, &status); // One-halo power      
      double p_full = p_halomod(cosmo, k, a, &status); // Full halo-model power

      printf("%e\t %e\t %e\t %e\t %e\t %e\n", k, p_lin, p_nl, p_twohalo, p_onehalo, p_full);
      fprintf(fp, "%e\t %e\t %e\t %e\t %e\t %e\n", k, p_lin, p_nl, p_twohalo, p_onehalo, p_full);

    }
    printf("=============================================================================================\n");
    printf("\n");
    fclose(fp);
  
  }  
  
  // Always clean up the cosmology object!!
  ccl_cosmology_free(cosmo);

  // 0 is a successful return
  return 0;
}
