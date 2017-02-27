#include "ccl.h"
#include <stdio.h>

int main(int argc, char * argv[])
{
  int status=0;
  double Omega_c = 0.25;
  double Omega_b = 0.05;
  double h = 0.7;
  double A_s = 2.1e-9;
  double n_s = 0.96;
  double Omega_k = 0.;
  double Neff = 0.;
  double Nmass = 1;
  double mnu = 0.5;
  ccl_configuration config = default_config;
  //	config.transfer_function_method = ccl_bbks;
  config.transfer_function_method = ccl_boltzmann;
  
  ccl_parameters params = ccl_parameters_create_lcdm_nu(Omega_c, Omega_b, Omega_k,h, A_s, n_s, Neff, Nmass, mnu);
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  
  ccl_cosmology_compute_power(cosmo,&status);
  
  printf("# k [1/Mpc] P_lin(k,z=0) P_nl(k,z=0)\n");
  
  for (double k = 1e-3; k<1e3; k*=1.05){
    double p = ccl_linear_matter_power(cosmo, 1.0, k,&status);
    double pln = ccl_nonlin_matter_power(cosmo, 1.0, k,&status);
    printf("%le    %le %le\n", k, p,pln);
  }
  
  printf("sigma_8 = %.6lE\n", ccl_sigmaR(cosmo,8./h));

  printf("Completed. Status = %d\n",status);

  return 0;

}
