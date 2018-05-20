#include "ccl.h"
#include <stdio.h>
#include <math.h>
#include "ccl_params.h"

int main(int argc, char * argv[])
{
  int status=0;
  double Omega_c = 0.25;
  double Omega_b = 0.05;
  double h = 0.7;
  double A_s = 2.1e-9;
  double n_s = 0.96;
  double Omega_k = 0.;
  double Neff = 2.046;
  double Nmass = 1.;
  double mnu = 0.04;
  double k, p, Omnu;

  // Initialize the default configuration and set the transfer function method to use CLASS.
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_boltzmann;
  
  // Set parameters and initialize cosmology
  ccl_parameters params = ccl_parameters_create_lcdm_nu(Omega_c, Omega_b, Omega_k,h, A_s, n_s, Neff, Nmass, mnu, &status);
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  
  // Get Omega_nu(z) for massive neutrinos at a variety of redshifts ( / scale factor values).
  for (double z = 0.; z<=4.; z = z+0.2){
		Omnu = ccl_omega_x(cosmo, 1./ (1. + z), ccl_omega_nu_label, &status);
		printf("z=%.16le, Omnu(z)=%.16le \n",z,Omnu); 
	  
  } 	  
    
  // Get the power spectrum at a number of k values and output  
  for (double kpow = -4 ; kpow <=2; kpow = kpow+ 0.02){ 
		k = pow(10, kpow);
		// Get the nonlinear matter power spectrum today.
		p = ccl_nonlin_matter_power(cosmo, k, 1.0, &status); 
		printf("k=%.16le, P(k)=%.16le \n",k,p); 
  } 

  printf("Completed. Status = %d\n",status);

  return 0;

}
