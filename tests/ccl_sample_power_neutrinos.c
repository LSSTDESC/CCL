#include "ccl.h"
#include <stdio.h>
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
  double Neff = 3.046;
  double Nmass = 0;
  double mnu = 0.;
  FILE * input;
  FILE * output;
  char line[2000];
  double k, no;

  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_boltzmann;
  
  ccl_parameters params = ccl_parameters_create_lcdm_nu(Omega_c, Omega_b, Omega_k,h, A_s, n_s, Neff, Nmass, mnu, &status);
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

  // Open file to read k values from CLASS at which to compute things
  input = fopen("./fromCLASS_3massless_pk_nl.dat", "r");
  if(input==NULL) {
    fprintf(stderr,"Couldn't find benchmark file. Please execute this code from the \"tests\" directory\n");
    exit(1);
  }
  
  // Read the header 
  fgets(line,2000, input);
  fgets(line,2000, input);
  fgets(line,2000, input);
  fgets(line,2000, input);
  
  // Open file to output results
  output = fopen("./neutrinos_3massless_nl_pk_test.out", "w");
  while((fscanf(input, "%le %le\n", &k, &no)!=EOF)) { 
    if ((k<ccl_splines->K_MIN_DEFAULT) || (k>ccl_splines->K_MAX)) continue; 
    // Note CLASS k's are in h/Mpc but CCL takes k's in 1/Mpc, so we convert. 
    double p = ccl_nonlin_matter_power(cosmo, k * h , 1.0, &status); 
    // CCL outputs P(k) in Mpc^3, but we want to compare to class, which outputs in (Mpc/h)^3, so convert. Output is in k-> h / Mpc, P(k)-> (Mpc/h)^3.
    fprintf(output, "%.16le %.16le \n",k,p*h*h*h ); 
  } 

  printf("Completed. Status = %d\n",status);

  return 0;

}
