#include "ccl.h"
#include <math.h>
#include <stdio.h>
#include "ccl_params.h"

int main(int argc, char * argv[])
{
  int status=0;
  double Omega_c = 0.25;
  double h = 0.7;
  double Omega_b = 0.022/h/h;
  double normp = 0.8; //2.1e-9
  double n_s = 0.96;
  double Omega_k = 0.;
  /*If you want to call the emulator with
   massive neutrinos, in order for the number
  of relativistic species in the early Universe
  to be 3.04, compatible with the emulator
  set-up, we need:
  */
  double Neff = 3.04;
  double mnu[3] = {0.02, 0.02, 0.02};
  ccl_mnu_convention mnutype = ccl_mnu_list;
  /*In the case of the emulator without massive
   neutrinos, we simply set: 
  */
  //double Neff = 3.04;
  //double mnu[1] = {0};
  
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_emulator;
  config.matter_power_spectrum_method = ccl_emu;

  ccl_parameters params = ccl_parameters_create_lcdm_nu(Omega_c, Omega_b, Omega_k,h, normp, n_s, Neff, mnu, mnutype, &status);
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

  //The linear power spectrum will be from CLASS
  //The NL power spectrum comes from the emulator
  printf("# k [1/Mpc],Plin(k,z=0),Plin(k,z=1),Plin(k,z=2),Pnl(k,z=0),Pnl(k,z=1),Pnl(k,z=2)\n");
  
  double k,p,p1,p2,pnl,p1nl,p2nl;//,p3=0;
  double a_at_z1=0.5;
  double a_at_z2=1./3.;
  for (k = 1.e-3; k<5.; k*=1.05){
    pnl = ccl_nonlin_matter_power(cosmo, k,1.0,&status);
    p1nl = ccl_nonlin_matter_power(cosmo,k, a_at_z1,&status);
    p2nl = ccl_nonlin_matter_power(cosmo,k, a_at_z2,&status);
    p = ccl_linear_matter_power(cosmo, k,1.0, &status);
    p1 = ccl_linear_matter_power(cosmo,k, a_at_z1,&status);
    p2 = ccl_linear_matter_power(cosmo,k, a_at_z2,&status);
    printf("%le %le %le %le %le %le %le\n", k, p,p1,p2,pnl,p1nl,p2nl);
  }

  return 0;
}
