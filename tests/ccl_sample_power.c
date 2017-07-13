#include "ccl.h"
#include <math.h>
#include <stdio.h>
#include "ccl_params.h"

int main(int argc, char * argv[])
{
  int status=0;
  double Omega_c = 0.25;
  double Omega_b = 0.05;
  double h = 0.7;
  double normp = 0.8; //2.1e-9
  double n_s = 0.96;

  ccl_configuration config = default_config;
  //config.transfer_function_method = ccl_bbks;
  config.matter_power_spectrum_method=ccl_linear;

  ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, normp, n_s, &status);
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

  printf("# k [1/Mpc],P(k,z=0),P(k,z=1),P(k,z=2),P(k,z=3)\n");
  
  double k,p,p1,p2,p3=0;
  double a_at_z1=0.5;
  double a_at_z2=1./3.;
  double a_at_z3=0.25;
  if(cosmo->config.matter_power_spectrum_method==ccl_linear) {
    for (k = ccl_splines->K_MIN_DEFAULT; k<ccl_splines->K_MAX; k*=1.05) {
      p = ccl_linear_matter_power(cosmo, k,1.0, &status);
      p1 = ccl_linear_matter_power(cosmo,k, a_at_z1,&status);
      p2 = ccl_linear_matter_power(cosmo,k, a_at_z2,&status);
      p3 = ccl_linear_matter_power(cosmo,k, a_at_z3,&status);
      printf("%le %le %le %le %le\n", k, p,p1,p2,p3);
    }
  }
  else {
    if(cosmo->config.matter_power_spectrum_method==ccl_halofit) {
      for (k = ccl_splines->K_MIN_DEFAULT; k<ccl_splines->K_MAX; k*=1.05) {
	p = ccl_nonlin_matter_power(cosmo, k,1.0,&status);
	p1 = ccl_nonlin_matter_power(cosmo,k, a_at_z1,&status);
	p2 = ccl_nonlin_matter_power(cosmo,k, a_at_z2,&status);
	p3 = ccl_nonlin_matter_power(cosmo,k, a_at_z3,&status);
	printf("%le %le %le %le %le\n", k, p,p1,p2,p3);
      }
    }
    else {
      printf("ccl_sample_power.c: Unknown power spectrum method.\n");
      return NAN;
    }
  }
  printf("sigma_8 = %.6lE\n", ccl_sigmaR(cosmo,8./h,&status));
  printf("Consistency check: sigma_8 = %.6lE\n", ccl_sigma8(cosmo,&status));
  printf("Completed. Status = %d\n",status);
  
  ccl_cosmology_free(cosmo);

  return 0;
}
