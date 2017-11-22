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
  double other_bcm_log10Mc = log10(1.7e14);
  double other_bcm_etab = 0.3;
  double other_bcm_ks = 75.;

  ccl_configuration config = default_config;
  config.baryons_power_spectrum_method=ccl_bcm;
  //The following two options are identical if the parameters being passed are the fiducial ones
  //In this case, we are passing user-specified parameters
  //ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, normp, n_s, &status);
  ccl_parameters params = ccl_parameters_create_flat_lcdm_bar(Omega_c, Omega_b, h, normp, n_s,
							      other_bcm_log10Mc, other_bcm_etab,
							      other_bcm_ks, &status);
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  
  double k,p,p1,p2,p3=0;
  double a_at_z1=0.5;
  double a_at_z2=1./3.;
  double a_at_z3=0.25;
  //The linear power spectrum is not changed when baryons are passed
  /*printf("Linear matter PS\n");
  printf("# k [1/Mpc],P(k,z=0),P(k,z=1),P(k,z=2),P(k,z=3)\n");
  for (k = ccl_splines->K_MIN_DEFAULT; k<ccl_splines->K_MAX; k*=1.05) {
      p = ccl_linear_matter_power(cosmo, k,1.0, &status);
      p1 = ccl_linear_matter_power(cosmo,k, a_at_z1,&status);
      p2 = ccl_linear_matter_power(cosmo,k, a_at_z2,&status);
      p3 = ccl_linear_matter_power(cosmo,k, a_at_z3,&status);
      printf("%le %le %le %le %le\n", k, p,p1,p2,p3);
      }*/
  printf("# Total matter power spectrum\n");
  printf("# k [1/Mpc],P(k,z=0),P(k,z=1),P(k,z=2),P(k,z=3)\n");
  for (k = ccl_splines->K_MIN_DEFAULT; k<ccl_splines->K_MAX; k*=1.05) {
    p = ccl_nonlin_matter_power(cosmo, k,1.0,&status);
    p1 = ccl_nonlin_matter_power(cosmo,k, a_at_z1,&status);
    p2 = ccl_nonlin_matter_power(cosmo,k, a_at_z2,&status);
    p3 = ccl_nonlin_matter_power(cosmo,k, a_at_z3,&status);
    printf("%le %le %le %le %le\n", k, p,p1,p2,p3);
  }

  printf("Completed. Status = %d\n",status);
  
  ccl_cosmology_free(cosmo);

  return 0;
}
