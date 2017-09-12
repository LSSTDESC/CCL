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
  double other_bcm_log10Mc = log10(1.2e14);//13.;
  double other_bcm_etab = 0.5;//0.7;
  double other_bcm_ks = 55.;//80.;

  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_baryons_bcm;
  //These two options are identical if the parameters being passed are the fiducial ones
  ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, normp, n_s, &status);
  /*ccl_parameters params = ccl_parameters_create_flat_lcdm_bar(Omega_c, Omega_b, h, normp, n_s,
							      other_bcm_log10Mc, other_bcm_etab,
							      other_bcm_ks, &status);*/
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

  printf("# k [1/Mpc],P(k,z=0),P(k,z=1),P(k,z=2),P(k,z=3)\n");
  
  double k,p,p1,p2,p3=0;
  double a_at_z1=0.5;
  double a_at_z2=1./3.;
  double a_at_z3=0.25;
  //The linear power spectrum is not changed when baryons are passed
  printf("Linear matter PS:\n");
  for (k = ccl_splines->K_MIN_DEFAULT; k<ccl_splines->K_MAX; k*=1.05) {
      p = ccl_linear_matter_power(cosmo, k,1.0, &status);
      p1 = ccl_linear_matter_power(cosmo,k, a_at_z1,&status);
      p2 = ccl_linear_matter_power(cosmo,k, a_at_z2,&status);
      p3 = ccl_linear_matter_power(cosmo,k, a_at_z3,&status);
      printf("%le %le %le %le %le\n", k, p,p1,p2,p3);
      }
  printf("NL matter PS:");
  for (k = ccl_splines->K_MIN_DEFAULT; k<ccl_splines->K_MAX; k*=1.05) {
    p = ccl_nonlin_matter_power(cosmo, k,1.0,&status);
    p1 = ccl_nonlin_matter_power(cosmo,k, a_at_z1,&status);
    p2 = ccl_nonlin_matter_power(cosmo,k, a_at_z2,&status);
    p3 = ccl_nonlin_matter_power(cosmo,k, a_at_z3,&status);
    printf("%le %le %le %le %le\n", k, p,p1,p2,p3);
    }
  printf("Baryonic FKZ:\n");
  for (k = ccl_splines->K_MIN_DEFAULT; k<ccl_splines->K_MAX; k*=1.05) {
    p = bcm_model_fkz(cosmo, k,1.0,&status);
    printf("%le %le\n", k, p);
    }
  //Notice that the ratio of p_nl with and without baryons can differ from fkz at
  //small scales most likely due to interpolation effects. This should be studied.
  //Checked that sigma8 is the same with or without baryons correction
  printf("sigma_8 = %.6lE\n", ccl_sigmaR(cosmo,8./h,&status));
  printf("Consistency check: sigma_8 = %.6lE\n", ccl_sigma8(cosmo,&status));
  printf("Completed. Status = %d\n",status);
  
  ccl_cosmology_free(cosmo);

  return 0;
}
