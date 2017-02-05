#include "ccl.h"
#include <stdio.h>


int main(int argc, char * argv[]){

	double Omega_c = 0.25;
	double Omega_b = 0.05;
	double h = 0.7;
	double A_s = 2.1e-9;
	double n_s = 0.96;
	ccl_configuration config = default_config;
//	config.transfer_function_method = ccl_bbks;
	config.transfer_function_method = ccl_boltzmann;

	ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, A_s, n_s);
	ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

	ccl_cosmology_compute_power(cosmo);
	
	printf("# k [1/Mpc] P_lin(k,z=0) P_nl(k,z=0)\n");

	for (double k = 1e-3; k<1e3; k*=1.05){
	  double p = ccl_linear_matter_power(cosmo, 1.0, k);
	  double pln = ccl_nonlin_matter_power(cosmo, 1.0, k);
	  printf("%le    %le %le\n", k, p,pln);
	}

	printf("sigma_8 = %.6lE\n", ccl_sigmaR(cosmo,8./h));

	return 0;

}
