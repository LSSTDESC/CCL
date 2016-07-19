#include "ccl.h"
#include <stdio.h>


int main(int argc, char * argv[]){

	double Omega_c = 0.25;
	double Omega_b = 0.05;
	double h = 0.7;
	double A_s = 2.1e-9;
	double n_s = 0.96;
	ccl_configuration config = default_config;
	config.transfer_function_method = ccl_boltzmann;
	double sigma_8 = 0.8;

	ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, A_s, n_s);
	params.sigma_8 = sigma_8;
	ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

	int status; 
	ccl_cosmology_compute_power(cosmo, &status);

	for (double k = 1e-3; k<10; k*=1.1){
		double p = ccl_linear_matter_power(cosmo, 1.0, k, &status);
		printf("%le    %le\n", k, p);
	}

	fprintf(stderr, "status = %d\n", status);
	return status;

}
