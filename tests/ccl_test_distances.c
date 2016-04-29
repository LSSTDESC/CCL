#include "ccl.h"
#include <stdio.h>


int main(int argc, char * argv[]){

	double Omega_c = 0.25;
	double Omega_b = 0.05;
	double h = 0.7;
	double A_s = 2.1e-9;
	double n_s = 0.96;

	ccl_parameters params = ccl_parameters_flat_lcdm(Omega_c, Omega_b, h, A_s, n_s);
	ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);

	int status; 
	ccl_cosmology_compute_distances(cosmo, &status);

	for (double z=0.0; z<=1.0; z+=0.01){
		double a = 1/(1.+z);
		double DL = 2997.0/h * ccl_luminosity_distance(cosmo, a);
		double DL_pc = DL*1e6;
		double mu = 5*log10(DL_pc)-5;
		printf("%le  %le\n", z, mu);
	}


}