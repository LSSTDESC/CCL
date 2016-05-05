#include "ccl.h"
#include <stdio.h>
#include <math.h>

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

	printf("[0]z, [1]chi(z), [2]dL(z), [3]mu(z), [4]D(z), [5]f(z)\n");
	for (double z=0.0; z<=1.0; z+=0.01){
	  int st;
		double a = 1/(1.+z);
		double DL = 1/h * ccl_luminosity_distance(cosmo, a);
		double DL_pc = DL*1e6;
		double mu = 5*log10(DL_pc)-5;
		double chi=ccl_comoving_radial_distance(cosmo,a);
		double gf=ccl_growth_factor(cosmo,a,&st);
		double fg=ccl_growth_rate(cosmo,a,&st);
		printf("%le  %le %le %le %le %le\n", z, chi,h*DL,mu,gf,fg);
	}


}
