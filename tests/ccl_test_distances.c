#include "ccl.h"
#include "ctest.h"

CTEST_DATA(lcdm) {
    double Omega_c;
    double Omega_b;
    double h;
    double A_s;
    double n_s;
};

CTEST_SETUP(lcdm){
    data->Omega_c = 0.25;
    data->Omega_b = 0.05;
    data->h = 0.7;
    data->A_s = 2.1e-9;
    data->n_s = 0.96;
}

CTEST2(lcdm, distance){
	ccl_parameters params = ccl_parameters_create_flat_lcdm(data->Omega_c, data->Omega_b, data->h, data->A_s, data->n_s);
	ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);
	ASSERT_NOT_NULL(cosmo);
	
}


// int main(int argc, char * argv[]){

// 	double Omega_c = 0.25;
// 	double Omega_b = 0.05;
// 	double h = 0.7;
// 	double A_s = 2.1e-9;
// 	double n_s = 0.96;
// 	double wa = 0.01;
// 	double w0 = -1.0;

// //	ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, A_s, n_s);
// 	ccl_parameters params = ccl_parameters_create_flat_wacdm(Omega_c, Omega_b, w0,wa, h, A_s, n_s);

// 	ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);

// 	int status; 
// 	ccl_cosmology_compute_distances(cosmo, &status);

// 	printf("[0]z, [1]chi(z), [2]dL(z), [3]mu(z), [4]D(z), [5]f(z)\n");
// 	for (double z=0.0; z<=1.0; z+=0.01){
// 	  int st;
// 		double a = 1/(1.+z);
// 		double DL = 1/h * ccl_luminosity_distance(cosmo, a);
// 		double DL_pc = DL*1e6;
// 		double mu = 5*log10(DL_pc)-5;
// 		double chi=ccl_comoving_radial_distance(cosmo,a);
// 		double gf=ccl_growth_factor(cosmo,a,&st);
// 		double fg=ccl_growth_rate(cosmo,a,&st);
// 		printf("%le  %le %le %le %le %le\n", z, chi,h*DL,mu,gf,fg);
// 	}


// }
