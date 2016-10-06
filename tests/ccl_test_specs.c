#include "ccl_placeholder.h"
#include <stdio.h>
#include <math.h>

int main(){

	double z_test, a_test;
	double dNdzk2, dNdzk1, dNdzk0pt5;
	double dNdz_clust;
	double sigz_src;
	double sigz_clust;
	double clust_bias;	
	int k;
	//int *status;
	double Omega_c = 0.25;
        double Omega_b = 0.05;
        double h = 0.7;
        double A_s = 2.1e-9;
        double n_s = 0.96;
	FILE * output;

	/* Call the different options of WL source dNdzs, the clustering dNdz, and the photometric redshift errors for src and clustering galaxies at each z and save the output */
	// This is just to test that the functions can be called and give something reasonable.

	// Open file for writing:
	output = fopen("./tests/specs_output_test.dat", "w");	

	// Test also the function for the bias in the clustering sample (requires setting up a cosmology to get the growth rate)
	//ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, A_s, n_s);
	//ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);

	//status = 0;
	for (k=0; k<100; k=k+1){

		//int status;	

		z_test = 0.03*k;
		a_test = 1./ (1 + z_test);
		dNdzk2 = dNdz_sources_k2(z_test);
		dNdzk1 = dNdz_sources_k1(z_test);
		dNdzk0pt5 = dNdz_sources_k0pt5(z_test);
		dNdz_clust = dNdz_clustering(z_test);
		sigz_src = sigmaz_sources(z_test);
		sigz_clust = sigmaz_clustering(z_test);
		//clust_bias = bias_clustering(cosmo, a_test, &status);
		
		fprintf(output, "%f %f %f %f %f %f %f \n", z_test,dNdzk2, dNdzk1, dNdzk0pt5, dNdz_clust, sigz_src, sigz_clust);
	}

	fclose(output);
}


