#include "ccl_placeholder.h"
#include <stdio.h>
#include <math.h>

int main(int argc,char **argv){

	double z_test, a_test;
	double dNdzk2, dNdzk1, dNdzk0pt5, dNdz_tomo;
	double dNdz_clust;
	double sigz_src;
	double sigz_clust;
	double clust_bias;	
	int k;
	double Omega_c = 0.25;
        double Omega_b = 0.05;
        double h = 0.7;
        double A_s = 2.1e-9;
        double n_s = 0.96;
	FILE * output;
	ccl_cosmology* cosmo_1;
	void * nothing;
	double Norm2, Norm1, Norm0pt5;

	/* Call the different options of WL source dNdzs, the clustering dNdz, and the photometric redshift errors for src and clustering galaxies at each z and save the output */
	// This is just to test that the functions can be called and give something reasonable.

	// Open file for writing:
	output = fopen("./tests/specs_output_test.dat", "w");	

	// Test also the function for the bias in the clustering sample (requires setting up a cosmology to get the growth rate)
	ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, A_s, n_s);
	
	// I think that ccl_cosmology_create returns a pointed to a ccl_cosmology struct (struct defined in ccl_core.h)
	cosmo_1 = ccl_cosmology_create(params, default_config);

	//printf("cosmo_1=> params.Omega_c=%f\n", cosmo_1->params.Omega_c);
	// The output of the above print statement suggests the call to ccl_cosmology_create is going fine.

	// Get the normalising factors for the various dNdz's:
	// We should put zmin / zmax in some kind of central place so we don't need to put them in by hand.
	Norm2 = norm_dNdz(0.0, 5.0, NULL, dNdz_sources_k2);
	Norm1 = norm_dNdz(0.0, 5.0, NULL,dNdz_sources_k1);
        Norm0pt5 = norm_dNdz(0.0, 5.0, NULL,dNdz_sources_k0pt5);

	printf("Norm2=%f, Norm1=%f, Norm0pt5= %f\n", Norm2, Norm1, Norm0pt5);

	for (k=0; k<100; k=k+1){

		z_test = 0.03*k;
		a_test = 1./ (1 + z_test);
		dNdzk2 = dNdz_sources_k2(z_test, nothing);
		dNdzk1 = dNdz_sources_k1(z_test, nothing);
		dNdzk0pt5 = dNdz_sources_k0pt5(z_test, nothing);
		dNdz_clust = dNdz_clustering(z_test);
		sigz_src = sigmaz_sources(z_test);
		sigz_clust = sigmaz_clustering(z_test);	
		clust_bias = bias_clustering(cosmo_1, a_test);
		// dNdz_tomo is not normalized because I haven't put it in the right form to be a gsl_function yet.
		dNdz_tomo = dNdz_sources_tomog(z_test, 0.7, 2.5, NULL,dNdz_sources_k2, photoz_dNdz);
		//Norm = norm_dNdz(0.11, 2.99, dNdz_sources_k1);		
		// cosmo_1 should be a pointer to a ccl_cosmology struct, a_test should be a double, and &status should be the address of an integer.
		fprintf(output, "%f %f %f %f %f %f %f %f %f \n", z_test,dNdzk2/ Norm2, dNdzk1 / Norm1, dNdzk0pt5 / Norm0pt5, dNdz_clust, sigz_src, sigz_clust, clust_bias, dNdz_tomo );
	}

	fclose(output);
}


