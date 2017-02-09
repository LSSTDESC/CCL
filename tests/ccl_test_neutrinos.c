#include "ccl.h"
#include "ccl_neutrinos.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char * argv[]){

	ccl_parameters params = ccl_parameters_create_flat_lcdm(0.3, 0.05, 0.7, 2.215e-9, 0.9619);
	ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);
	//int status; 
	int ai;
	double a, omnuh2_3massless, omnuh2_3massive;
	FILE * output;

	ccl_cosmology_compute_distances(cosmo);
	//printf ("mnu/T integral; zero limit: %g vs  %g\n",
	//	ccl_nu_phasespace_intg(cosmo->data.nu_pspace_int, 0),
	//	ccl_nu_phasespace_intg(cosmo->data.nu_pspace_int, CCL_NU_MNUT_MIN*1.001));
	//printf ("mnu/T integral; infty limit: %g vs  %g\n",
	//	ccl_nu_phasespace_intg(cosmo->data.nu_pspace_int, CCL_NU_MNUT_MAX*1.01),
	//	ccl_nu_phasespace_intg(cosmo->data.nu_pspace_int, CCL_NU_MNUT_MAX*0.99));
 

	// Get Omeganuh^2 at several values of a

	output = fopen("./neutrinos_example.out", "w"); 
	for(ai=1; ai<=50; ai++){
		a= ai*0.02;
   		// Examples of calling Omeganuh2 for different neutrino configurations:

		// All neutrinos massless:	
                omnuh2_3massless = Omeganuh2(a, 3.046, 0., 2.7255, cosmo->data.nu_pspace_int);	
		// Three massive neutrinos of 0.04 eV each. Adding a small contribution from massless neutrinos as described in CLASS explanatory.ini to ensure N = 3.046 at early times.
  		omnuh2_3massive = Omeganuh2(a, 3., 0.12, 2.7255, cosmo->data.nu_pspace_int)+ Omeganuh2(a, 0.00641, 0., 2.7255, cosmo->data.nu_pspace_int);

		fprintf(output, "%.16f %.16f %.16f \n",a, omnuh2_3massless, omnuh2_3massive); 
		
	}

	fclose(output);

	return 0;
}
