#include "ccl.h"
#include "ccl_neutrinos.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char * argv[]){

	ccl_parameters params = ccl_parameters_create_flat_lcdm(0.3, 0.05, 0.7, 1.0, 1.0);
	ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);
	int status; 
	ccl_cosmology_compute_distances(cosmo, &status);
	printf ("mnu/T integral; zero limit: %g vs  %g\n",
		ccl_nu_phasespace_intg(cosmo->data.nu_pspace_int, 0),
		ccl_nu_phasespace_intg(cosmo->data.nu_pspace_int, CCL_NU_MNUT_MIN*1.001));
	printf ("mnu/T integral; infty limit: %g vs  %g\n",
		ccl_nu_phasespace_intg(cosmo->data.nu_pspace_int, CCL_NU_MNUT_MAX*1.01),
		ccl_nu_phasespace_intg(cosmo->data.nu_pspace_int, CCL_NU_MNUT_MAX*0.99));


	return 0;
}
