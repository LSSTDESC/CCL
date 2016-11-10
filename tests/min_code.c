#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"

#define OC 0.25
#define OB 0.05
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define AS 2.1E-9
#define ZD 0.5

int main(int argc,char **argv){
	// Initialize cosmological parameters
	ccl_parameters params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,AS,NS,-1,NULL,NULL);

	// Initialize cosmology object given cosmo params
	ccl_cosmology *cosmo=ccl_cosmology_create(params,default_config);

	// Compute radial distances
	printf("Comoving distance to z = %.3lf is chi = %.3lf Mpc\n",
		ZD,ccl_comoving_radial_distance(cosmo,1./(1+ZD)));
	printf("Luminosity distance to z = %.3lf is chi = %.3lf Mpc\n",
		ZD,ccl_luminosity_distance(cosmo,1./(1+ZD)));

	// Compute growth factor and growth rate
	printf("Growth factor and growth rate at z = %.3lf are D = %.3lf and f = %.3lf\n",
		ZD, ccl_growth_factor(cosmo,1./(1+ZD)),ccl_growth_rate(cosmo,1./(1+ZD)));

	// Compute sigma_8
	printf("sigma_8 = %.3lf\n", ccl_sigma8(cosmo));

	//Always clean up!!
	ccl_cosmology_free(cosmo);

	return 0;
}
