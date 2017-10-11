#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"
#include "ccl_halofit.h"

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

	printf("# Initializing linear power spectra...\n");
	ccl_cosmology_compute_power(cosmo);
/*
	printf("# R\tgauss\ttop-head\n");
	for (double R = 0.1; R < 10; R+=0.05)
	{
		printf("%.1f\t%.4f\t%.4f\n", R, ccl_sigmaR_gauss(cosmo, R), ccl_sigmaR(cosmo, R));
	}
*/
	printf("# Linear power spectra initialized...\n");
	
	printf("#k\t Pk (CLASS)\tPk(halofit)\tRel. diff\n");
	
	double pk_class, pk_halofit, pk_diff, k;
	ccl_cosmology_halofit params_halofit = ccl_new_ccl_cosmology_halofit();
	
	for(k = 1e-4; k < 10; k*=1.1)
	{
		pk_class = ccl_nonlin_matter_power(cosmo, 1., k);
		pk_halofit = ccl_nonlin_matter_power_halofit(cosmo, 1., k, &params_halofit);
		pk_diff = 1 - pk_halofit / pk_class;
		
		printf("%f\t%f\t%f\t%f\n", k, pk_class, pk_halofit, pk_diff);
	}

	//Always clean up!!
	ccl_cosmology_free(cosmo);

	return 0;
}
