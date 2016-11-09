#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"

#define OC 0.25
#define OB 0.05
#define OL 0.70
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define AS 2.1E-9
#define ZD 0.5
#define NZ 128
#define Z0_GC 0.50
#define SZ_GC 0.05
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 500

int ask_to_con(){
	printf("Do you want to continue? (YES / NO)\n");
	char in[10];
	int i = 0;
	while (true){
		scanf("%10s", in);
		if ((strcmp(in, "") == 0) || (strcmp(in, "yes") == 0) || (strcmp(in, "YES") == 0)) return 0;
		else if ((strcmp(in, "no") == 0) || (strcmp(in, "NO") == 0)){printf("Exiting test...\n"); return 1;}
		else if (i > 10){printf("Too many unrecongized options. Exiting test...\n"); return 1;}
		else {printf("Unrecognized option.\n"); i++;}
  }
}

int main(int argc,char **argv)
{
	int err;
	printf("\n***********************");
	printf("\n* CCL EXAMPLE PROGRAM *");
	printf("\n***********************\n\n");
	printf("* Example program which goes through basic of CCL library. For more information see file 'readne.md' in root directory.\n*\n");
  
	//Initialize cosmological parameters
	printf("* Initialize cosmological parameters with\n*\n");
	printf("* ccl_parameters ccl_parameters_create(\n");
	printf("*\tdouble Omega_c, double Omega_b, double Omega_k, double Omega_n, double w0, double wa, double h, double A_s, double n_s,\n");
	printf("*\tint nz_mgrowth,double *zarr_mgrowth,double *dfarr_mgrowth);\n*\n\n");
	printf("\tccl_parameters params = ccl_parameters_create (0.25,0.05,0.00,0.00,-1.00,0.00,0.70,2.1E-9,0.96,-1,NULL,NULL);\n\n*\n");
	ccl_parameters params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,AS,NS,-1,NULL,NULL);
	//if (ask_to_con()) return 0;
	
	//Initialize cosmology object given cosmo params
	printf("* Initialize cosmology with call for\n*\n");
	printf("* ccl_cosmology * ccl_cosmology_create(ccl_parameters params, ccl_configuration config);\n*\n\n");
	printf("\tccl_cosmology *cosmo = ccl_cosmology_create (params,default_config);\n\n*\n");
	ccl_cosmology *cosmo=ccl_cosmology_create(params,default_config);
	//if (ask_to_con()) return 0;
	
	//Compute radial distance (see include/ccl_background.h for more routines)
	printf("* With initialized comsology we can compute distances, growth factor or sigma_8.\n*\n");
	printf("* \tdouble ccl_comoving_radial_distance(ccl_cosmology * cosmo, double a);\n");
	printf("* \tdouble ccl_luminosity_distance(ccl_cosmology * cosmo, double a)\n");
	printf("* \tdouble ccl_growth_factor_unnorm(ccl_cosmology * cosmo, double a);\n");
	printf("* \tdouble ccl_sigma8(ccl_cosmology * cosmo);\n*\n");
	
	printf("* For our cosmological parameters, comoving distance to z = %.3lf is chi = %.3lf Mpc/h\n",ZD,ccl_comoving_radial_distance(cosmo,1./(1+ZD)));
	printf("* (consistency check) Scale factor at chi = %.3lf Mpc/h is a = %.3lf.\n",
		ccl_comoving_radial_distance(cosmo,1./(1+ZD)), ccl_scale_factor_of_chi(cosmo,ccl_comoving_radial_distance(cosmo,1./(1+ZD))));
	printf("* Luminosity distance to z = %.3lf is chi = %.3lf Mpc/h\n",ZD,ccl_luminosity_distance(cosmo,1./(1+ZD)));
	printf("* Growth factor and growth rate at z = %.3lf are D = %.3lf and f = %.3lf\n",ZD, ccl_growth_factor(cosmo,1./(1+ZD)),ccl_growth_rate(cosmo,1./(1+ZD)));
	//Compute sigma_8
	printf("* Computing sigma_8 (this may take a while)...\n");
	printf("* sigma_8 = %.3lf\n", ccl_sigma8(cosmo));
	return 0;
}
