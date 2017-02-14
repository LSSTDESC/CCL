#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"
#include "ccl_lsst_specs.h"

#define OC 0.25
#define OB 0.05
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define S8 0.80
#define ZD 0.5
#define NZ 128
#define Z0_GC 0.50
#define SZ_GC 0.05
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 512
#define PS 0.1 

double pz_func_example (double photo_z, double spec_z, void *param){
	double delta_z = photo_z - spec_z;
	return 1.0 / sqrt(PS*2*M_PI) * exp(-delta_z*delta_z / (2.0 * PS));
}

int main(int argc,char **argv){

	// Initialize cosmological parameters
	ccl_configuration config=default_config;
	config.transfer_function_method=ccl_bbks;
	ccl_parameters params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,NAN,NS,-1,NULL,NULL);
	params.sigma_8=S8;

	// Initialize cosmology object given cosmo params
	ccl_cosmology *cosmo=ccl_cosmology_create(params,config);

	// Compute radial distances (see include/ccl_background.h for more routines)
	printf("Comoving distance to z = %.3lf is chi = %.3lf Mpc\n",
		ZD,ccl_comoving_radial_distance(cosmo,1./(1+ZD)));
	printf("Luminosity distance to z = %.3lf is chi = %.3lf Mpc\n",
		ZD,ccl_luminosity_distance(cosmo,1./(1+ZD)));
	
	//Consistency check
	printf("Scale factor at chi=%.3lf Mpc is a=%.3lf Mpc\n",
	ccl_comoving_radial_distance(cosmo,1./(1+ZD)),
	ccl_scale_factor_of_chi(cosmo,ccl_comoving_radial_distance(cosmo,1./(1+ZD))));
	 
	// Compute growth factor and growth rate (see include/ccl_background.h for more routines)
	printf("Growth factor and growth rate at z = %.3lf are D = %.3lf and f = %.3lf\n",
		ZD, ccl_growth_factor(cosmo,1./(1+ZD)),ccl_growth_rate(cosmo,1./(1+ZD)));

	// Compute sigma_8
	printf("Initializing power spectrum...\n");
	printf("sigma_8 = %.3lf\n\n", ccl_sigma8(cosmo));

	//Create tracers for angular power spectra
	double z_arr_gc[NZ],z_arr_sh[NZ],nz_arr_gc[NZ],nz_arr_sh[NZ],bz_arr[NZ];
	for(int i=0;i<NZ;i++)
	{
		z_arr_gc[i]=Z0_GC-5*SZ_GC+10*SZ_GC*(i+0.5)/NZ;
		nz_arr_gc[i]=exp(-0.5*pow((z_arr_gc[i]-Z0_GC)/SZ_GC,2));
		bz_arr[i]=1+z_arr_gc[i];
		z_arr_sh[i]=Z0_SH-5*SZ_SH+10*SZ_SH*(i+0.5)/NZ;
		nz_arr_sh[i]=exp(-0.5*pow((z_arr_sh[i]-Z0_SH)/SZ_SH,2));
	}

	//Galaxy clustering tracer
	CCL_ClTracer *ct_gc=ccl_cl_tracer_number_counts_simple_new(cosmo,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr);

	//Cosmic shear tracer
	CCL_ClTracer *ct_wl=ccl_cl_tracer_lensing_simple_new(cosmo,NZ,z_arr_sh,nz_arr_sh);
	printf("ell C_ell(g,g) C_ell(g,s) C_ell(s,s) | r(g,s)\n");
	for(int l=2;l<=NL;l*=2)
	{
		double cl_gg=ccl_angular_cl(cosmo,l,ct_gc,ct_gc); //Galaxy-galaxy
		double cl_gs=ccl_angular_cl(cosmo,l,ct_gc,ct_wl); //Galaxy-lensing
		double cl_ss=ccl_angular_cl(cosmo,l,ct_wl,ct_wl); //Lensing-lensing
		printf("%d %.3lE %.3lE %.3lE | %.3lE\n",l,cl_gg,cl_gs,cl_ss,cl_gs/sqrt(cl_gg*cl_ss));
	}
	printf("\n");

	//Free up tracers
	ccl_cl_tracer_free(ct_gc);
	ccl_cl_tracer_free(ct_wl);
	
	//Halo mass function
	printf("M\tdN/dM(z = 0, 0.5, 1))\n");
	for(int logM=9;logM<=15;logM+=1)
	{
		printf("%.1e\t",pow(10,logM));
		for(double z=0; z<=1; z+=0.5)
		{
			printf("%e\t", ccl_massfunc(cosmo, pow(10,logM),z));
		}
		printf("\n");
	}
	printf("\n");

	// LSST Specification
	user_pz_info* pz_info_example = ccl_specs_create_photoz_info(NULL, pz_func_example);
	
	double z_test;
	double dNdz_tomo;
	int z,status;
	FILE * output;
	
	//Try splitting dNdz (lensing) into 5 redshift bins
	double tmp1,tmp2,tmp3,tmp4,tmp5;
	printf("Trying splitting dNdz (lensing) into 5 redshift bins. Output written into file tests/specs_example_tomo_lens.out\n");
	output = fopen("./tests/specs_example_tomo_lens.out", "w");     
	for (z=0; z<100; z=z+1){
		z_test = 0.035*z;
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,6., pz_info_example,&dNdz_tomo); 
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,0.6, pz_info_example,&tmp1); 
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.6,1.2, pz_info_example,&tmp2);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.2,1.8, pz_info_example,&tmp3);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.8,2.4, pz_info_example,&tmp4); 
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 2.4,3.0, pz_info_example,&tmp5);
		fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
	}

	fclose(output);

	//Try splitting dNdz (clustering) into 5 redshift bins
	printf("Trying splitting dNdz (clustering) into 5 redshift bins. Output written into file tests/specs_example_tomo_lens.out\n");
	output = fopen("./tests/specs_example_tomo_clu.out", "w");     
	for (z=0; z<100; z=z+1){
		z_test = 0.035*z;
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.,6., pz_info_example,&dNdz_tomo); 
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.,0.6, pz_info_example,&tmp1);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.6,1.2, pz_info_example,&tmp2);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,1.2,1.8, pz_info_example,&tmp3);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,1.8,2.4, pz_info_example,&tmp4);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,2.4,3.0, pz_info_example,&tmp5);
		fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
	}

	fclose(output);
	
	//Free up photo-z info
	ccl_specs_free_photoz_info(pz_info_example);

	//Always clean up!!
	ccl_cosmology_free(cosmo);

	return 0;
}

