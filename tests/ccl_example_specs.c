#include "ccl_lsst_specs.h"
#include <stdio.h>
#include <math.h>

int main(int argc,char **argv){

        double z_test;
	double dNdz_tomo;
	int z,status;
	FILE * output;

	/* Obtain dNdz for weak lensing sources and clustering 
	   in a given number of tomographic bins 
	   NB: The tracer is specified by a macro in 
	   ccl_lsst_specs.h. 
	   DNDZ_NC: clustering
	   DNDZ_WL_CONS: WL conservative 
	   DNDZ_WL_FID: WL fiducial
	   DNDZ_WL_OPT: WL optimistic
	*/       

	//Try splitting dNdz (lensing) into 5 redshift bins
	double tmp1,tmp2,tmp3,tmp4,tmp5;
	output = fopen("./tests/specs_example_tomo_lens.out", "w");     
	for (z=0; z<100; z=z+1){
		z_test = 0.035*z;
	        status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,6.,&dNdz_tomo); 
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,0.6,&tmp1); 
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.6,1.2,&tmp2);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.2,1.8,&tmp3);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.8,2.4,&tmp4); 
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 2.4,3.0,&tmp5);
		fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
	}

	fclose(output);

	//Try splitting dNdz (clustering) into 5 redshift bins
	output = fopen("./tests/specs_example_tomo_clu.out", "w");     
	for (z=0; z<100; z=z+1){
		z_test = 0.035*z;
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.,6.,&dNdz_tomo); 
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.,0.6,&tmp1);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.6,1.2,&tmp2);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,1.2,1.8,&tmp3);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,1.8,2.4,&tmp4);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,2.4,3.0,&tmp5);
		fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
	}

	fclose(output);


}


