#include "ccl_lsst_specs.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

// This is a test file to include a user-defined photo-z function

// Here is a function of z for sigma_z that is required as a parameter of the user-defined photo_z function (here a Gaussian).
double sigmaz_sources(double z)
{
  return 0.05*(1.0+z);
}


// The user defines a structure of parameters to the user-defined function
// This is the parameters that go into the pdf-producing function
struct user_func_params{
	double (* sigma_z) (double);
};

// The user defines a function of the form double function ( z_ph, z_spec, void * user_pz_params) where user_pz_params is a pointer to the parameters of the user-defined function. This returns the probabilty of obtaining a given photo-z given a particular spec_z.

double user_pz_probability(double z_ph, double z_s, void * user_par){

        struct user_func_params * p = (struct user_func_params *) user_par;

        return exp(- (z_ph-z_s)*(z_ph-z_s) / (2.*(p->sigma_z(z_s))*(p->sigma_z(z_s)))) / (pow(2.*M_PI,0.5)*(p->sigma_z(z_s))*(p->sigma_z(z_s)));

        }

// Beginning of test 
int main(int argc,char **argv){

	// The user declares and sets an instance of parameters to photo_z function:
	struct user_func_params my_params_example;
	my_params_example.sigma_z = sigmaz_sources;


	// The user declares and initializes an instance of struct type user_pz_info 
	user_pz_info my_info_val;

	my_info_val.your_pz_params = &my_params_example;
	my_info_val.your_pz_func = user_pz_probability;

	int status = 1;
	double z_test;
	int z;
	double tmp1,tmp2,tmp3,tmp4,tmp5;
	double dNdz_tomo;
	FILE * output;
        output = fopen("./tests/specs_example_tomo_lens_user_pz.out", "w");
        for (z=0; z<100; z=z+1){ 
		z_test = 0.035*z;
                status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,6.,&my_info_val,&dNdz_tomo);
                status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,0.6,&my_info_val, &tmp1);
                status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.6,1.2,&my_info_val, &tmp2);
                status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.2,1.8,&my_info_val, &tmp3);
                status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.8,2.4,&my_info_val, &tmp4);
                status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 2.4,3.0,&my_info_val, &tmp5);
                fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
        }


	output = fopen("./tests/specs_example_tomo_clust_user_pz.out", "w");
        for (z=0; z<100; z=z+1){
		z_test = 0.035*z;	
                status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC, 0.,6.,&my_info_val,&dNdz_tomo);
                status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC, 0.,0.6,&my_info_val, &tmp1);
                status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC, 0.6,1.2,&my_info_val, &tmp2);
                status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC, 1.2,1.8,&my_info_val, &tmp3);
                status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC, 1.8,2.4,&my_info_val, &tmp4);
                status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC, 2.4,3.0,&my_info_val, &tmp5);
                fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
        }


        fclose(output);


	

}
