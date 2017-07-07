#include "ccl_lsst_specs.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

// This is a test file to include a user-defined photo-z function

// The user defines a structure of parameters to the user-defined function for the photo-z probability 
struct user_func_params {
  double (* sigma_z) (double);
};

// The user defines a function of the form double function ( z_ph, z_spec, void * user_pz_params, int *status) where user_pz_params is a pointer to the parameters of the user-defined function. This returns the probabilty of obtaining a given photo-z given a particular spec_z.

double user_pz_probability(double z_ph, double z_s, void * user_par, int *status)
{
  struct user_func_params * p = (struct user_func_params *) user_par;
  
  return exp(- (z_ph-z_s)*(z_ph-z_s) / (2.*(p->sigma_z(z_s))*(p->sigma_z(z_s)))) / (pow(2.*M_PI,0.5)*(p->sigma_z(z_s))*(p->sigma_z(z_s)));
}

// Beginning of test 
int main(int argc,char **argv)
{
  // The user declares and sets an instance of parameters to their photo_z function:
  struct user_func_params my_params_example;
  my_params_example.sigma_z = ccl_specs_sigmaz_sources;
  
  // Declare a variable of the type of user_pz_info to hold the struct to be created.
  user_pz_info * my_info;
  
  // Create the struct to hold the user information about photo_z's.
  my_info = ccl_specs_create_photoz_info(&my_params_example, &user_pz_probability); 
  
  int status = 0;
  double z_test;
  int z;
  double tmp1,tmp2,tmp3,tmp4,tmp5;
  double dNdz_tomo;
  FILE * output;
  output = fopen("./tests/specs_example_tomo_lens_user_pz.out", "w");
  for (z=0; z<100; z=z+1) { 
    z_test = 0.035*z;
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,6.,my_info,&dNdz_tomo, &status);
    if (status!=0) {
      printf("You have selected an unsupported dNdz type. Exiting.\n");
      exit(1);
    }
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,0.6,my_info, &tmp1, &status);
    if (status!=0) {
      printf("You have selected an unsupported dNdz type. Exiting.\n");
      exit(1);
    }	
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.6,1.2,my_info, &tmp2, &status);
    if (status!=0) {
      printf("You have selected an unsupported dNdz type. Exiting.\n");
      exit(1);
    }
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.2,1.8,my_info, &tmp3, &status);
    if (status!=0) {
      printf("You have selected an unsupported dNdz type. Exiting.\n");
      exit(1);
    }
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.8,2.4,my_info, &tmp4, &status);
    if (status!=0) {
      printf("You have selected an unsupported dNdz type. Exiting.\n");
      exit(1);
    }
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 2.4,3.0,my_info, &tmp5, &status);
    if (status!=0) {
      printf("You have selected an unsupported dNdz type. Exiting.\n");
      exit(1);
    }
    fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
  }
  
  
  output = fopen("./tests/specs_example_tomo_clust_user_pz.out", "w");
  for (z=0; z<100; z=z+1) {
    z_test = 0.035*z;	
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC, 0.,6.,my_info,&dNdz_tomo, &status);
    if (status!=0) {
      printf("You have selected an unsupported dNdz type. Exiting.\n");
      exit(1);
    }
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC, 0.,0.6,my_info, &tmp1, &status);
    if (status!=0) {
      printf("You have selected an unsupported dNdz type. Exiting.\n");
      exit(1);
    }
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC, 0.6,1.2,my_info, &tmp2, &status);
    if (status!=0) {
      printf("You have selected an unsupported dNdz type. Exiting.\n");
      exit(1);
    }
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC, 1.2,1.8, my_info, &tmp3, &status);
    if (status!=0) {
      printf("You have selected an unsupported dNdz type. Exiting.\n");
      exit(1);
    }
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC, 1.8,2.4, my_info, &tmp4, &status);
    if (status!=0) {
      printf("You have selected an unsupported dNdz type. Exiting.\n");
      exit(1);
    }
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC, 2.4,3.0, my_info, &tmp5, &status);
    if (status!=0) {
      printf("You have selected an unsupported dNdz type. Exiting.\n");
      exit(1);
    }
    fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
  }
  fclose(output);
  
  // Free the photo_z information
  ccl_specs_free_photoz_info(my_info);
  
  return 0;
}
