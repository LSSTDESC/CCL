#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <ccl.h>
#include <ccl_redshifts.h>

// This is an example code showing how to incldue a user-defined photo-z 
// function and user-defined true redshift distribution 

// The user defines a structure of parameters to the user-defined function for the photo-z probability 
struct user_pz_params {
  double (* sigma_z) (double);
};

// Define the function we want to use for sigma_z which is included in the above struct.
double sigmaz_sources(double z)
{
  return 0.05*(1.0+z);
}

// The user defines a function of the form double function ( z_ph, z_spec, void * user_pz_params, int *status) where user_pz_params is a pointer to the parameters of the user-defined function. This returns the probabilty of obtaining a given photo-z given a particular spec_z.

double user_pz_probability(double z_ph, double z_s, void * user_par, int *status)
{
  struct user_pz_params * p = (struct user_pz_params *) user_par;
  
  return exp(- (z_ph-z_s)*(z_ph-z_s) / (2.*(p->sigma_z(z_s))*(p->sigma_z(z_s)))) / (pow(2.*M_PI,0.5)*(p->sigma_z(z_s))*(p->sigma_z(z_s)));
}

// The user defines a structure of parameters to the user-defined function for the true dNdz
struct user_dN_params {
  double alpha;
  double beta;
  double z0;
};

// The user defines a function of the form double function ( z, void * params, int *status) where params is a pointer to the parameters of the user-defined function. This returns the true dNdz.

double user_dNdz(double z, void * user_par, int *status)
{
  struct user_dN_params * p = (struct user_dN_params *) user_par;
  
  return pow(z, p->alpha) * exp(- pow(z/(p->z0), p->beta) );

}

int main(int argc,char **argv)
{
  // The user declares and sets an instance of parameters to their photo_z function:
  struct user_pz_params my_pz_params_example;
  my_pz_params_example.sigma_z = sigmaz_sources;
  struct user_dN_params my_dN_params_example;
  my_dN_params_example.alpha = 1.24;
  my_dN_params_example.beta = 1.01;
  my_dN_params_example.z0 = 0.51;
  
  // Declare a variable of the type of user_pz_info to hold the struct to be created.
  pz_info * my_pz_info;
  
  // Create the struct to hold the user information about photo_z's.
  my_pz_info = ccl_create_photoz_info(&my_pz_params_example, &user_pz_probability);
  
  // Declare a variable of the type of user_dN_info to hold the struct to be created
  dNdz_info * my_dN_info; 
  
  // Create a simple analytic true redshift distribution:
  my_dN_info = ccl_create_dNdz_info(&my_dN_params_example, &user_dNdz);
  
  int status = 0;
  double z_test;
  int z;
  double tmp1,tmp2,tmp3,tmp4,tmp5;
  double dNdz_tomo;
  FILE * output;
  output = fopen("./tests/example_tomographic_bins.out", "w");
  for (z=0; z<100; z=z+1) { 
    z_test = 0.035*z;

    ccl_dNdz_tomog(z_test,  0.,6.,my_pz_info, my_dN_info, &dNdz_tomo, &status);
    if (status!=0) {
      printf("Error in initiating the tomographic bins. Exiting.\n");
      exit(1);
    }
    
    ccl_dNdz_tomog(z_test, 0.,0.6,my_pz_info, my_dN_info,  &tmp1, &status);
    if (status!=0) {
      printf("Error in initiating the tomographic bins. Exiting.\n");
      exit(1);
    }	
    
    ccl_dNdz_tomog(z_test, 0.6,1.2,my_pz_info, my_dN_info,  &tmp2, &status);
    if (status!=0) {
      printf("Error in initiating the tomographic bins. Exiting.\n");
      exit(1);
    }
    ccl_dNdz_tomog(z_test, 1.2,1.8,my_pz_info, my_dN_info,  &tmp3, &status);
    if (status!=0) {
      printf("Error in initiating the tomographic bins. Exiting.\n");
      exit(1);
    }
    ccl_dNdz_tomog(z_test, 1.8,2.4,my_pz_info, my_dN_info,  &tmp4, &status);
    if (status!=0) {
      printf("Error in initiating the tomographic bins. Exiting.\n");
      exit(1);
    }
    ccl_dNdz_tomog(z_test, 2.4,3.0,my_pz_info, my_dN_info,  &tmp5, &status);
    if (status!=0) {
      printf("Error in initiating the tomographic bins. Exiting.\n");
      exit(1);
    }
    fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
  }
  
  fclose(output);
  
  // Free the photo_z information
  ccl_free_photoz_info(my_pz_info);
  
  // Free the dNdz information
  ccl_free_dNdz_info(my_dN_info);
  
  return 0;
}
