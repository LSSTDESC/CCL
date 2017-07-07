#include "ccl.h"
#include "ccl_neutrinos.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char * argv[])
{
  double OmC = 0.25;
  double OmB = 0.05;
  double OmK = 0.;
  double h = 0.7;
  double As = 2.1e-9;
  double ns = 0.96;
  double N_ur = 2.0328;
  double N_ncdm = 1.;
  double mnu = 0.04;
  int status=0;
  double a, omnuh2, h_of_a;
  FILE * output;
  FILE * input;
  double z, no_a, no_b, no_c, no_d, no_e, no_f, no_g, no_h, no_i, no_j, no_k, no_l, no_m, no_n, no_o, no_p, no_q; //Dummy parameters to read CLASS file.
  char line[2000];
  // Set up parameters and cosmologies.
  ccl_parameters params = ccl_parameters_create_lcdm_nu(OmC, OmB, OmK, h, As, ns, N_ur, N_ncdm, mnu, &status);
  ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);	
  
  // Get the spline of the phase-space integral
  //temp_spline = ccl_calculate_nu_phasespace_spline();
  
  // Print derived parameters (computed when setting up cosmology)
  printf("OmegaL=%1.12f\n",cosmo->params.Omega_l);
  printf("OmegaNuRel=%1.12f\n", cosmo->params.Omega_n_rel);
  printf("OmegaNuMass=%1.12f\n", cosmo->params.Omega_n_mass);
  printf("OmegaM=%1.12f\n", cosmo->params.Omega_m);
  printf("Omegag=%1.12f\n", cosmo->params.Omega_g);
  
  // Open file to read z values from CLASS at which to compute things
  input = fopen("./compare_ccl_1massive2massless_background.dat", "r");
  
  // Read the header 
  fgets(line,2000, input);
  fgets(line,2000, input);
  fgets(line,2000, input);
  fgets(line,2000, input);
  
  // Open file to output results
  output = fopen("./neutrinos_example_1massivem2massless_h.out", "w");
  
  while((fscanf(input, "%le %le %le %le %le %le %le %le %le %le %le %le %le %le %le %le %le %le\n", &z, &no_a, &no_b, &no_c, &no_d, &no_e, &no_f, &no_g, &no_h, &no_i, &no_j, &no_k, &no_l, &no_m, &no_n, &no_o, &no_p, &no_q)!=EOF)) {	
    a = 1. / (1. + z);		
    if ((z>(1./A_SPLINE_MIN -1.))) continue;  // Only continue if the z is within what can be handled by the splines in CCL.
    h_of_a  = ccl_h_over_h0(cosmo, a, &status);
    omnuh2 = Omeganuh2(a, params.N_nu_mass, params.mnu, params.T_CMB, cosmo->data.nu_pspace_int)+ Omeganuh2(a, params.N_nu_rel, 0., params.T_CMB, cosmo->data.nu_pspace_int);
    fprintf(output, "%.16le %.16le %.16le \n",a, omnuh2, h_of_a); 
    
  }
  
  fclose(output);
  
  return 0;
}
