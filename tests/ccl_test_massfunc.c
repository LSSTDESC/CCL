#include "ccl.h"
#include "ccl_core.h"
#include "ccl_utils.h"
#include "ccl_massfunc.h"
#include <math.h>
#include <stdio.h>

// just a test main function until things are working. Not for final dist.
int main(){
    // set base cosmology for testing purposes
    double Omega_c = 0.25;
    double Omega_b = 0.05;
    double h = 0.7;
    double A_s = 2.1E-9;
    double n_s = 0.96;

    double logmass, mass, redshift, test;
    int i, j;
    FILE * fp;

    fp = fopen("test_massfunc.txt", "w");

    ccl_configuration config = default_config;
    config.transfer_function_method = ccl_bbks;

    ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, A_s, n_s);
    params.sigma_8 = 0.8; // default for testing purposes since NaN

    ccl_cosmology * cosmo = ccl_cosmology_create(params, config);

    logmass = 11;
    for(i=0; i<9; i++){
       mass = pow(10, logmass);
       fprintf(fp, "%le ", mass);
       redshift = 0;
       for(j=0; j<7; j++){
          test = ccl_massfunc(cosmo, mass, redshift);
          fprintf(fp, "%le ", test);
          redshift += 0.2;
       }
       fprintf(fp, "\n");
       logmass += 0.5;
    }
    fclose(fp);

    return 0;
}
