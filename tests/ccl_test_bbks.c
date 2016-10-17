#include "ccl.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char * argv[]){
  
        if(argv[1]==NULL){
          printf("No model argument given - program exiting!\n Integer arguments between 1 and 3 are allowed.\n");
          return 0;
        }
        int i_model=atoi(argv[1]);
        if( (i_model>3) || (i_model<1)){
          printf("Model argument not between 1 and 3 - program exiting!\n Integer arguments between 1 and 3 are allowed.\n");
          return 0;
        }
	
        int i, j, nk, testflag=0;
	double Omega_c = 0.25;
	double Omega_b = 0.05;
	double h = 0.7;
	double A_s = 2.1e-9;
	double sigma_8=0.8;
  	double n_s = 0.96;
        double Omega_v[5] = {0.7, 0.7, 0.7, 0.65, 0.75};
        double w_0[5] = {-1.0, -0.9, -0.9, -0.9, -0.9};
        double w_a[5] = {0.0, 0.0, 0.1, 0.1, 0.1};
        double Omega_n = 0.0;
        double Omega_k;
        double k_comp[41], pk_comp[6][41], diffpk, pk_z, log_pk_z;
        const char *fname[3];       
        char str[1024];
        FILE * file;
        
        fname[0]="./benchmark/model1_pk.txt";
        fname[1]="./benchmark/model2_pk.txt";
        fname[2]="./benchmark/model3_pk.txt";

        file = fopen(fname[i_model-1], "r");
        fgets(str, 1024, file);
        if (file) {
          i=0;
	  //k units are Mpc/h, Pk units are Mpc/h^3
          while (fscanf(file, "%le %le %le %le %le %le %le", &k_comp[i], &pk_comp[0][i], &pk_comp[1][i], &pk_comp[2][i], &pk_comp[3][i], &pk_comp[4][i], &pk_comp[5][i])!= EOF){
	    i++;
          }
          fclose(file);
	  nk = i;
        } else {
	  printf("Benchmark P(k) not found in ccl_test_bbks. Exit.\n");
          return 0;
	}

	ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, A_s, n_s);
	
        if(i_model > 1){
          Omega_k = 1.0 - Omega_c - Omega_b - Omega_n - Omega_v[i_model-1];
          params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Omega_n, w_0[i_model-1], w_a[i_model-1], h, A_s, n_s);
        }
	
	//overwritting sigma_8 for this test only
	params.sigma_8=sigma_8;
	printf("You just overwrote sigma8 in the params file for BBKS test.\n");
	ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);

	int status; 
  // remove this once ccl_power is switched to implicit computation!
	ccl_cosmology_compute_power_bbks(cosmo, &status); //it should already be normalized to correct sigma_8
	double D_z0 = ccl_growth_factor(cosmo, 1.);
	
        for (int i=0; i<nk ; i++){

	  k_comp[i]*=cosmo->params.h;
          j = 0;
          for (double z=0.0; z<=5.0; z+=1.0){
	    
	    pk_comp[j][i]/=pow(cosmo->params.h,3.);
	    double a = 1/(1.+z);
	    double D = ccl_growth_factor(cosmo, a);
	    status = gsl_spline_eval_e(cosmo->data.p_lin, log(k_comp[i]), NULL,&log_pk_z);
	    
            pk_z = exp(log_pk_z)*D*D/D_z0/D_z0;
            diffpk = (pk_z - pk_comp[j][i]) / pk_comp[j][i];
	    //printf("%e %e %e\n",k_comp[i],pk_z,diffpk);
            if((fabs(diffpk) >= 1e-4)){
              testflag=1;
              printf("P(k) FAIL: z:%lf, k:%lf, pk_z:%le, pk_comp: %le, ratio pk: %e\n", z, k_comp[i], pk_z, pk_comp[j][i],pk_z/pk_comp[j][i]);
            }
            j++;
	  }
        }
        if(testflag!=0){
          printf("FAIL: Values above do not match benchmark!\n");
        }
        else{
          printf("PASS!\n");
        }

	
}
