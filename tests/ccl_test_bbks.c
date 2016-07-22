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
  	double n_s = 0.96;
        double Omega_v[5] = {0.7, 0.7, 0.7, 0.65, 0.75};
        double w_0[5] = {-1.0, -0.9, -0.9, -0.9, -0.9};
        double w_a[5] = {0.0, 0.0, 0.1, 0.1, 0.1};
        double Omega_n = 0.0;
        double Omega_k;
        double z_comp[6] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        double k_comp[41], pk_comp[6][41], diffpk, pk_z, pk_z0, temp;
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
          while (fscanf(file, "%le %le %le %le %le %le %le", &k_comp[i], &pk_comp[0][i], &pk_comp[1][i], &pk_comp[2][i], &pk_comp[3][i], &pk_comp[4][i], &pk_comp[5][i])!= EOF){
            i++;
          }
          fclose(file);
        }
        nk = i;

        printf("read complete\n");

	ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, A_s, n_s);

        if(i_model > 1){
          Omega_k = 1.0 - Omega_c - Omega_b - Omega_n - Omega_v[i_model-1];
          params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Omega_n, w_0[i_model-1], w_a[i_model-1], h, A_s, n_s);
        }
	ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);

	int status; 
        ccl_cosmology_compute_distances(cosmo, &status);

        for (int i=0; i<nk ; i++){
          pk_z0 = ccl_bbks_power(&params, k_comp[i]);        
          j = 0;
          for (double z=0.0; z<=5.0; z+=1.0){
	    int st;
	    double a = 1/(1.+z);
	    double gf=ccl_growth_factor(cosmo,a,&st);
            pk_z = pk_z0*gf*gf;
            diffpk = (pk_z - pk_comp[j][i]) / pk_comp[j][i];
            if((fabs(diffpk) >= 1e-4)){
              testflag=1;
              printf("P(k) FAIL: k:%lf, z:%lf, pk_z:%le, pk_comp: %le\n", z, k_comp[i], pk_z, pk_comp[j][i]);
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
/*
        printf("i_model == %i\n", i_model);
        printf("Omega_v == %lf, w_0 == %lf, w_a == %lf\n", Omega_v[i_model-1], w_0[i_model-1], w_a[i_model-1]);
	printf("[0]z, [1]chi(z), [2]dL(z), [3]mu(z), [4]D(z), [5]f(z)\n");
*/
}
