#include "ccl.h"
#include <stdio.h>
#include <math.h>

int main(int argc, char * argv[]){
  
        if(argv[1]==NULL){
          printf("No model argument given - program exiting!\n Integer arguments between 1 and 5 are allowed.\n");
          return 0;
        }
        int i_model=atoi(argv[1]);
        if( (i_model>5) || (i_model<1)){
          printf("Model argument not between 1 and 5 - program exiting!\n Integer arguments between 1 and 5 are allowed.\n");
          return 0;
        }
        int i, testflag=0;
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
        double z_comp[6], chi_comp[6], gf_comp[6], diffchi, diffgf;
//        const char *fname[5];       
        double temp[5];
        char str[1024];
        FILE * chifile, * gffile;
        
        chifile = fopen("./benchmark/chi_model1-5.txt", "r");
        fgets(str, 1024, chifile);
        if (chifile) {
          i=0;
          while (fscanf(chifile, "%le %le %le %le %le %le\n", &z_comp[i], &temp[0], &temp[1], &temp[2], &temp[3], &temp[4])!=EOF){
            chi_comp[i] = temp[i_model-1];
            i++;
          }
        }
 
        gffile = fopen("./benchmark/growth_model1-5.txt", "r");
        fgets(str, 1024, gffile);
        if (gffile) {
          i=0;
          while (fscanf(gffile, "%le %le %le %le %le %le\n", &z_comp[i], &temp[0], &temp[1], &temp[2], &temp[3], &temp[4])!=EOF){
            gf_comp[i] = temp[i_model-1];
            i++;
          }
        }
/* 
        fname[0]="./benchmark/model1_chi.txt";
        fname[1]="./benchmark/model2_chi.txt";
        fname[2]="./benchmark/model3_chi.txt";
        fname[3]="./benchmark/model4_chi.txt";
        fname[4]="./benchmark/model5_chi.txt";

        file = fopen(fname[i_model-1], "r");
        fgets(str, 1024, file);
        if (file) {
          i=0;
          while (fscanf(file, "%le %le", &z_comp[i], &chi_comp[i])!= EOF){
            i++;
          }
          fclose(file);
        }
*/        
       
	ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, A_s, n_s);

        if(i_model > 1){
          Omega_k = 1.0 - Omega_c - Omega_b - Omega_n - Omega_v[i_model-1];
          params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Omega_n, w_0[i_model-1], w_a[i_model-1], h, A_s, n_s);
        }
	ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);


        printf("i_model == %i\n", i_model);
        printf("Omega_v == %lf, w_0 == %lf, w_a == %lf\n", Omega_v[i_model-1], w_0[i_model-1], w_a[i_model-1]);
	printf("[0]z, [1]chi(z), [2]dL(z), [3]mu(z), [4]D(z), [5]f(z)\n");
        i = 0;
	for (double z=0.0; z<=5.0; z+=1.0){
	  int st;
		double a = 1/(1.+z);
		//double DL = 1/h * ccl_luminosity_distance(cosmo, a);
		//double DL_pc = DL*1e6;
		//double mu = 5*log10(DL_pc)-5;
		double chi=ccl_comoving_radial_distance(cosmo,a);
		double gf=ccl_growth_factor(cosmo,a,&st);
		//double fg=ccl_growth_rate(cosmo,a,&st);
                diffchi = (chi - chi_comp[i]) / chi_comp[i];
                diffgf = (gf - gf_comp[i]) / gf_comp[i];
                if((fabs(diffchi) >= 1e-4) && (z>0.0)){
                  testflag=1;
                  printf("CHI FAIL: z:%lf, chi:%le, chi_comp: %le\n", z, chi, chi_comp[i]);
                }
                if(fabs(diffgf) >= 1e-4){
                  testflag=1;
                  printf("GROWTH FAIL: z:%lf, gf:%le, gf_comp: %le\n", z, gf, gf_comp[i]);
                }
                i++;
		//printf("%le  %le %le %le %le %le\n", z, chi,h*DL,mu,gf,fg);
	}
        if(testflag!=0){
          printf("FAIL: Values above do not match benchmark!\n");
        }
        else{
          printf("PASS!\n");
        }

}
