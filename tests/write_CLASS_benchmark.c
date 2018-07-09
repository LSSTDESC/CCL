#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

#define NREL 3.046
#define NMAS 0
#define MNU 0.0

void write_Pk_CLASS(int i_model)
{
  int status=0;
  char fname[256];
  FILE *f;
  int i_model_vec[6]={1,3,5,6,8,10}; //The emulator cosmologies we can compare to
  double z = 0.0;

  double *sigma8;
  double *Omega_c;
  double *Omega_b;
  double *n_s;
  double *h;
  double *w_0;
  double *w_a;
  sigma8=malloc(6*sizeof(double));
  Omega_c=malloc(6*sizeof(double));
  Omega_b=malloc(6*sizeof(double));
  n_s=malloc(6*sizeof(double));
  h=malloc(6*sizeof(double));
  w_0=malloc(6*sizeof(double));
  w_a=malloc(6*sizeof(double));
  
  sprintf(fname,"./tests/benchmark/emu_cosmologies.txt");
  f=fopen(fname,"r");
  if(f==NULL) {
    fprintf(stderr,"Error opening file %s\n",fname);
    exit(1);
  }
  
  for(int i=0;i<6;i++) {
    
    status=fscanf(f,"%le %le %le %le %le %le %le\n",&Omega_c[i],&Omega_b[i],&h[i],&sigma8[i],&n_s[i],&w_0[i],&w_a[i]);
    if(status!=7) {
      fprintf(stderr,"Error reading file %s, line %d\n",fname,i);
      exit(1);
    }
  }
  fclose(f);


  ccl_configuration config = default_config;
  status = 0;
  //None of the current cosmologies being checked include neutrinos
  ccl_parameters params = ccl_parameters_create(Omega_c[i_model-1],Omega_b[i_model-1],0.0,NREL, NMAS, MNU,w_0[i_model-1],w_a[i_model-1],h[i_model-1],sigma8[i_model-1],n_s[i_model-1],-1,NULL,NULL, &status);
  params.Omega_g=0;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);  
  sprintf(fname,"./tests/benchmark/CLASS_Pk_M%d.txt",i_model_vec[i_model-1]);
  ccl_cosmology_write_power_class_z(fname, cosmo, z, &status);
  printf("Wrote %s, status = %d\n", fname,status);
  ccl_cosmology_free(cosmo);
}

int main (void){
  for (int i = 1; i < 7; i++){
	write_Pk_CLASS(i);
  }
}
