#include "ccl.h"
#include "ccl_neutrinos.h" //Needed for Omeganu->M computation
#include "ctest.h"
#include <stdio.h>
#include <math.h>

/*   Automated test for power spectrum emulation within CCL 
     using the Lawrence et al. (2017) code.
     The test compares the smoothed simulated power spectra
     provided by the paper authors to the CCL output of the 
     power spectrum via the emulator. This test corresponds
     to Figure 5 of the emulator paper, for a specific subset 
     of the cosmologies: {38,39,40,42}. Other cosmologies
     are not allowed because CLASS fails when w(z) crosses -1
     and we need the linear power spectrum from CLASS in general
     for sigma8 computation.
*/

#define EMU_TOLERANCE 3.0E-2
//This is the tolerance we have required based on the emulator
//paper results (Fig 5).

CTEST_DATA(emu_nu) {
  double Neff;
  double *mnu[4];
  ccl_mnu_convention mnu_type;
  double sigma8[4];
  double Omega_c[4];
  double Omega_b[4];
  double n_s[4];
  double h[4];
  double w_0[4];
  double w_a[4];
};

CTEST_SETUP(emu_nu) {
  
  data->Neff = 3.04;
  data->mnu_type = ccl_mnu_list;

  double *sigma8;
  double *Omega_c;
  double *Omega_b;
  double *Omega_nu;
  double *n_s;
  double *h;
  double *w_0;
  double *w_a;
  double *Mnu_out;
  int status=0;
  char fname[256],str[1024];
  char* rtn;
  FILE *f;
  int i;
  ccl_parameters * params;

  sigma8=malloc(4*sizeof(double));
  Omega_c=malloc(4*sizeof(double));
  Omega_b=malloc(4*sizeof(double));
  n_s=malloc(4*sizeof(double));
  h=malloc(4*sizeof(double));
  w_0=malloc(4*sizeof(double));
  w_a=malloc(4*sizeof(double));
  Omega_nu=malloc(4*sizeof(double));
  
  // Omeganuh2_to_mnu will output a pointer to an array of 3 neutrino masses.
  Mnu_out=malloc(3*sizeof(double));

  //Each line of this file corresponds to the cosmological parameters for
  //cosmologies {38,39,40,42} of the emulator set. Notice that Omega_i
  //are big Omegas and not little omegas (Omega_i*h**2=omega_i)
  sprintf(fname,"./tests/benchmark/emu_nu_cosmologies.txt");
  f=fopen(fname,"r");
  if(f==NULL) {
    fprintf(stderr,"Error opening file %s\n",fname);
    exit(1);
  }
  
  double tmp;
  int omnustatus=0;
  for(int i=0;i<4;i++) {
    
    status=fscanf(f,"%le %le %le %le %le %le %le %le\n",&Omega_c[i],&Omega_b[i],&h[i],&sigma8[i],&n_s[i],&w_0[i],&w_a[i],&Omega_nu[i]);
    if(status!=8) {
      fprintf(stderr,"Error reading file %s, line %d\n",fname,i);
      exit(1);
    }
    data->w_0[i] = w_0[i];
    data->w_a[i] = w_a[i];
    data->h[i] = h[i];
    data->sigma8[i] = sigma8[i];
    data->Omega_c[i] = Omega_c[i];
    data->Omega_b[i] = Omega_b[i];
    data->n_s[i] = n_s[i];
    // Number of neutrino species is fixed to 3
    Mnu_out = ccl_nu_masses(Omega_nu[i]*h[i]*h[i], ccl_nu_equal, 2.725, &omnustatus);
    /*if (omnustatus){
      printf("%s\n",cosmo->status_message);
      exit(1);
      }*/
    data->mnu[i]=Mnu_out;
  }
  fclose(f);
}

static int linecount(FILE *f)
{
  //////
  // Counts #lines from file
  int i0=0;
  char ch[1000];
  while((fgets(ch,sizeof(ch),f))!=NULL) {
    i0++;
  }
  return i0;
}

static void compare_emu_nu(int i_model,struct emu_nu_data * data)
{
  int nk,i,j;
  int status=0;
  char fname[256],str[1024];
  char* rtn;
  FILE *f;
  int i_model_vec[4]={38,39,40,42}; //The emulator cosmologies we can compare to 
  
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_emulator;
  config.matter_power_spectrum_method = ccl_emu;
 
  //None of the current cosmologies being checked include neutrinos
  ccl_parameters params = ccl_parameters_create(data->Omega_c[i_model-1],data->Omega_b[i_model-1],0.0,data->Neff, data->mnu[i_model-1], data->mnu_type, data->w_0[i_model-1],data->w_a[i_model-1],data->h[i_model-1],data->sigma8[i_model-1],data->n_s[i_model-1],-1,-1,-1,-1,NULL,NULL, &status);
  params.Omega_l=params.Omega_l+params.Omega_g;
  params.Omega_g=0;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);

  //These files contain the smoothed power spectra with neutrinos
  //These are obtained as follows:
  // (1) Find the z=0 P(k) corresponding to M038. This would be column 304 of the 
  //     'yalt...' file (with column numbering starting in 1).
  // (2) This column has log10(Delta_cb^2/k^1.5), so convert accordingly to Delta_cb^2.
  // (3) Take the power spectrum in the file ‘pk_lin...' for M038 (second column for P(k) in Mpc^3)
  //     and convert to Delta_nu^2 by multiplying by k^3/(2pi^2).
  // (4) Add to obtain Delta_tot=(sqrt(Delta_nu)+sqrt(Delta_cb))^2.
  // (5) Convert back to P(k)
  sprintf(fname,"./tests/benchmark/emu_nu_smooth_pk_M%d.txt",i_model_vec[i_model-1]);
  f=fopen(fname,"r");
  if(f==NULL) {
    fprintf(stderr,"Error opening file %s\n",fname);
    exit(1);
  }
  nk=linecount(f)-1; rewind(f);
  
  double k=0.,pk_bench=0.,pk_ccl,err;
  double z=0.; //Other redshift checks are possible but not currently implemented
  int stat=0;
  
  for(i=0;i<nk;i++) {      
    stat=fscanf(f,"%le %le\n",&k, &pk_bench);
    if(stat!=2) {
      fprintf(stderr,"Error reading file %s, line %d\n",fname,i);
      exit(1);
    }
    pk_ccl=ccl_nonlin_matter_power(cosmo,k,1./(1+z),&status);
    if (status) printf("%s\n",cosmo->status_message);
    err=fabs(pk_ccl/pk_bench-1);
    //printf("%le %le %le %le\n",k,pk_ccl,pk_bench,err);
    ASSERT_DBL_NEAR_TOL(err,0.,EMU_TOLERANCE);
  }

  fclose(f);

  ccl_cosmology_free(cosmo);
}

CTEST2(emu_nu,model_1) {
  int model=1;
  compare_emu_nu(model,data);
}


CTEST2(emu_nu,model_2) {
  int model=2;
  compare_emu_nu(model,data);
}

CTEST2(emu_nu,model_3) {
  int model=3;
  compare_emu_nu(model,data);
}

CTEST2(emu_nu,model_4) {
  int model=4;
  compare_emu_nu(model,data);
}
