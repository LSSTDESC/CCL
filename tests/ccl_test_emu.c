#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

/*   Automated test for power spectrum emulation within CCL
     using the Lawrence et al. (2017) code.
     The test compares the smoothed simulated power spectra
     provided by the paper authors to the CCL output of the
     power spectrum via the emulator. This test corresponds
     to Figure 6 of the emulator paper, for a specific subset
     of the cosmologies: {1,3,5,6,8,10}. Other cosmologies
     are not allowed because CLASS fails when w(z) crosses -1
     and we need the linear power spectrum from CLASS in general
     for sigma8 computation.
*/


#define EMU_TOLERANCE 3.0E-2
//This is the tolerance we have required based on the emulator
//paper results (Section 3.3, Fig 6).

CTEST_DATA(emu) {
  double Neff;
  double* mnu;
  ccl_mnu_convention mnu_type;
  double sigma8[6];
  double Omega_c[6];
  double Omega_b[6];
  double n_s[6];
  double h[6];
  double w_0[6];
  double w_a[6];
};

CTEST_SETUP(emu) {

  //This test corresponds to emulator cosmologies
  //without neutrinos. Other cosmologies are
  //implemented in ccl_test_emu_nu.c
  data->Neff=3.04;
  double mnuval = 0.;
  data->mnu=&mnuval;
  data->mnu_type=ccl_mnu_sum;

  int status=0;
  char fname[256],str[1024];
  char* rtn;
  FILE *f;
  int i;

  //Each line of this file corresponds to the cosmological parameters for
  //cosmologies {1,3,5,6,8,10} of the emulator set. Notice that Omega_i
  //are big Omegas and not little omegas (Omega_i*h**2=omega_i)
  sprintf(fname,"./tests/benchmark/emu_cosmologies.txt");
  f=fopen(fname,"r");
  if(f==NULL) {
    fprintf(stderr,"Error opening file %s\n",fname);
    exit(1);
  }

  double tmp;
  for(int i=0;i<6;i++) {
    status=fscanf(f,"%le %le %le %le %le %le %le\n",&(data->Omega_c[i]),&(data->Omega_b[i]),&(data->h[i]),&(data->sigma8[i]),&(data->n_s[i]),&(data->w_0[i]),&(data->w_a[i]));
    if(status!=7) {
      fprintf(stderr,"Error reading file %s, line %d\n",fname,i);
      exit(1);
    }
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

static void compare_emu(int i_model,struct emu_data * data)
{
  int nk,i,j;
  int status=0;
  char fname[256],str[1024];
  char* rtn;
  FILE *f;
  int i_model_vec[6]={1,3,5,6,8,10};
  //The emulator cosmologies we can compare to
  //without CLASS failing due to w(z) crossing -1.

  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_emulator;
  config.matter_power_spectrum_method = ccl_emu;

  //None of the current cosmologies being checked include neutrinos
  ccl_parameters params = ccl_parameters_create(data->Omega_c[i_model-1],data->Omega_b[i_model-1],0.0,data->Neff,
						data->mnu, data->mnu_type, data->w_0[i_model-1],data->w_a[i_model-1],
						data->h[i_model-1],data->sigma8[i_model-1],data->n_s[i_model-1],
						-1,-1,-1,-1,NULL,NULL, &status);
  params.Omega_l=params.Omega_l+params.Omega_g;
  params.Omega_g=0;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);
  //Each of these files has the smoothed simulated power spectrum for
  //the corresponding cosmology, kindly provided by E. Lawrence.
  sprintf(fname,"./tests/benchmark/emu_smooth_pk_M%d.txt",i_model_vec[i_model-1]);
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
    ASSERT_DBL_NEAR_TOL(err,0.,EMU_TOLERANCE);
  }

  fclose(f);

  ccl_cosmology_free(cosmo);
}

//Cosmology M001
CTEST2(emu,model_1) {
  int model=1;
  compare_emu(model,data);
}

CTEST2(emu,model_2) {
  int model=2;
  compare_emu(model,data);
}

CTEST2_SKIP(emu,model_3) {
  int model=3;
  compare_emu(model,data);
}

CTEST2(emu,model_4) {
  int model=4;
  compare_emu(model,data);
}

CTEST2_SKIP(emu,model_5) {
  int model=5;
  compare_emu(model,data);
}

CTEST2(emu,model_6) {
  int model=6;
  compare_emu(model,data);
}
