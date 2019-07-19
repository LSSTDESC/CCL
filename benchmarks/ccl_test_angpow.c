#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

#define NZ 1024
#define Z0_GC 1.0 
#define SZ_GC 0.02
#define NL 499

#define CLS_PRECISION 3E-3 

CTEST_DATA(angpow) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double Neff;
  double* mnu;
  ccl_mnu_convention mnu_type;
  double Omega_v;
  double Omega_k;
  double w_0;
  double w_a;
  double mu_0;
  double sigma_0;
};




// Set up the cosmological parameters to be used 
CTEST_SETUP(angpow){
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->Omega_v = 0.7;
  data->Neff=3.046;
  double mnuval = 0.;
  data->mnu = &mnuval;
  data->mnu_type = ccl_mnu_sum;
  data->w_0     = -1;
  data->w_a    = 0;
  data->Omega_k = 0;
  data->mu_0 = 0.;
  data->sigma_0 = 0.;
}



static void test_angpow_precision(struct angpow_data * data)
{
  // Status flag
  int status =0;
  
  // Initialize cosmological parameters
  ccl_configuration ccl_config=default_config;
  ccl_config.transfer_function_method=ccl_boltzmann_class;
  ccl_config.matter_power_spectrum_method=ccl_linear;
  ccl_parameters ccl_params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k,
						    data->Neff, data->mnu, data->mnu_type,data->w_0,
						    data->w_a, data->h, data->A_s, data->n_s,
						    -1,-1,-1,data->mu_0, data->sigma_0,-1,NULL,NULL, &status);
  ccl_params.Omega_g=0.;
  ccl_params.Omega_l=data->Omega_v;

  // Initialize cosmology object given cosmo params
  ccl_cosmology *ccl_cosmo=ccl_cosmology_create(ccl_params,ccl_config);

  // Create tracers for angular power spectra
  double z_arr_gc[NZ],nz_arr_gc[NZ],bz_arr[NZ];
  for(int i=0;i<NZ;i++)
    {
      z_arr_gc[i]=Z0_GC-5*SZ_GC+10*SZ_GC*(i+0.5)/NZ;
      nz_arr_gc[i]=exp(-0.5*pow((z_arr_gc[i]-Z0_GC)/SZ_GC,2));
      bz_arr[i]=1;
    }
  
  // Galaxy clustering tracer
  CCL_ClTracer *ct_gc_A=ccl_cl_tracer_number_counts(ccl_cosmo,1,1,0,
						    NZ,z_arr_gc,nz_arr_gc,
						    NZ,z_arr_gc,bz_arr,
						    -1,NULL,NULL, &status);
  CCL_ClTracer *ct_gc_B=ccl_cl_tracer_number_counts(ccl_cosmo,1,1,0,
						    NZ,z_arr_gc,nz_arr_gc,
						    NZ,z_arr_gc,bz_arr,
						    -1,NULL,NULL, &status);
  
  int *ells=malloc(NL*sizeof(int));
  double *cells_gg_angpow=malloc(NL*sizeof(double));
  for(int ii=0;ii<NL;ii++)
    ells[ii]=ii;


  // Workspaces
  double linstep = 40;
  double logstep = 1.15;
  
  // Compute C_ell
  ccl_angular_cls_nonlimber(ccl_cosmo,logstep,linstep,
			    ct_gc_A,ct_gc_A,NULL,
			    NL,ells,cells_gg_angpow,&status);
  double rel_precision = 0.;
  FILE *f=fopen("./benchmarks/data/angpow_gg.txt","r");
  for(int ii=2;ii<NL;ii++) {
    int l;
    double ratio,cl_gg_nl,cl_gg_ap=cells_gg_angpow[ii];
    int stat=fscanf(f,"%d %lE",&l,&cl_gg_nl);
    ASSERT_TRUE(l==ells[ii]);
    ASSERT_TRUE(stat==2);
    ratio = fabs(cl_gg_nl-cl_gg_ap)/cl_gg_nl;
    rel_precision += ratio;
  }
  fclose(f);
  rel_precision /= NL;
  ASSERT_TRUE(rel_precision < CLS_PRECISION);
  
  //Free up tracers
  ccl_cl_tracer_free(ct_gc_A);
  ccl_cl_tracer_free(ct_gc_B);
  free(ells);
  free(cells_gg_angpow);
  ccl_cosmology_free(ccl_cosmo);  
}

CTEST2(angpow,precision) {
  test_angpow_precision(data);
}
