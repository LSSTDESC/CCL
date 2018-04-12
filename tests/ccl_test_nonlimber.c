#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

#define NZ 1024
#define Z0_GC 1.0 
#define SZ_GC 0.1
#define NL 999

#define CLS_PRECISION 2E-2 // with respect to cosmic variance


CTEST_DATA(nonlimber) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double Neff;
  double* mnu;
  ccl_mnu_convention mnu_type;
  double Omega_n;
  double Omega_v;
  double Omega_k;
  double w_0;
  double w_a;
};




// Set up the cosmological parameters to be used 
CTEST_SETUP(nonlimber){
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->Omega_n = 0.0;
  data->Omega_v = 0;
  data->Neff=3.046;
  double mnuval = 0.;
  data->mnu = &mnuval;
  data->mnu_type = ccl_mnu_sum;
  data->w_0     = -1;
  data->w_a    = 0;
  data->Omega_k = 0;
}





static void test_nonlimber_precision(struct nonlimber_data * data)
{
  // Status flag
  int status =0;
  
  // Initialize cosmological parameters
  ccl_configuration ccl_config=default_config;
  ccl_config.transfer_function_method=ccl_boltzmann_class;
  ccl_config.matter_power_spectrum_method=ccl_linear;
  ccl_parameters ccl_params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k, data->Neff, data->mnu, data->mnu_type,data->w_0, data->w_a, data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);

  // Initialize cosmology object given cosmo params
  ccl_cosmology *ccl_cosmo=ccl_cosmology_create(ccl_params,ccl_config);

  // Create tracers for angular power spectra
  double z_arr_gc[NZ],nz_arr_gc[NZ],bz_arr[NZ],sz_arr[NZ];
  for(int i=0;i<NZ;i++)
    {
      z_arr_gc[i]=Z0_GC-5*SZ_GC+10*SZ_GC*(i+0.5)/NZ;
      nz_arr_gc[i]=exp(-0.5*pow((z_arr_gc[i]-Z0_GC)/SZ_GC,2));
      bz_arr[i]=1;
      sz_arr[i]=exp(-0.5*pow((z_arr_gc[i]-Z0_GC)/SZ_GC,2));
    }
  
  // Galaxy clustering tracer
  bool has_rsd = true;
  bool has_magnification = false;
  CCL_ClTracer *ct_gc_A=ccl_cl_tracer_number_counts(ccl_cosmo,has_rsd,has_magnification,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr,-1,NULL,NULL, &status);
  CCL_ClTracer *ct_gc_B=ccl_cl_tracer_number_counts(ccl_cosmo,has_rsd,has_magnification,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr,-1,NULL,NULL, &status);
  
  int *ells=malloc(NL*sizeof(int));
  double *cells_gg_native=malloc(NL*sizeof(double));
  double *cells_gg_limber=malloc(NL*sizeof(double));
  for(int ii=0;ii<NL;ii++)
    ells[ii]=ii;


  // Workspaces
  double linstep = 40;
  double logstep = 1.3;
  double dchi = (ct_gc_A->chimax-ct_gc_A->chimin)/500.; 
  double dlk = 0.003;
  double zmin = 0.05;
  CCL_ClWorkspace *wlim=ccl_cl_workspace_default_limber(NL+1,logstep,linstep,dlk,&status);
  CCL_ClWorkspace *wnl=ccl_cl_workspace_default(NL+1,2*ells[NL-1],CCL_NONLIMBER_METHOD_NATIVE,logstep,linstep,dchi,dlk,zmin,&status);

  
  // Compute C_ell
  ccl_angular_cls(ccl_cosmo,wlim,ct_gc_A,ct_gc_A,NL,ells,cells_gg_limber,&status);
  ccl_angular_cls(ccl_cosmo,wnl,ct_gc_B,ct_gc_B,NL,ells,cells_gg_native,&status);

  double rel_precision = 0.;
  for(int ii=2;ii<NL;ii++) {
    int l = ells[ii];
    double cl_gg_nl=cells_gg_native[ii];
    double cl_gg_lim=cells_gg_limber[ii];
    double ratio = fabs(cl_gg_nl-cl_gg_lim)/cl_gg_nl;
    if(l>NL/2)
      rel_precision += ratio / sqrt(2./(2*l+1));
  }
  rel_precision /= NL/2;
  ASSERT_TRUE(rel_precision < CLS_PRECISION);

  
  //Free up tracers
  ccl_cl_tracer_free(ct_gc_A);
  ccl_cl_tracer_free(ct_gc_B);
  free(ells);
  ccl_cl_workspace_free(wnl);
  ccl_cl_workspace_free(wlim);
  free(cells_gg_native);
  free(cells_gg_limber);
  ccl_cosmology_free(ccl_cosmo);  
}

CTEST2(nonlimber,precision) {
  test_nonlimber_precision(data);
}
