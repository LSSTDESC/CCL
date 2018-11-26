#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"

#define NZ 1024
#define Z0_GC 1.0 
#define SZ_GC 0.02
#define NL 499

int main(int argc,char **argv)
{
  double Omega_c = 0.25;
  double Omega_b = 0.05;
  double h = 0.7;
  double A_s = 2.1e-9;
  double n_s = 0.96;
  double Omega_v = 0.7;
  double Neff=3.046;
  double mnu = 0.;
  ccl_mnu_convention mnu_type = ccl_mnu_sum;
  double w_0     = -1;
  double w_a    = 0;
  double Omega_k = 0;

  // Status flag
  int status =0;
  
  // Initialize cosmological parameters
  ccl_configuration ccl_config=default_config;
  ccl_config.transfer_function_method=ccl_boltzmann_class;
  ccl_config.matter_power_spectrum_method=ccl_linear;
  ccl_parameters ccl_params = ccl_parameters_create(Omega_c, Omega_b, Omega_k, Neff,
						    &mnu, mnu_type,w_0, w_a, h, A_s,
						    n_s,-1,-1,-1,-1,NULL,NULL, &status);
  ccl_params.Omega_g=0.;
  ccl_params.Omega_l=Omega_v;
  
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
  bool has_rsd = true;
  bool has_magnification = false;
  CCL_ClTracer *ct_gc_A=ccl_cl_tracer_number_counts(ccl_cosmo,has_rsd,has_magnification,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr,-1,NULL,NULL, &status);
  CCL_ClTracer *ct_gc_B=ccl_cl_tracer_number_counts(ccl_cosmo,has_rsd,has_magnification,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr,-1,NULL,NULL, &status);
  
  int *ells=malloc(NL*sizeof(int));
  double *cells_gg_native=malloc(NL*sizeof(double));
  for(int ii=0;ii<NL;ii++)
    ells[ii]=ii;


  // Workspaces
  double linstep = 40;
  double logstep = 1.15;
  double dchi = (ct_gc_A->chimax-ct_gc_A->chimin)/1000.; 
  double dlk = 0.003;
  double zmin = 0.05;
  CCL_ClWorkspace *wnl=ccl_cl_workspace_default(NL+1,2*ells[NL-1],CCL_NONLIMBER_METHOD_NATIVE,logstep,linstep,dchi,dlk,zmin,&status);
  
  // Compute C_ell
  ccl_angular_cls(ccl_cosmo,wnl,ct_gc_B,ct_gc_B,NL,ells,cells_gg_native,&status);
  FILE *f=fopen("angpow_gg.txt","w");
  for(int ii=2;ii<NL;ii++)
    fprintf(f,"%d %lE\n",ells[ii],cells_gg_native[ii]);
  fclose(f);
  
  //Free up tracers
  ccl_cl_tracer_free(ct_gc_A);
  ccl_cl_tracer_free(ct_gc_B);
  free(ells);
  ccl_cl_workspace_free(wnl);
  free(cells_gg_native);
  ccl_cosmology_free(ccl_cosmo);

  return 0;
}
