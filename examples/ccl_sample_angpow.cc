#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"
#include "time.h"

#define OC 0.25
#define OB 0.05
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
//#define NORMPS 0.80
#define NORMPS 2.215e-9
#define ZD 0.5
#define NZ 1024
#define Z1_GC 1.0
#define SZ1_GC 0.02
#define Z0_GC 1.0
#define SZ_GC 0.02
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 499
#define PS 0.1
#define NEFF 3.046

void print_params(int l_limber,const char *fname_params,const char *prefix_out)
{
  FILE *fl=fopen(fname_params,"w");
  fprintf(fl,"omega_m= %lf\n",OC+OB);
  fprintf(fl,"omega_l= %lf\n",1-OC-OB);
  fprintf(fl,"omega_b= %lf\n",OB);
  fprintf(fl,"w0= %lf\n",W0);
  fprintf(fl,"wa= %lf\n",WA);
  fprintf(fl,"h= %lf\n",HH);
  fprintf(fl,"ns= %lf\n",NS);
  fprintf(fl,"s8= %lf\n",NORMPS);
  fprintf(fl,"l_limber_min= %d\n",l_limber);
  fprintf(fl,"d_chi= 3.\n");
  fprintf(fl,"z_kappa= 20.\n");
  fprintf(fl,"z_isw= 20.\n");
  fprintf(fl,"l_max= %d\n",NL);
  fprintf(fl,"do_nc= 1\n");
  fprintf(fl,"has_nc_dens= 1\n");
  fprintf(fl,"has_nc_rsd= 0\n");
  fprintf(fl,"has_nc_lensing= 0\n");
  fprintf(fl,"do_shear= 0\n");
  fprintf(fl,"has_sh_intrinsic= 0\n");
  fprintf(fl,"do_cmblens= 0\n");
  fprintf(fl,"do_isw= 0\n");
  fprintf(fl,"do_w_theta= 0\n");
  fprintf(fl,"use_logbin= 0\n");
  fprintf(fl,"theta_min= 0\n");
  fprintf(fl,"theta_max= 0\n");
  fprintf(fl,"n_bins_theta= 0\n");
  fprintf(fl,"n_bins_decade= 0\n");
  fprintf(fl,"window_1_fname= nz.txt\n");
  fprintf(fl,"window_2_fname= nz.txt\n");
  fprintf(fl,"bias_fname= bz.txt\n");
  fprintf(fl,"sbias_fname= nothing\n");
  fprintf(fl,"abias_fname= nothing\n");
  fprintf(fl,"pk_fname= pk.txt\n");
  fprintf(fl,"prefix_out= %s\n",prefix_out);
  fclose(fl);
}

using namespace std;
int main(int argc,char **argv)
{
  //status flag
  int status =0;
  // Initialize cosmological parameters
  ccl_configuration config=default_config;
  config.transfer_function_method=ccl_boltzmann_class;
  config.matter_power_spectrum_method=ccl_linear;

  // Set neutrino masses
  double* MNU;
  double mnuval = 0.;
  MNU = &mnuval;
  ccl_mnu_convention MNUTYPE = ccl_mnu_sum;
  ccl_parameters params = ccl_parameters_create(OC, OB, OK, NEFF, MNU, MNUTYPE, W0, WA, HH, NORMPS, NS,-1,-1,-1,-1,NULL,NULL, &status);

  // Initialize cosmology object given cosmo params
  ccl_cosmology *cosmo=ccl_cosmology_create(params,config);

  //Create tracers for angular power spectra
  FILE *fn=fopen("nz.txt","w");
  FILE *fb=fopen("bz.txt","w");
  double z_arr_gc[NZ],nz_arr_gc[NZ],bz_arr[NZ];
  double z1_arr_gc[NZ],nz1_arr_gc[NZ],bz1_arr[NZ];
  for(int i=0;i<NZ;i++) {
    z_arr_gc[i]=Z0_GC-5*SZ_GC+10*SZ_GC*(i+0.5)/NZ;
    nz_arr_gc[i]=exp(-0.5*pow((z_arr_gc[i]-Z0_GC)/SZ_GC,2))/sqrt(2*3.1415926536*SZ_GC*SZ_GC);
    z1_arr_gc[i]=Z1_GC-5*SZ1_GC+10*SZ1_GC*(i+0.5)/NZ;
    nz1_arr_gc[i]=exp(-0.5*pow((z1_arr_gc[i]-Z1_GC)/SZ1_GC,2))/sqrt(2*3.1415926536*SZ1_GC*SZ1_GC);
    bz_arr[i]=1.;//+z_arr_gc[i];
    fprintf(fn,"%lE %lE\n",z_arr_gc[i],nz_arr_gc[i]);
    fprintf(fb,"%lE %lE\n",z_arr_gc[i],bz_arr[i]);
  }
  fclose(fn);
  fclose(fb);

  FILE *fp=fopen("pk.txt","w");
  for(int i=0;i<1024;i++) {
    double lk=-4.+5*(i+0.5)/1024.;
    double kk=pow(10.,lk);
    double pk=ccl_nonlin_matter_power(cosmo,kk,1.,&status);
    fprintf(fp,"%lE %lE\n",kk,pk);
  }
  fclose(fp);

  print_params(-1,"params_lj_limber.ini","out_lj_limber");
  print_params(NL,"params_lj_nonlimber.ini","out_lj_nonlimber");

  //Galaxy clustering tracer
  CCL_ClTracer *ct_gc_A=ccl_cl_tracer_number_counts(cosmo,1,0,NZ,z_arr_gc,nz_arr_gc,
							NZ,z_arr_gc,bz_arr,-1,NULL,NULL,&status);
  CCL_ClTracer *ct_gc_B=ccl_cl_tracer_number_counts(cosmo,1,0,NZ,z_arr_gc,nz_arr_gc,
							NZ,z_arr_gc,bz_arr,-1,NULL,NULL,&status);
  int ells[NL];
  double cells_gg_angpow[NL];
  double cells_gg_native[NL];
  double cells_gg_limber[NL];
  for(int ii=0;ii<NL;ii++)
    ells[ii]=ii;

  double linstep = 40;
  double logstep = 1.3;
  double dchi = (ct_gc_A->chimax-ct_gc_A->chimin)/500.; // must be below 3 to converge toward limber computation at high ell
  double dlk = 0.003;
  double zmin = 0.05;
  CCL_ClWorkspace *wyl=ccl_cl_workspace_default(NL+1,-1          ,CCL_NONLIMBER_METHOD_ANGPOW,logstep,linstep,dchi,dlk,zmin,&status);
  CCL_ClWorkspace *wnl=ccl_cl_workspace_default(NL+1,2*ells[NL-1],CCL_NONLIMBER_METHOD_NATIVE,logstep,linstep,dchi,dlk,zmin,&status);
  CCL_ClWorkspace *wap=ccl_cl_workspace_default(NL+1,2*ells[NL-1],CCL_NONLIMBER_METHOD_ANGPOW,logstep,linstep,dchi,dlk,zmin,&status);
  double start = clock();
  ccl_angular_cls(cosmo,wyl,ct_gc_A,ct_gc_A,NL,ells,cells_gg_limber,&status);
  double end = clock();
  printf("Limber: %.6f seconds\n", (end-start)/CLOCKS_PER_SEC);
  start = clock();
  ccl_angular_cls(cosmo,wnl,ct_gc_B,ct_gc_B,NL,ells,cells_gg_native,&status);
  end = clock();
  printf("Native: %.6f seconds\n", (end-start)/CLOCKS_PER_SEC);
  start = clock();
  ccl_angular_cls(cosmo,wap,ct_gc_A,ct_gc_A,NL,ells,cells_gg_angpow,&status);
  end = clock();
  printf("Angpow: %.6f seconds\n", (end-start)/CLOCKS_PER_SEC);
  printf("Done\n");

  FILE *fo=fopen("cls_val.txt","w");
  for(int ii=0;ii<NL;ii++) {
    double cl_gg_yl=cells_gg_limber[ii];
    double cl_gg_nl=cells_gg_native[ii];
    double cl_gg_ap=cells_gg_angpow[ii];
    fprintf(fo,"%d %lE %lE %lE\n",ells[ii],cl_gg_yl,cl_gg_nl,cl_gg_ap);
  }
  fclose(fo);

  //Free up tracers
  ccl_cl_tracer_free(ct_gc_A);
  ccl_cl_tracer_free(ct_gc_B);
  ccl_cl_workspace_free(wap);
  ccl_cl_workspace_free(wnl);
  ccl_cl_workspace_free(wyl);

  //Always clean up!!
  ccl_cosmology_free(cosmo);

  return 0;
}
