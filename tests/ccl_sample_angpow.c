#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"

#define OC 0.25
#define OB 0.05
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define NORMPS 0.80
#define ZD 0.5
#define NZ 1024
#define Z0_GC 0.80
#define SZ_GC 0.05
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 512

void print_params(int l_limber,char *fname_params,char *prefix_out)
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

int main(int argc,char **argv)
{
  //status flag
  int status =0;
  // Initialize cosmological parameters
  ccl_configuration config=default_config;
  config.transfer_function_method=ccl_bbks;
  ccl_parameters params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,NORMPS,NS,-1,NULL,NULL);

  // Initialize cosmology object given cosmo params
  ccl_cosmology *cosmo=ccl_cosmology_create(params,config);
  // Compute radial distances (see include/ccl_background.h for more routines)
  //  printf("Comoving distance to z = %.3lf is chi = %.3lf Mpc\n",
  //	 ZD,ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status));
  //  printf("Luminosity distance to z = %.3lf is chi = %.3lf Mpc\n",
  //	 ZD,ccl_luminosity_distance(cosmo,1./(1+ZD), &status));

  //Create tracers for angular power spectra
  FILE *fn=fopen("nz.txt","w");
  FILE *fb=fopen("bz.txt","w");
  double z_arr_gc[NZ],nz_arr_gc[NZ],bz_arr[NZ];
  for(int i=0;i<NZ;i++) {
    z_arr_gc[i]=Z0_GC-5*SZ_GC+10*SZ_GC*(i+0.5)/NZ;
    nz_arr_gc[i]=exp(-0.5*pow((z_arr_gc[i]-Z0_GC)/SZ_GC,2))/sqrt(2*3.1415926536*SZ_GC*SZ_GC);
    bz_arr[i]=1.;//+z_arr_gc[i];
    fprintf(fn,"%lE %lE\n",z_arr_gc[i],nz_arr_gc[i]);
    fprintf(fb,"%lE %lE\n",z_arr_gc[i],bz_arr[i]);
  }
  fclose(fn);
  fclose(fb);

  printf("%lE \n",ccl_sigma8(cosmo,&status));

  FILE *fp=fopen("pk.txt","w");
  for(int i=0;i<1024;i++) {
    double lk=-4.+5*(i+0.5)/1024.;
    double kk=pow(10.,lk);
    double pk=ccl_nonlin_matter_power(cosmo,kk*HH,1.,&status);
    fprintf(fp,"%lE %lE\n",kk,pk);
  }
  fclose(fp);

  print_params(-1,"params_lj_limber.ini","out_lj_limber");
  print_params(NL,"params_lj_nonlimber.ini","out_lj_nonlimber");

  //Galaxy clustering tracer
  CCL_ClTracer *ct_gc=ccl_cl_tracer_number_counts_new(cosmo,0,0,NZ,z_arr_gc,nz_arr_gc,
						      NZ,z_arr_gc,bz_arr,-1,NULL,NULL,&status);
  int *ells=malloc(NL*sizeof(int));
  double *cells_gg_angpow=malloc(NL*sizeof(double));
  double *cells_gg_limber=malloc(NL*sizeof(double));
  for(int ii=0;ii<NL;ii++)
    ells[ii]=ii;
  ccl_angular_cls(cosmo,ct_gc,ct_gc,NL,ells,cells_gg_limber, -1,&status);
  ccl_angular_cls(cosmo,ct_gc,ct_gc,NL,ells,cells_gg_angpow,ells[NL-1]+1,&status);
  //  ccl_angular_cls(cosmo,ct_gc,ct_gc,NL,ells,cells_gg_angpow,200,&status);

  FILE *fo=fopen("tests/cls_val.txt","w");
  //  printf("ell C_ell(g,g) C_ell(g,s) C_ell(s,s) | r(g,s)\n");
  for(int ii=0;ii<NL;ii++) {
    double cl_gg=cells_gg_limber[ii];
    double cl_gg_ap=cells_gg_angpow[ii];
    fprintf(fo,"%d %.3lE %.3lE\n",ells[ii],cl_gg,cl_gg_ap);
  }
  //  printf("\n");
  fclose(fo);
  free(ells); free(cells_gg_angpow); free(cells_gg_limber);

  //Free up tracers
  ccl_cl_tracer_free(ct_gc);

  //Always clean up!!
  ccl_cosmology_free(cosmo);
  
  return 0;
}
