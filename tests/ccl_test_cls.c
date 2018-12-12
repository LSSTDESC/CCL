#include "ccl.h"
#include "../include/ccl_params.h"
#include "ctest.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define SZ_VAL 0.4 //This will cancel the magnification contribution
#define CLS_TOLERANCE 1E-3
#define ELS_TOLERANCE 0.1
#define NELLS 3001

CTEST_DATA(cls) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double sigma8;
};

CTEST_SETUP(cls) {
  data->Omega_c = 0.30;
  data->Omega_b = 0.00;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->sigma8=0.8;
  data->n_s = 0.96;
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

static void compare_cls(char *compare_type,struct cls_data * data)
{
  int status=0;
  double zlss=1100.;

  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  config.matter_power_spectrum_method = ccl_linear;
  ccl_parameters params = ccl_parameters_create_flat_lcdm(data->Omega_c,data->Omega_b,data->h,
							  data->A_s,data->n_s, &status);
  params.Omega_n_rel=0;
  params.Omega_l = 0.7;
  params.sigma8=data->sigma8;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);

  double epsrel_save;
  if(!strcmp(compare_type,"histo")) { //This is needed for the histogrammed N(z) in order to pass the IA tests
    epsrel_save=ccl_gsl->INTEGRATION_LIMBER_EPSREL;
    ccl_gsl->INTEGRATION_LIMBER_EPSREL=2.5E-5;
    ccl_gsl->INTEGRATION_EPSREL=2.5E-5;
    ccl_set_debug_policy(CCL_DEBUG_MODE_OFF);
  }

  int nz;
  double *zarr_1,*pzarr_1,*zarr_2,*pzarr_2,*bzarr,*az1arr,*rz1arr,*az2arr,*rz2arr;
  if(!strcmp(compare_type,"analytic")) {
    char str[1024];
    char* rtn;
    int stat;
    FILE *ampz1=fopen("./tests/benchmark/codecomp_step2_outputs/cclamparranalytic1nz512nb.txt","r");
    ASSERT_NOT_NULL(ampz1);
    FILE *ampz2=fopen("./tests/benchmark/codecomp_step2_outputs/cclamparranalytic2nz512nb.txt","r");
    ASSERT_NOT_NULL(ampz2);
    //Create arrays for N(z)
    double zmean_1=1.0,sigz_1=0.15;
    double zmean_2=1.5,sigz_2=0.15;
    nz=512;
    zarr_1=malloc(nz*sizeof(double));
    pzarr_1=malloc(nz*sizeof(double));
    zarr_2=malloc(nz*sizeof(double));
    pzarr_2=malloc(nz*sizeof(double));
    bzarr=malloc(nz*sizeof(double));
    az1arr=malloc(nz*sizeof(double));
    rz1arr=malloc(nz*sizeof(double));
    az2arr=malloc(nz*sizeof(double));
    rz2arr=malloc(nz*sizeof(double));
    for(int ii=0;ii<nz;ii++) {
      double zia1,zia2,aia1,aia2;
      stat = fscanf(ampz1,"%lf %lf",&zia1,&aia1);
      stat = fscanf(ampz2,"%lf %lf",&zia2,&aia2);
      az1arr[ii]=aia1;
      rz1arr[ii]=1.;
      az2arr[ii]=aia2;
      rz2arr[ii]=1.;
      double z1=zmean_1-5*sigz_1+10*sigz_1*(ii+0.5)/nz;
      double z2=zmean_2-5*sigz_2+10*sigz_2*(ii+0.5)/nz;
      double pz1=exp(-0.5*((z1-zmean_1)*(z1-zmean_1)/(sigz_1*sigz_1)));
      double pz2=exp(-0.5*((z2-zmean_2)*(z2-zmean_2)/(sigz_2*sigz_2)));
      zarr_1[ii]=z1;
      zarr_2[ii]=z2;
      pzarr_1[ii]=pz1;
      pzarr_2[ii]=pz2;
      bzarr[ii]=1.;
    }
  }
  else {
    char str[1024];
    char* rtn;
    int stat;
    FILE *fnz1=fopen("./tests/benchmark/codecomp_step2_outputs/bin1_histo.txt","r");
    ASSERT_NOT_NULL(fnz1);
    FILE *fnz2=fopen("./tests/benchmark/codecomp_step2_outputs/bin2_histo.txt","r");
    ASSERT_NOT_NULL(fnz2);
    FILE *ampz1=fopen("./tests/benchmark/codecomp_step2_outputs/cclamparrhisto1nznb.txt","r");
    ASSERT_NOT_NULL(ampz1);
    FILE *ampz2=fopen("./tests/benchmark/codecomp_step2_outputs/cclamparrhisto2nznb.txt","r");
    ASSERT_NOT_NULL(ampz2);
    nz=linecount(fnz1)-1; rewind(fnz1);
    zarr_1=malloc(nz*sizeof(double));
    pzarr_1=malloc(nz*sizeof(double));
    zarr_2=malloc(nz*sizeof(double));
    pzarr_2=malloc(nz*sizeof(double));
    bzarr=malloc(nz*sizeof(double));
    az1arr=malloc(nz*sizeof(double));
    rz1arr=malloc(nz*sizeof(double));
    az2arr=malloc(nz*sizeof(double));
    rz2arr=malloc(nz*sizeof(double));
    rtn = fgets(str,1024,fnz1);
    rtn = fgets(str,1024,fnz2);
    for(int ii=0;ii<nz;ii++) {
      double z1,z2,nz1,nz2,zia1,zia2,aia1,aia2;
      stat = fscanf(fnz1,"%lf %lf",&z1,&nz1);
      stat = fscanf(fnz2,"%lf %lf",&z2,&nz2);
      stat = fscanf(ampz1,"%lf %lf",&zia1,&aia1);
      stat = fscanf(ampz2,"%lf %lf",&zia2,&aia2);
      zarr_1[ii]=z1; zarr_2[ii]=z2;
      pzarr_1[ii]=nz1; pzarr_2[ii]=nz2;
      bzarr[ii]=1.;
      az1arr[ii]=aia1;
      rz1arr[ii]=1.;
      az2arr[ii]=aia2;
      rz2arr[ii]=1.;
    }
  }

  char fname[256];
  FILE *fi_dd_11,*fi_dd_12,*fi_dd_22;
  FILE *fi_dl_12,*fi_dl_11,*fi_dl_22,*fi_dl_21;
  FILE *fi_di_12,*fi_di_11,*fi_di_22,*fi_di_21;
  FILE *fi_dc_1,*fi_dc_2;
  FILE *fi_ll_11,*fi_ll_12,*fi_ll_22;
  FILE *fi_ii_11,*fi_ii_12,*fi_ii_22;
  FILE *fi_li_11,*fi_li_12,*fi_li_22;
  FILE *fi_lc_1,*fi_lc_2;
  FILE *fi_cc;
  CCL_ClTracer *tr_nc_1=ccl_cl_tracer_number_counts_simple(cosmo,nz,zarr_1,pzarr_1,nz,zarr_1,bzarr,&status);
  ASSERT_NOT_NULL(tr_nc_1);
  CCL_ClTracer *tr_nc_2=ccl_cl_tracer_number_counts_simple(cosmo,nz,zarr_2,pzarr_2,nz,zarr_2,bzarr,&status);
  ASSERT_NOT_NULL(tr_nc_2);
  CCL_ClTracer *tr_wl_1=ccl_cl_tracer_lensing_simple(cosmo,nz,zarr_1,pzarr_1,&status);
  ASSERT_NOT_NULL(tr_wl_1);
  CCL_ClTracer *tr_wl_2=ccl_cl_tracer_lensing_simple(cosmo,nz,zarr_2,pzarr_2,&status);
  ASSERT_NOT_NULL(tr_wl_2);
  CCL_ClTracer *tr_wli_1=ccl_cl_tracer_lensing(cosmo,1,nz,zarr_1,pzarr_1,nz,zarr_1,az1arr,nz,zarr_1,rz1arr,&status);
  ASSERT_NOT_NULL(tr_wli_1);
  CCL_ClTracer *tr_wli_2=ccl_cl_tracer_lensing(cosmo,1,nz,zarr_2,pzarr_2,nz,zarr_2,az2arr,nz,zarr_2,rz2arr,&status);
  ASSERT_NOT_NULL(tr_wli_2);
  CCL_ClTracer *tr_cl=ccl_cl_tracer_cmblens(cosmo,zlss,&status);
  ASSERT_NOT_NULL(tr_cl);

  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_dd.txt",compare_type);
  fi_dd_11=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dd_11);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_dd.txt",compare_type);
  fi_dd_12=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dd_12);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_dd.txt",compare_type);
  fi_dd_22=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dd_22);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_dc.txt",compare_type);
  fi_dc_1=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dc_1);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_dc.txt",compare_type);
  fi_dc_2=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dc_2);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_ll.txt",compare_type);
  fi_ll_11=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ll_11);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_ll.txt",compare_type);
  fi_ll_12=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ll_12);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_ll.txt",compare_type);
  fi_ll_22=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ll_22);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_ii.txt",compare_type);
  fi_ii_11=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ii_11);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_ii.txt",compare_type);
  fi_ii_12=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ii_12);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_ii.txt",compare_type);
  fi_ii_22=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ii_22);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_li.txt",compare_type);
  fi_li_11=fopen(fname,"r"); ASSERT_NOT_NULL(fi_li_11);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_li.txt",compare_type);
  fi_li_12=fopen(fname,"r"); ASSERT_NOT_NULL(fi_li_12);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_li.txt",compare_type);
  fi_li_22=fopen(fname,"r"); ASSERT_NOT_NULL(fi_li_22);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_dl.txt",compare_type);
  fi_dl_11=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dl_11);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_dl.txt",compare_type);
  fi_dl_12=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dl_12);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b1%s_log_cl_dl.txt",compare_type);
  fi_dl_21=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dl_21);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_dl.txt",compare_type);
  fi_dl_22=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dl_22);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_di.txt",compare_type);
  fi_di_11=fopen(fname,"r"); ASSERT_NOT_NULL(fi_di_11);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_di.txt",compare_type);
  fi_di_12=fopen(fname,"r"); ASSERT_NOT_NULL(fi_di_12);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b1%s_log_cl_di.txt",compare_type);
  fi_di_21=fopen(fname,"r"); ASSERT_NOT_NULL(fi_di_21);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_di.txt",compare_type);
  fi_di_22=fopen(fname,"r"); ASSERT_NOT_NULL(fi_di_22);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_lc.txt",compare_type);
  fi_lc_1=fopen(fname,"r"); ASSERT_NOT_NULL(fi_lc_1);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_lc.txt",compare_type);
  fi_lc_2=fopen(fname,"r"); ASSERT_NOT_NULL(fi_lc_2);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_log_cl_cc.txt");
  fi_cc=fopen(fname,"r"); ASSERT_NOT_NULL(fi_cc);
  
  int *ells=malloc(NELLS*sizeof(int));
  double *cls_dd_11_b=malloc(NELLS*sizeof(double));
  double *cls_dd_12_b=malloc(NELLS*sizeof(double));
  double *cls_dd_22_b=malloc(NELLS*sizeof(double));
  double *cls_dl_11_b=malloc(NELLS*sizeof(double));
  double *cls_dl_12_b=malloc(NELLS*sizeof(double));
  double *cls_dl_21_b=malloc(NELLS*sizeof(double));
  double *cls_dl_22_b=malloc(NELLS*sizeof(double));
  double *cls_di_11_b=malloc(NELLS*sizeof(double));
  double *cls_di_12_b=malloc(NELLS*sizeof(double));
  double *cls_di_21_b=malloc(NELLS*sizeof(double));
  double *cls_di_22_b=malloc(NELLS*sizeof(double));
  double *cls_dc_1_b=malloc(NELLS*sizeof(double));
  double *cls_dc_2_b=malloc(NELLS*sizeof(double));
  double *cls_ll_11_b=malloc(NELLS*sizeof(double));
  double *cls_ll_12_b=malloc(NELLS*sizeof(double));
  double *cls_ll_22_b=malloc(NELLS*sizeof(double));
  double *cls_li_11_b=malloc(NELLS*sizeof(double));
  double *cls_li_12_b=malloc(NELLS*sizeof(double));
  double *cls_li_22_b=malloc(NELLS*sizeof(double));
  double *cls_ii_11_b=malloc(NELLS*sizeof(double));
  double *cls_ii_12_b=malloc(NELLS*sizeof(double));
  double *cls_ii_22_b=malloc(NELLS*sizeof(double));
  double *cls_lc_1_b=malloc(NELLS*sizeof(double));
  double *cls_lc_2_b=malloc(NELLS*sizeof(double));
  double *cls_cc_b=malloc(NELLS*sizeof(double));
  double *cls_dd_11_h=malloc(NELLS*sizeof(double));
  double *cls_dd_12_h=malloc(NELLS*sizeof(double));
  double *cls_dd_22_h=malloc(NELLS*sizeof(double));
  double *cls_dl_11_h=malloc(NELLS*sizeof(double));
  double *cls_dl_12_h=malloc(NELLS*sizeof(double));
  double *cls_dl_21_h=malloc(NELLS*sizeof(double));
  double *cls_dl_22_h=malloc(NELLS*sizeof(double));
  double *cls_dltot_11_h=malloc(NELLS*sizeof(double));
  double *cls_dltot_12_h=malloc(NELLS*sizeof(double));
  double *cls_dltot_21_h=malloc(NELLS*sizeof(double));
  double *cls_dltot_22_h=malloc(NELLS*sizeof(double));
  double *cls_dc_1_h=malloc(NELLS*sizeof(double));
  double *cls_dc_2_h=malloc(NELLS*sizeof(double));
  double *cls_ll_11_h=malloc(NELLS*sizeof(double));
  double *cls_ll_12_h=malloc(NELLS*sizeof(double));
  double *cls_ll_22_h=malloc(NELLS*sizeof(double));
  double *cls_lltot_11_h=malloc(NELLS*sizeof(double));
  double *cls_lltot_12_h=malloc(NELLS*sizeof(double));
  double *cls_lltot_22_h=malloc(NELLS*sizeof(double));
  double *cls_lli_11_h=malloc(NELLS*sizeof(double));
  double *cls_lli_12_h=malloc(NELLS*sizeof(double));
  double *cls_lli_21_h=malloc(NELLS*sizeof(double));
  double *cls_lli_22_h=malloc(NELLS*sizeof(double));
  double *cls_lc_1_h=malloc(NELLS*sizeof(double));
  double *cls_lc_2_h=malloc(NELLS*sizeof(double));
  double *cls_cc_h=malloc(NELLS*sizeof(double));

  for(int ii=0;ii<NELLS;ii++) {
    int l,stat;
    stat=fscanf(fi_dd_11,"%d %lf",&l,&(cls_dd_11_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_dd_12,"%d %lf",&l,&(cls_dd_12_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_dd_22,"%d %lf",&l,&(cls_dd_22_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_dl_11,"%d %lf",&l,&(cls_dl_11_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_dl_12,"%d %lf",&l,&(cls_dl_12_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_dl_22,"%d %lf",&l,&(cls_dl_22_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_dl_21,"%d %lf",&l,&(cls_dl_21_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_di_11,"%d %lf",&l,&(cls_di_11_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_di_12,"%d %lf",&l,&(cls_di_12_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_di_22,"%d %lf",&l,&(cls_di_22_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_di_21,"%d %lf",&l,&(cls_di_21_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_dc_1,"%d %lf",&l,&(cls_dc_1_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_dc_2,"%d %lf",&l,&(cls_dc_2_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_ll_11,"%d %lf",&l,&(cls_ll_11_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_ll_12,"%d %lf",&l,&(cls_ll_12_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_ll_22,"%d %lf",&l,&(cls_ll_22_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_li_11,"%d %lf",&l,&(cls_li_11_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_li_12,"%d %lf",&l,&(cls_li_12_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_li_22,"%d %lf",&l,&(cls_li_22_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_ii_11,"%d %lf",&l,&(cls_ii_11_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_ii_12,"%d %lf",&l,&(cls_ii_12_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_ii_22,"%d %lf",&l,&(cls_ii_22_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_lc_1,"%d %lf",&l,&(cls_lc_1_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_lc_2,"%d %lf",&l,&(cls_lc_2_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    stat=fscanf(fi_cc,"%d %lf",&l,&(cls_cc_b[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading benchmark file");
      exit(1);
    }
    ells[ii]=l;
  }

  fclose(fi_dd_11);
  fclose(fi_dd_12);
  fclose(fi_dd_22);
  fclose(fi_dl_12);
  fclose(fi_dl_11);
  fclose(fi_dl_21);
  fclose(fi_dl_22);
  fclose(fi_di_12);
  fclose(fi_di_11);
  fclose(fi_di_21);
  fclose(fi_di_22);
  fclose(fi_dc_1);
  fclose(fi_dc_2);
  fclose(fi_ll_11);
  fclose(fi_ll_12);
  fclose(fi_ll_22);
  fclose(fi_li_11);
  fclose(fi_li_12);
  fclose(fi_li_22);
  fclose(fi_ii_11);
  fclose(fi_ii_12);
  fclose(fi_ii_22);
  fclose(fi_lc_1);
  fclose(fi_lc_2);
  fclose(fi_cc);

  double l_logstep = 1.05;
  double l_linstep = 5.;
  CCL_ClWorkspace *w=ccl_cl_workspace_new_limber(NELLS,l_logstep,l_linstep,&status);

  ccl_angular_cls(cosmo,w,tr_nc_1,tr_nc_1,NELLS,ells,cls_dd_11_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_1,tr_nc_2,NELLS,ells,cls_dd_12_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_2,tr_nc_2,NELLS,ells,cls_dd_22_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_1,tr_wl_1,NELLS,ells,cls_dl_11_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_1,tr_wl_2,NELLS,ells,cls_dl_12_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_2,tr_wl_2,NELLS,ells,cls_dl_22_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_2,tr_wl_1,NELLS,ells,cls_dl_21_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_1,tr_wli_1,NELLS,ells,cls_dltot_11_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_1,tr_wli_2,NELLS,ells,cls_dltot_12_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_2,tr_wli_1,NELLS,ells,cls_dltot_21_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_2,tr_wli_2,NELLS,ells,cls_dltot_22_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wli_1,tr_wli_1,NELLS,ells,cls_lltot_11_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wli_1,tr_wli_2,NELLS,ells,cls_lltot_12_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wli_2,tr_wli_2,NELLS,ells,cls_lltot_22_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wl_1,tr_wli_1,NELLS,ells,cls_lli_11_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wl_1,tr_wli_2,NELLS,ells,cls_lli_12_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wl_2,tr_wli_1,NELLS,ells,cls_lli_21_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wl_2,tr_wli_2,NELLS,ells,cls_lli_22_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_1,tr_cl,NELLS,ells,cls_dc_1_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_2,tr_cl,NELLS,ells,cls_dc_2_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wl_1,tr_wl_1,NELLS,ells,cls_ll_11_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wl_1,tr_wl_2,NELLS,ells,cls_ll_12_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wl_2,tr_wl_2,NELLS,ells,cls_ll_22_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wl_1,tr_cl,NELLS,ells,cls_lc_1_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wl_2,tr_cl,NELLS,ells,cls_lc_2_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_cl,tr_cl,NELLS,ells,cls_cc_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  
  double fraction_failed=0;
  for(int ii=2;ii<w->n_ls-1;ii++) {
    int l=w->l_arr[ii];
    double ell_correct,ell_correct2;
    double el_dd_11,el_dd_12,el_dd_22;
    double el_dl_11,el_dl_12,el_dl_21,el_dl_22;
    double el_dltot_11,el_dltot_12,el_dltot_21,el_dltot_22;
    double el_dc_1,el_dc_2;
    double el_ll_11,el_ll_12,el_ll_22;
    double el_li_11,el_li_12,el_li_22;
    double el_ii_11,el_ii_12,el_ii_22;
    double el_lltot_11,el_lltot_12,el_lltot_22;
    double el_lc_1,el_lc_2;
    double el_cc;
    double cl_dd_11,cl_dd_12,cl_dd_22;
    double cl_dl_12,cl_dl_11,cl_dl_21,cl_dl_22;
    double cl_di_12,cl_di_11,cl_di_21,cl_di_22;
    double cl_dltot_12,cl_dltot_11,cl_dltot_21,cl_dltot_22;
    double cl_lltot_11,cl_lltot_12,cl_lltot_22;
    double cl_dc_1,cl_dc_2;
    double cl_ll_11,cl_ll_12,cl_ll_22;
    double cl_li_11,cl_li_12,cl_li_22;
    double cl_ii_11,cl_ii_12,cl_ii_22;
    double cl_lc_1,cl_lc_2;
    double cl_cc;
    double cl_dd_11_h,cl_dd_12_h,cl_dd_22_h;
    double cl_dl_12_h,cl_dl_11_h,cl_dl_21_h,cl_dl_22_h;
    double cl_di_12_h,cl_di_11_h,cl_di_21_h,cl_di_22_h;
    double cl_dltot_12_h,cl_dltot_11_h,cl_dltot_21_h,cl_dltot_22_h;
    double cl_dc_1_h,cl_dc_2_h;
    double cl_ll_11_h,cl_ll_12_h,cl_ll_22_h;
    double cl_li_11_h,cl_li_12_h,cl_li_22_h;
    double cl_ii_11_h,cl_ii_12_h,cl_ii_22_h;
    double cl_lltot_11_h,cl_lltot_12_h,cl_lltot_22_h;
    double cl_lc_1_h,cl_lc_2_h;
    double cl_cc_h;
    if(l<=1)
      ell_correct=1;
    else{
      ell_correct=l*(l+1.)/sqrt((l+2.)*(l+1.)*l*(l-1.));
      ell_correct2=(l+0.5)*(l+0.5)/sqrt((l+2.)*(l+1.)*l*(l-1.));
    }
    cl_dd_11  =cls_dd_11_b[l];
    cl_dd_12  =cls_dd_12_b[l];
    cl_dd_22  =cls_dd_22_b[l];
    cl_dl_12  =cls_dl_12_b[l];
    cl_dl_11  =cls_dl_11_b[l];
    cl_dl_21  =cls_dl_21_b[l];
    cl_dl_22  =cls_dl_22_b[l];
    cl_di_12  =cls_di_12_b[l];
    cl_di_11  =cls_di_11_b[l];
    cl_di_21  =cls_di_21_b[l];
    cl_di_22  =cls_di_22_b[l];
    cl_dltot_11=cl_dl_11+cl_di_11;
    cl_dltot_12=cl_dl_12+cl_di_12;
    cl_dltot_21=cl_dl_21+cl_di_21;
    cl_dltot_22=cl_dl_22+cl_di_22;
    cl_dc_1  =cls_dc_1_b[l];
    cl_dc_2  =cls_dc_2_b[l];
    cl_ll_11  =cls_ll_11_b[l];
    cl_ll_12  =cls_ll_12_b[l];
    cl_ll_22  =cls_ll_22_b[l];
    cl_li_11  =cls_li_11_b[l];
    cl_li_12  =cls_li_12_b[l];
    cl_li_22  =cls_li_22_b[l];
    cl_ii_11  =cls_ii_11_b[l];
    cl_ii_12  =cls_ii_12_b[l];
    cl_ii_22  =cls_ii_22_b[l];
    cl_lltot_11=cl_ll_11+cl_li_11+cl_ii_11;
    cl_lltot_12=cl_ll_12+cl_li_12+cl_ii_12;
    cl_lltot_22=cl_ll_22+cl_li_22+cl_ii_22;
    cl_lc_1  =cls_lc_1_b[l];
    cl_lc_2  =cls_lc_2_b[l];
    cl_cc    =cls_cc_b[l];
    el_dd_11=ELS_TOLERANCE*sqrt((cl_dd_11*cl_dd_11+cl_dd_11*cl_dd_11)/(2*l+1.));
    el_dd_12=ELS_TOLERANCE*sqrt((cl_dd_11*cl_dd_22+cl_dd_12*cl_dd_12)/(2*l+1.));
    el_dd_22=ELS_TOLERANCE*sqrt((cl_dd_22*cl_dd_22+cl_dd_22*cl_dd_22)/(2*l+1.));
    el_dl_11=ELS_TOLERANCE*sqrt((cl_dd_11*cl_ll_11+cl_dl_11*cl_dl_11)/(2*l+1.));
    el_dl_12=ELS_TOLERANCE*sqrt((cl_dd_11*cl_ll_22+cl_dl_12*cl_dl_12)/(2*l+1.));
    el_dl_21=ELS_TOLERANCE*sqrt((cl_dd_22*cl_ll_11+cl_dl_21*cl_dl_21)/(2*l+1.));
    el_dl_22=ELS_TOLERANCE*sqrt((cl_dd_22*cl_ll_22+cl_dl_22*cl_dl_22)/(2*l+1.));
    el_dltot_11=ELS_TOLERANCE*sqrt((cl_dd_11*cl_lltot_11+cl_dltot_11*cl_dltot_11)/(2*l+1.));
    el_dltot_12=ELS_TOLERANCE*sqrt((cl_dd_11*cl_lltot_22+cl_dltot_12*cl_dltot_12)/(2*l+1.));
    el_dltot_21=ELS_TOLERANCE*sqrt((cl_dd_22*cl_lltot_11+cl_dltot_21*cl_dltot_21)/(2*l+1.));
    el_dltot_22=ELS_TOLERANCE*sqrt((cl_dd_22*cl_lltot_22+cl_dltot_22*cl_dltot_22)/(2*l+1.));
    el_dc_1=ELS_TOLERANCE*sqrt((cl_dd_11*cl_cc+cl_dc_1*cl_dc_1)/(2*l+1.));
    el_dc_2=ELS_TOLERANCE*sqrt((cl_dd_22*cl_cc+cl_dc_2*cl_dc_2)/(2*l+1.));
    el_ll_11=ELS_TOLERANCE*sqrt((cl_ll_11*cl_ll_11+cl_ll_11*cl_ll_11)/(2*l+1.));
    el_ll_12=ELS_TOLERANCE*sqrt((cl_ll_11*cl_ll_22+cl_ll_12*cl_ll_12)/(2*l+1.));
    el_ll_22=ELS_TOLERANCE*sqrt((cl_ll_22*cl_ll_22+cl_ll_22*cl_ll_22)/(2*l+1.));
    el_li_11=ELS_TOLERANCE*sqrt((cl_li_11*cl_li_11+cl_li_11*cl_li_11)/(2*l+1.));
    el_li_12=ELS_TOLERANCE*sqrt((cl_li_11*cl_li_22+cl_li_12*cl_li_12)/(2*l+1.));
    el_li_22=ELS_TOLERANCE*sqrt((cl_li_22*cl_li_22+cl_li_22*cl_li_22)/(2*l+1.));
    el_ii_11=ELS_TOLERANCE*sqrt((cl_ii_11*cl_ii_11+cl_ii_11*cl_ii_11)/(2*l+1.));
    el_ii_12=ELS_TOLERANCE*sqrt((cl_ii_11*cl_ii_22+cl_ii_12*cl_ii_12)/(2*l+1.));
    el_ii_22=ELS_TOLERANCE*sqrt((cl_ii_22*cl_ii_22+cl_ii_22*cl_ii_22)/(2*l+1.));
    el_lltot_11=ELS_TOLERANCE*sqrt((cl_lltot_11*cl_lltot_11+cl_lltot_11*cl_lltot_11)/(2*l+1.));
    el_lltot_12=ELS_TOLERANCE*sqrt((cl_lltot_11*cl_lltot_22+cl_lltot_12*cl_lltot_12)/(2*l+1.));
    el_lltot_22=ELS_TOLERANCE*sqrt((cl_lltot_22*cl_lltot_22+cl_lltot_22*cl_lltot_22)/(2*l+1.));
    el_lc_1=ELS_TOLERANCE*sqrt((cl_ll_11*cl_cc+cl_lc_1*cl_lc_1)/(2*l+1.));
    el_lc_2=ELS_TOLERANCE*sqrt((cl_ll_22*cl_cc+cl_lc_2*cl_lc_2)/(2*l+1.));
    el_cc=ELS_TOLERANCE*sqrt((cl_cc*cl_cc+cl_cc*cl_cc)/(2*l+1.));
    cl_dd_11_h=cls_dd_11_h[l];
    cl_dd_12_h=cls_dd_12_h[l];
    cl_dd_22_h=cls_dd_22_h[l];
    cl_dl_11_h=cls_dl_11_h[l]*ell_correct2;
    cl_dl_12_h=cls_dl_12_h[l]*ell_correct2;
    cl_dl_21_h=cls_dl_21_h[l]*ell_correct2;
    cl_dl_22_h=cls_dl_22_h[l]*ell_correct2;
    cl_dltot_11_h=cls_dltot_11_h[l]*ell_correct2;
    cl_dltot_12_h=cls_dltot_12_h[l]*ell_correct2;
    cl_dltot_21_h=cls_dltot_21_h[l]*ell_correct2;
    cl_dltot_22_h=cls_dltot_22_h[l]*ell_correct2;
    cl_di_11_h=cl_dltot_11_h-cl_dl_11_h;
    cl_di_12_h=cl_dltot_12_h-cl_dl_12_h;
    cl_di_21_h=cl_dltot_21_h-cl_dl_21_h;
    cl_di_22_h=cl_dltot_22_h-cl_dl_22_h;
    cl_dc_1_h=cls_dc_1_h[l];
    cl_dc_2_h=cls_dc_2_h[l];
    cl_ll_11_h=cls_ll_11_h[l]*ell_correct2*ell_correct2;
    cl_ll_12_h=cls_ll_12_h[l]*ell_correct2*ell_correct2;
    cl_ll_22_h=cls_ll_22_h[l]*ell_correct2*ell_correct2;
    cl_li_11_h=2*(cls_lli_11_h[l]-cls_ll_11_h[l])*ell_correct2*ell_correct2;
    cl_li_12_h=(cls_lli_12_h[l]+cls_lli_21_h[l]-2*cls_ll_12_h[l])*ell_correct2*ell_correct2;
    cl_li_22_h=2*(cls_lli_22_h[l]-cls_ll_22_h[l])*ell_correct2*ell_correct2;
    cl_ii_11_h=(cls_lltot_11_h[l]+cls_ll_11_h[l]-2*cls_lli_11_h[l])*ell_correct2*ell_correct2;
    cl_ii_12_h=(cls_lltot_12_h[l]+cls_ll_12_h[l]-cls_lli_12_h[l]-cls_lli_21_h[l])*ell_correct2*ell_correct2;
    cl_ii_22_h=(cls_lltot_22_h[l]+cls_ll_22_h[l]-2*cls_lli_22_h[l])*ell_correct2*ell_correct2;
    cl_lltot_11_h=cls_lltot_11_h[l]*ell_correct2*ell_correct2;
    cl_lltot_12_h=cls_lltot_12_h[l]*ell_correct2*ell_correct2;
    cl_lltot_22_h=cls_lltot_22_h[l]*ell_correct2*ell_correct2;
    cl_lc_1_h=cls_lc_1_h[l]*ell_correct;
    cl_lc_2_h=cls_lc_2_h[l]*ell_correct;
    cl_cc_h=cls_cc_h[l];

    ASSERT_TRUE(fabs(cl_dd_11_h-cl_dd_11)<el_dd_11);
    ASSERT_TRUE(fabs(cl_dd_12_h-cl_dd_12)<el_dd_12);
    ASSERT_TRUE(fabs(cl_dd_22_h-cl_dd_22)<el_dd_22);
    ASSERT_TRUE(fabs(cl_dc_1_h-cl_dc_1)<el_dc_1);
    ASSERT_TRUE(fabs(cl_dc_2_h-cl_dc_2)<el_dc_2);

    ASSERT_TRUE(fabs(cl_dl_11_h-cl_dl_11)<el_dl_11);
    ASSERT_TRUE(fabs(cl_dl_12_h-cl_dl_12)<el_dl_12);
    ASSERT_TRUE(fabs(cl_dl_21_h-cl_dl_21)<el_dl_21);
    ASSERT_TRUE(fabs(cl_dl_22_h-cl_dl_22)<el_dl_22);
    //  comparing galaxy-intrinsic wrt full GGL error, CCL needs to be further improved to avoid spikes
    ASSERT_TRUE(fabs(cl_di_11_h-cl_di_11)<el_dltot_11); 
    ASSERT_TRUE(fabs(cl_di_12_h-cl_di_12)<el_dltot_12);
    ASSERT_TRUE(fabs(cl_di_21_h-cl_di_21)<el_dltot_21);
    ASSERT_TRUE(fabs(cl_di_22_h-cl_di_22)<el_dltot_22);

    ASSERT_TRUE(fabs(cl_ll_11_h-cl_ll_11)<el_ll_11);
    ASSERT_TRUE(fabs(cl_ll_12_h-cl_ll_12)<el_ll_12);
    ASSERT_TRUE(fabs(cl_ll_22_h-cl_ll_22)<el_ll_22);
    //  comparing shear-intrinsic and intrinsic-intrinsic wrt full cosmic shear error, CCL needs to be further improved to avoid spikes
    ASSERT_TRUE(fabs(cl_li_11_h-cl_li_11)<el_lltot_11);
    ASSERT_TRUE(fabs(cl_li_12_h-cl_li_12)<el_lltot_12);
    ASSERT_TRUE(fabs(cl_li_22_h-cl_li_22)<el_lltot_22);
    ASSERT_TRUE(fabs(cl_ii_11_h-cl_ii_11)<el_lltot_11);
    ASSERT_TRUE(fabs(cl_ii_12_h-cl_ii_12)<el_lltot_12);
    ASSERT_TRUE(fabs(cl_ii_22_h-cl_ii_22)<el_lltot_22);

    ASSERT_TRUE(fabs(cl_cc_h-cl_cc)<el_cc);
    ASSERT_TRUE(fabs(cl_lc_1_h-cl_lc_1)<el_lc_1);
    ASSERT_TRUE(fabs(cl_lc_2_h-cl_lc_2)<el_lc_2);
  }
  ccl_cl_workspace_free(w);
  if(!strcmp(compare_type,"histo")) {
    ccl_gsl->INTEGRATION_EPSREL=epsrel_save;
    ccl_gsl->INTEGRATION_LIMBER_EPSREL=epsrel_save;
    ccl_set_debug_policy(CCL_DEBUG_MODE_WARNING);
  }
    
  free(ells);
  free(cls_dd_11_b); free(cls_dd_12_b); free(cls_dd_22_b); 
  free(cls_dl_12_b);free(cls_dl_21_b);free(cls_dl_11_b);free(cls_dl_22_b);
  free(cls_di_12_b);free(cls_di_21_b);free(cls_di_11_b);free(cls_di_22_b);
  free(cls_dc_1_b); free(cls_dc_2_b);
  free(cls_ll_11_b); free(cls_ll_12_b); free(cls_ll_22_b); 
  free(cls_li_11_b); free(cls_li_12_b); free(cls_li_22_b); 
  free(cls_ii_11_b); free(cls_ii_12_b); free(cls_ii_22_b); 
  free(cls_lc_1_b); free(cls_lc_2_b);
  free(cls_cc_b);

  free(cls_dd_11_h); free(cls_dd_12_h); free(cls_dd_22_h); 
  free(cls_dl_12_h);free(cls_dl_11_h);free(cls_dl_21_h);free(cls_dl_22_h);
  free(cls_dltot_12_h);free(cls_dltot_11_h);free(cls_dltot_21_h);free(cls_dltot_22_h);
  free(cls_dc_1_h); free(cls_dc_2_h);
  free(cls_ll_11_h); free(cls_ll_12_h); free(cls_ll_22_h); 
  free(cls_lltot_11_h); free(cls_lltot_12_h); free(cls_lltot_22_h); 
  free(cls_lli_11_h); free(cls_lli_12_h); free(cls_lli_21_h); free(cls_lli_22_h); 
  free(cls_lc_1_h); free(cls_lc_2_h);
  free(cls_cc_h);

  free(zarr_1);
  free(zarr_2);
  free(pzarr_1);
  free(pzarr_2);
  free(az1arr);
  free(az2arr);
  free(rz1arr);
  free(rz2arr);
  free(bzarr);
  ccl_cl_tracer_free(tr_nc_1);
  ccl_cl_tracer_free(tr_nc_2);
  ccl_cl_tracer_free(tr_wl_1);
  ccl_cl_tracer_free(tr_wl_2);
  ccl_cl_tracer_free(tr_wli_1);
  ccl_cl_tracer_free(tr_wli_2);
  ccl_cl_tracer_free(tr_cl);
  ccl_cosmology_free(cosmo);
}

CTEST2(cls,analytic) {
  compare_cls("analytic",data);
}

CTEST2(cls,histo) {
  compare_cls("histo",data);
}
