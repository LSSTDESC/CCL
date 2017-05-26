#include "ccl.h"
#include "ctest.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define SZ_VAL 0.4 //This will cancel the magnification contribution
#define CLS_TOLERANCE 1E-3
#define CLS_FRACTION 1E-3

CTEST_DATA(cls) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double sigma_8;
};

CTEST_SETUP(cls) {
  data->Omega_c = 0.30;
  data->Omega_b = 0.00;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->sigma_8=0.8;
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
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  config.matter_power_spectrum_method = ccl_linear;
  ccl_parameters params = ccl_parameters_create_flat_lcdm(data->Omega_c,data->Omega_b,data->h,
							  data->A_s,data->n_s);
  params.Omega_g=0;
  params.sigma_8=data->sigma_8;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  int status=0;
  ASSERT_NOT_NULL(cosmo);

  int nz;
  double *zarr_1,*pzarr_1,*zarr_2,*pzarr_2,*bzarr;
  if(!strcmp(compare_type,"analytic")) {
    //Create arrays for N(z)
    double zmean_1=1.0,sigz_1=0.15;
    double zmean_2=1.5,sigz_2=0.15;
    nz=512;
    zarr_1=malloc(nz*sizeof(double));
    pzarr_1=malloc(nz*sizeof(double));
    zarr_2=malloc(nz*sizeof(double));
    pzarr_2=malloc(nz*sizeof(double));
    bzarr=malloc(nz*sizeof(double));
    for(int ii=0;ii<nz;ii++) {
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
    FILE *fnz1=fopen("./tests/benchmark/codecomp_step2_outputs/bin1_histo.txt","r");
    ASSERT_NOT_NULL(fnz1);
    FILE *fnz2=fopen("./tests/benchmark/codecomp_step2_outputs/bin2_histo.txt","r");
    ASSERT_NOT_NULL(fnz2);
    nz=linecount(fnz1)-1; rewind(fnz1);
    zarr_1=malloc(nz*sizeof(double));
    pzarr_1=malloc(nz*sizeof(double));
    zarr_2=malloc(nz*sizeof(double));
    pzarr_2=malloc(nz*sizeof(double));
    bzarr=malloc(nz*sizeof(double));
    fgets(str,1024,fnz1);
    fgets(str,1024,fnz2);
    for(int ii=0;ii<nz;ii++) {
      double z1,z2,nz1,nz2;
      fscanf(fnz1,"%lf %lf",&z1,&nz1);
      fscanf(fnz2,"%lf %lf",&z2,&nz2);
      zarr_1[ii]=z1; zarr_2[ii]=z2;
      pzarr_1[ii]=nz1; pzarr_2[ii]=nz2;
      bzarr[ii]=1.;
    }
  }

  char fname[256];
  FILE *fi_dd_11,*fi_dd_12,*fi_dd_22,*fi_ll_11,*fi_ll_12,*fi_ll_22;
  CCL_ClTracer *tr_nc_1=ccl_cl_tracer_number_counts_simple_new(cosmo,nz,zarr_1,pzarr_1,nz,zarr_1,bzarr,&status);
  ASSERT_NOT_NULL(tr_nc_1);
  CCL_ClTracer *tr_nc_2=ccl_cl_tracer_number_counts_simple_new(cosmo,nz,zarr_2,pzarr_2,nz,zarr_2,bzarr,&status);
  ASSERT_NOT_NULL(tr_nc_2);
  CCL_ClTracer *tr_wl_1=ccl_cl_tracer_lensing_simple_new(cosmo,nz,zarr_1,pzarr_1,&status);
  ASSERT_NOT_NULL(tr_wl_1);
  CCL_ClTracer *tr_wl_2=ccl_cl_tracer_lensing_simple_new(cosmo,nz,zarr_2,pzarr_2,&status);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_dd.txt",compare_type);
  fi_dd_11=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dd_11);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_dd.txt",compare_type);
  fi_dd_12=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dd_12);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_dd.txt",compare_type);
  fi_dd_22=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dd_22);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_ll.txt",compare_type);
  fi_ll_11=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ll_11);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_ll.txt",compare_type);
  fi_ll_12=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ll_12);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_ll.txt",compare_type);
  fi_ll_22=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ll_22);

  int *ells=malloc(3001*sizeof(int));
  double *cls_dd_11_b=malloc(3001*sizeof(double));
  double *cls_dd_12_b=malloc(3001*sizeof(double));
  double *cls_dd_22_b=malloc(3001*sizeof(double));
  double *cls_ll_11_b=malloc(3001*sizeof(double));
  double *cls_ll_12_b=malloc(3001*sizeof(double));
  double *cls_ll_22_b=malloc(3001*sizeof(double));
  double *cls_dd_11_h=malloc(3001*sizeof(double));
  double *cls_dd_12_h=malloc(3001*sizeof(double));
  double *cls_dd_22_h=malloc(3001*sizeof(double));
  double *cls_ll_11_h=malloc(3001*sizeof(double));
  double *cls_ll_12_h=malloc(3001*sizeof(double));
  double *cls_ll_22_h=malloc(3001*sizeof(double));

  for(int ii=0;ii<3001;ii++) {
    int l;
    double cl_dd_11,cl_dd_12,cl_dd_22;
    double cl_ll_11,cl_ll_12,cl_ll_22;
    double cl_dd_11_h,cl_dd_12_h,cl_dd_22_h;
    double cl_ll_11_h,cl_ll_12_h,cl_ll_22_h;
    fscanf(fi_dd_11,"%d %lf",&l,&(cls_dd_11_b[ii]));
    fscanf(fi_dd_12,"%d %lf",&l,&(cls_dd_12_b[ii]));
    fscanf(fi_dd_22,"%d %lf",&l,&(cls_dd_22_b[ii]));
    fscanf(fi_ll_11,"%d %lf",&l,&(cls_ll_11_b[ii]));
    fscanf(fi_ll_12,"%d %lf",&l,&(cls_ll_12_b[ii]));
    fscanf(fi_ll_22,"%d %lf",&l,&(cls_ll_22_b[ii]));
    ells[ii]=l;
  }

  fclose(fi_dd_11);
  fclose(fi_dd_12);
  fclose(fi_dd_22);
  fclose(fi_ll_11);
  fclose(fi_ll_12);
  fclose(fi_ll_22);

  CCL_ClWorkspace *w=ccl_cl_workspace_new(3001,-1,CCL_NONLIMBER_METHOD_NATIVE,1.05,20,3.,0.003,0.05,&status);

  ccl_angular_cls(cosmo,w,tr_nc_1,tr_nc_1,3001,ells,cls_dd_11_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_1,tr_nc_2,3001,ells,cls_dd_12_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_nc_2,tr_nc_2,3001,ells,cls_dd_22_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wl_1,tr_wl_1,3001,ells,cls_ll_11_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wl_1,tr_wl_2,3001,ells,cls_ll_12_h,&status);
  if (status) printf("%s\n",cosmo->status_message);
  ccl_angular_cls(cosmo,w,tr_wl_2,tr_wl_2,3001,ells,cls_ll_22_h,&status);
  if (status) printf("%s\n",cosmo->status_message);

  ccl_cl_workspace_free(w);

  double fraction_failed=0;
  for(int ii=0;ii<3001;ii++) {
    int l=ells[ii];
    double cl_dd_11  =cls_dd_11_b[ii];
    double cl_dd_12  =cls_dd_12_b[ii];
    double cl_dd_22  =cls_dd_22_b[ii];
    double cl_ll_11  =cls_ll_11_b[ii];
    double cl_ll_12  =cls_ll_12_b[ii];
    double cl_ll_22  =cls_ll_22_b[ii];
    double cl_dd_11_h=cls_dd_11_h[ii];
    double cl_dd_12_h=cls_dd_12_h[ii];
    double cl_dd_22_h=cls_dd_22_h[ii];
    double cl_ll_11_h=cls_ll_11_h[ii];
    double cl_ll_12_h=cls_ll_12_h[ii];
    double cl_ll_22_h=cls_ll_22_h[ii];

    if(fabs(cl_dd_11_h/cl_dd_11-1)>CLS_TOLERANCE)
      fraction_failed++;
    if(fabs(cl_dd_12_h/cl_dd_12-1)>CLS_TOLERANCE)
      fraction_failed++;
    if(fabs(cl_dd_22_h/cl_dd_22-1)>CLS_TOLERANCE)
      fraction_failed++;
    if(fabs(cl_ll_11_h/cl_ll_11-1)>CLS_TOLERANCE)
      fraction_failed++;
    if(fabs(cl_ll_12_h/cl_ll_12-1)>CLS_TOLERANCE)
      fraction_failed++;
    if(fabs(cl_ll_22_h/cl_ll_22-1)>CLS_TOLERANCE)
      fraction_failed++;
  }

  free(ells);
  free(cls_dd_11_b); free(cls_dd_12_b); free(cls_dd_22_b); 
  free(cls_ll_11_b); free(cls_ll_12_b); free(cls_ll_22_b); 
  free(cls_dd_11_h); free(cls_dd_12_h); free(cls_dd_22_h); 
  free(cls_ll_11_h); free(cls_ll_12_h); free(cls_ll_22_h); 

  printf("%d, ",(int)fraction_failed);
  fraction_failed/=6*3001;
  printf("%lf %%\n",fraction_failed*100);
  ASSERT_TRUE((fraction_failed<CLS_FRACTION));

  free(zarr_1);
  free(zarr_2);
  free(pzarr_1);
  free(pzarr_2);
  free(bzarr);
  ccl_cosmology_free(cosmo);
}

CTEST2(cls,analytic) {
  compare_cls("analytic",data);
}

CTEST2(cls,histo) {
  compare_cls("histo",data);
}
