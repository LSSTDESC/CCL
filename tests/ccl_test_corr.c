#include "ccl.h"
#include "ctest.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define CORR_TOLERANCE 1E-3
#define CORR_FRACTION 1E-3

CTEST_DATA(corrs) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double sigma_8;
};

CTEST_SETUP(corrs) {
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

static void compare_corr(char *compare_type,struct corrs_data * data)
{
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  ccl_parameters params = ccl_parameters_create_flat_lcdm(data->Omega_c,data->Omega_b,data->h,data->A_s,data->n_s);
  params.sigma_8=data->sigma_8;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
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
  FILE *fi_dd_11,*fi_dd_12,*fi_dd_22;
  FILE *fi_ll_11_pp,*fi_ll_12_pp,*fi_ll_22_pp;
  FILE *fi_ll_11_mm,*fi_ll_12_mm,*fi_ll_22_mm;
  CCL_ClTracer *tr_nc_1=ccl_cl_tracer_new(cosmo,CL_TRACER_NC,nz,zarr_1,pzarr_1,nz,zarr_1,bzarr);
  ASSERT_NOT_NULL(tr_nc_1);
  CCL_ClTracer *tr_nc_2=ccl_cl_tracer_new(cosmo,CL_TRACER_NC,nz,zarr_2,pzarr_2,nz,zarr_2,bzarr);
  ASSERT_NOT_NULL(tr_nc_2);
  CCL_ClTracer *tr_wl_1=ccl_cl_tracer_new(cosmo,CL_TRACER_WL,nz,zarr_1,pzarr_1,nz,NULL,NULL );
  ASSERT_NOT_NULL(tr_wl_1);
  CCL_ClTracer *tr_wl_2=ccl_cl_tracer_new(cosmo,CL_TRACER_WL,nz,zarr_2,pzarr_2,nz,NULL,NULL );
  ASSERT_NOT_NULL(tr_wl_2);

  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_wt_dd.txt",compare_type);
  fi_dd_11=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dd_11);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_wt_dd.txt",compare_type);
  fi_dd_12=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dd_12);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_wt_dd.txt",compare_type);
  fi_dd_22=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dd_22);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_wt_ll_pp.txt",compare_type);
  fi_ll_11_pp=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ll_11_pp);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_wt_ll_pp.txt",compare_type);
  fi_ll_12_pp=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ll_12_pp);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_wt_ll_pp.txt",compare_type);
  fi_ll_22_pp=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ll_22_pp);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_wt_ll_mm.txt",compare_type);
  fi_ll_11_mm=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ll_11_mm);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_wt_ll_mm.txt",compare_type);
  fi_ll_12_mm=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ll_12_mm);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_wt_ll_mm.txt",compare_type);
  fi_ll_22_mm=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ll_22_mm);

  double fraction_failed=0;
  int nofl=15;
  double wt_dd_11[nofl],wt_dd_12[nofl],wt_dd_22[nofl];
  double wt_ll_11_mm[nofl],wt_ll_12_mm[nofl],wt_ll_22_mm[nofl];
  double wt_ll_11_pp[nofl],wt_ll_12_pp[nofl],wt_ll_22_pp[nofl];
  double *wt_dd_11_h,*wt_dd_12_h,*wt_dd_22_h;
  double *wt_ll_11_h_mm,*wt_ll_12_h_mm,*wt_ll_22_h_mm;
  double *wt_ll_11_h_pp,*wt_ll_12_h_pp,*wt_ll_22_h_pp;
  double theta_in[nofl],*theta_arr;


  for(int ii=0;ii<nofl;ii++) {
    fscanf(fi_dd_11,"%lf %lf",&theta_in[ii],&wt_dd_11[ii]);
    fscanf(fi_dd_12,"%*lf %lf",&wt_dd_12[ii]);
    fscanf(fi_dd_22,"%*lf %lf",&wt_dd_22[ii]);
    fscanf(fi_ll_11_pp,"%*lf %lf",&wt_ll_11_pp[ii]);
    fscanf(fi_ll_12_pp,"%*lf %lf",&wt_ll_12_pp[ii]);
    fscanf(fi_ll_22_pp,"%*lf %lf",&wt_ll_22_pp[ii]);
    fscanf(fi_ll_11_mm,"%*lf %lf",&wt_ll_11_mm[ii]);
    fscanf(fi_ll_12_mm,"%*lf %lf",&wt_ll_12_mm[ii]);
    fscanf(fi_ll_22_mm,"%*lf %lf",&wt_ll_22_mm[ii]);
    printf("%i %e %e %e \n",ii,theta_in[ii],wt_ll_12_mm[ii],wt_ll_11_pp[ii]);
  }
  
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_nc_1,tr_nc_1,0,&wt_dd_11_h);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_nc_1,tr_nc_2,0,&wt_dd_12_h);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_nc_2,tr_nc_2,0,&wt_dd_22_h);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_wl_1,tr_wl_1,0,&wt_ll_11_h_pp);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_wl_1,tr_wl_2,0,&wt_ll_12_h_pp);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_wl_2,tr_wl_2,0,&wt_ll_22_h_pp);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_wl_1,tr_wl_1,4,&wt_ll_11_h_mm);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_wl_1,tr_wl_2,4,&wt_ll_12_h_mm);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_wl_2,tr_wl_2,4,&wt_ll_22_h_mm);

  FILE *output2 = fopen("cc_test_corr_out_fftlog.dat", "w");
  for (int i=0;i<NL;i++)
    {
      theta_arr[i]*=sqrt(4*M_PI);
      theta_arr[i]=theta_arr[i]*180/M_PI;
      fprintf(output2,"%lf %lf %lf \n",theta_arr[i],wt_dd_11_h[i],wt_ll_11_h_pp[i]);
    }
  fclose(output2);
  /*
  double *theta_arr_inv=(double *)malloc(sizeof(double)*NL);
  double *wt_dd_11_h_inv=(double *)malloc(sizeof(double)*NL);
  for (int i=0;i<NL;i++)
    {
      theta_arr_inv[NL-1-i]=theta_arr[i];
      wt_dd_11_h_inv[NL-1-i]=wt_dd_11_h[i];
    }
  */
  //Spline
  gsl_spline * spl_wt_dd_11_h = gsl_spline_alloc(K_SPLINE_TYPE,NL);
  int status = gsl_spline_init(spl_wt_dd_11_h, theta_arr, wt_dd_11_h, NL);
  gsl_spline * spl_wt_dd_12_h = gsl_spline_alloc(K_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_dd_12_h, theta_arr, wt_dd_12_h, NL);
  gsl_spline * spl_wt_dd_22_h = gsl_spline_alloc(K_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_dd_22_h, theta_arr, wt_dd_22_h, NL);
  gsl_spline * spl_wt_ll_11_h_pp = gsl_spline_alloc(K_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_dd_11_h, theta_arr, wt_ll_11_h_pp, NL);
  gsl_spline * spl_wt_ll_12_h_pp = gsl_spline_alloc(K_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_dd_12_h, theta_arr, wt_ll_12_h_pp, NL);
  gsl_spline * spl_wt_ll_22_h_pp = gsl_spline_alloc(K_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_dd_22_h, theta_arr, wt_ll_22_h_pp, NL);
  gsl_spline * spl_wt_ll_11_h_mm = gsl_spline_alloc(K_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_dd_11_h, theta_arr, wt_ll_11_h_mm, NL);
  gsl_spline * spl_wt_ll_12_h_mm = gsl_spline_alloc(K_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_dd_12_h, theta_arr, wt_ll_12_h_mm, NL);
  gsl_spline * spl_wt_ll_22_h_mm = gsl_spline_alloc(K_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_dd_22_h, theta_arr, wt_ll_22_h_mm, NL);

  double tmp;
  FILE *output = fopen("cc_test_corr_out.dat", "w");
  printf("theta min max  %f  %f \n",theta_arr[0],theta_arr[NL-1]);
  for(int ii=0;ii<nofl;ii++) {
    tmp=gsl_spline_eval(spl_wt_dd_11_h, theta_in[ii], NULL);
    if(fabs(tmp/wt_dd_11[ii]-1)>CORR_TOLERANCE)
      fraction_failed++;
    fprintf(output,"%lf %lf %lf",theta_in[ii],tmp,wt_dd_11[ii]);

    tmp=gsl_spline_eval(spl_wt_dd_12_h, theta_in[ii], NULL);
    if(fabs(tmp/wt_dd_12[ii]-1)>CORR_TOLERANCE)
      fraction_failed++;
    fprintf(output," %lf %lf",tmp,wt_dd_12[ii]);

    printf("%i %lf %lf %lf\n",ii,theta_in[ii],tmp,wt_dd_12[ii]);

    tmp=gsl_spline_eval(spl_wt_dd_22_h, theta_in[ii], NULL);
    if(fabs(tmp/wt_dd_22[ii]-1)>CORR_TOLERANCE)
      fraction_failed++;
    fprintf(output," %lf %lf",tmp,wt_dd_22[ii]);

    gsl_spline_eval_e(spl_wt_ll_11_h_pp, theta_in[ii], NULL,&tmp);
    if(fabs(tmp/wt_ll_11_pp[ii]-1)>CORR_TOLERANCE)
      fraction_failed++;
    fprintf(output," %lf %lf",tmp,wt_ll_11_h_pp[ii]);
    
    gsl_spline_eval_e(spl_wt_ll_12_h_pp, theta_in[ii], NULL,&tmp);
    if(fabs(tmp/wt_ll_12_pp[ii]-1)>CORR_TOLERANCE)
      fraction_failed++;
    fprintf(output," %lf %lf",tmp,wt_ll_12_h_pp[ii]);

    gsl_spline_eval_e(spl_wt_ll_22_h_pp, theta_in[ii], NULL,&tmp);
    if(fabs(tmp/wt_ll_22_pp[ii]-1)>CORR_TOLERANCE)
      fraction_failed++;
    fprintf(output," %lf %lf",tmp,wt_ll_22_h_pp[ii]);

    gsl_spline_eval_e(spl_wt_ll_11_h_mm, theta_in[ii], NULL,&tmp);
    if(fabs(tmp/wt_ll_11_mm[ii]-1)>CORR_TOLERANCE)
      fraction_failed++;
    fprintf(output," %lf %lf",tmp,wt_ll_11_h_mm[ii]);

    gsl_spline_eval_e(spl_wt_ll_12_h_mm, theta_in[ii], NULL,&tmp);
    if(fabs(tmp/wt_ll_12_mm[ii]-1)>CORR_TOLERANCE)
      fraction_failed++;
    fprintf(output," %lf %lf",tmp,wt_ll_12_h_mm[ii]);

    gsl_spline_eval_e(spl_wt_ll_22_h_mm, theta_in[ii], NULL,&tmp);
    if(fabs(tmp/wt_ll_22_mm[ii]-1)>CORR_TOLERANCE)
      fraction_failed++;
    fprintf(output," %lf %lf \n",tmp,wt_ll_22_h_mm[ii]);
  }
  fclose(output);
  gsl_spline_free(spl_wt_dd_11_h);
  gsl_spline_free(spl_wt_dd_12_h);
  gsl_spline_free(spl_wt_dd_22_h);
  gsl_spline_free(spl_wt_ll_11_h_pp);
  gsl_spline_free(spl_wt_ll_12_h_pp);
  gsl_spline_free(spl_wt_ll_22_h_pp);
  gsl_spline_free(spl_wt_ll_11_h_mm);
  gsl_spline_free(spl_wt_ll_12_h_mm);
  gsl_spline_free(spl_wt_ll_22_h_mm);

  fclose(fi_dd_11);
  fclose(fi_dd_12);
  fclose(fi_dd_22);
  fclose(fi_ll_11_pp);
  fclose(fi_ll_12_pp);
  fclose(fi_ll_22_pp);
  fclose(fi_ll_11_mm);
  fclose(fi_ll_12_mm);
  fclose(fi_ll_22_mm);

  fraction_failed/=9*nofl;
  printf("%lf %%\n",fraction_failed*100);
  ASSERT_TRUE((fraction_failed<CORR_FRACTION));

  free(zarr_1);
  free(zarr_2);
  free(pzarr_1);
  free(pzarr_2);
  free(bzarr);
  ccl_cosmology_free(cosmo);
}

CTEST2(corrs,analytic) {
  compare_corr("analytic",data);
}

CTEST2(corrs,histo) {
  compare_corr("histo",data);
}
