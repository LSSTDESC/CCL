#include "ccl.h"
#include "ctest.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define CORR_FRACTION 1E-3

CTEST_DATA(corrs) {
  double Omega_c;
  double Omega_b;
  double h;
  double n_s;
  double sigma_8;
};

CTEST_SETUP(corrs) {
  data->Omega_c = 0.30;
  data->Omega_b = 0.00;
  data->h = 0.7;
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
  int status=0;
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  ccl_parameters params = ccl_parameters_create_flat_lcdm(data->Omega_c,data->Omega_b,data->h,data->sigma_8,data->n_s,&status);
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);

  int nz;
  double *zarr_1,*pzarr_1,*zarr_2,*pzarr_2,*bzarr;
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

  char fname[256];
  FILE *fi_dd_11,*fi_dd_12,*fi_dd_22;
  FILE *fi_ll_11_pp,*fi_ll_12_pp,*fi_ll_22_pp;
  FILE *fi_ll_11_mm,*fi_ll_12_mm,*fi_ll_22_mm;
  int has_rsd=0,has_magnification=0, has_intrinsic_alignment=0;
  int status2=0;
  CCL_ClTracer *tr_nc_1=ccl_cl_tracer_number_counts_simple_new(cosmo,nz,zarr_1,pzarr_1,nz,zarr_1,bzarr,&status2);
  ASSERT_NOT_NULL(tr_nc_1);
  CCL_ClTracer *tr_nc_2=ccl_cl_tracer_number_counts_simple_new(cosmo,nz,zarr_2,pzarr_2,nz,zarr_2,bzarr,&status2);
  ASSERT_NOT_NULL(tr_nc_2);
  CCL_ClTracer *tr_wl_1=ccl_cl_tracer_lensing_simple_new(cosmo,nz,zarr_1,pzarr_1,&status2);
  ASSERT_NOT_NULL(tr_wl_1);
  CCL_ClTracer *tr_wl_2=ccl_cl_tracer_lensing_simple_new(cosmo,nz,zarr_2,pzarr_2,&status2 );
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
  bool taper_cl=false;
  double taper_cl_limits[4]={1,2,10000,15000};//{0,0,0,0};
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
  }
  time_t start_time,end_time;
  double time_sec=0;

  //Now obtain the correlation from CCL
  taper_cl=false;
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_nc_1,tr_nc_1,0,taper_cl,taper_cl_limits,
		  &wt_dd_11_h,CCL_CORR_FFTLOG);

  time(&end_time);
  time_sec=difftime(end_time,start_time);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_nc_1,tr_nc_2,0,taper_cl,taper_cl_limits,
		  &wt_dd_12_h,CCL_CORR_FFTLOG);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_nc_2,tr_nc_2,0,taper_cl,taper_cl_limits,
		  &wt_dd_22_h,CCL_CORR_FFTLOG);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_wl_1,tr_wl_1,0,taper_cl,taper_cl_limits,
		  &wt_ll_11_h_pp,CCL_CORR_FFTLOG);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_wl_1,tr_wl_2,0,taper_cl,taper_cl_limits,
		  &wt_ll_12_h_pp,CCL_CORR_FFTLOG);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_wl_2,tr_wl_2,0,taper_cl,taper_cl_limits,
		  &wt_ll_22_h_pp,CCL_CORR_FFTLOG);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_wl_1,tr_wl_1,4,taper_cl,taper_cl_limits,
		  &wt_ll_11_h_mm,CCL_CORR_FFTLOG);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_wl_1,tr_wl_2,4,taper_cl,taper_cl_limits,
		  &wt_ll_12_h_mm,CCL_CORR_FFTLOG);
  ccl_tracer_corr(cosmo,NL,&theta_arr,tr_wl_2,tr_wl_2,4,taper_cl,taper_cl_limits,
		  &wt_ll_22_h_mm,CCL_CORR_FFTLOG);

  //Re-scale theta from radians to degrees
  for (int i=0;i<NL;i++){
    theta_arr[i]=theta_arr[i]*180/M_PI;
  }
  
  /*Print to a file
  FILE *output2 = fopen("cc_test_corr_out_fftlog.dat", "w");
  for (int ii=0;ii<NL;ii++){
    fprintf(output2,"%.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e \n",
	    theta_arr[ii],wt_dd_11_h[ii],wt_dd_12_h[ii],wt_dd_22_h[ii],
	    wt_ll_11_h_pp[ii],wt_ll_12_h_pp[ii],wt_ll_22_h_pp[ii],wt_ll_11_h_mm[ii],
	    wt_ll_12_h_mm[ii],wt_ll_22_h_mm[ii]);
  }
  fclose(output2);
  printf("CCL correlation output done.\n");*/

  //Spline
  gsl_spline * spl_wt_dd_11_h = gsl_spline_alloc(L_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_dd_11_h, theta_arr, wt_dd_11_h, NL);
  gsl_spline * spl_wt_dd_12_h = gsl_spline_alloc(L_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_dd_12_h, theta_arr, wt_dd_12_h, NL);
  gsl_spline * spl_wt_dd_22_h = gsl_spline_alloc(L_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_dd_22_h, theta_arr, wt_dd_22_h, NL);
  gsl_spline * spl_wt_ll_11_h_pp = gsl_spline_alloc(L_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_ll_11_h_pp, theta_arr, wt_ll_11_h_pp, NL);
  gsl_spline * spl_wt_ll_12_h_pp = gsl_spline_alloc(L_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_ll_12_h_pp, theta_arr, wt_ll_12_h_pp, NL);
  gsl_spline * spl_wt_ll_22_h_pp = gsl_spline_alloc(L_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_ll_22_h_pp, theta_arr, wt_ll_22_h_pp, NL);
  gsl_spline * spl_wt_ll_11_h_mm = gsl_spline_alloc(L_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_ll_11_h_mm, theta_arr, wt_ll_11_h_mm, NL);
  gsl_spline * spl_wt_ll_12_h_mm = gsl_spline_alloc(L_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_ll_12_h_mm, theta_arr, wt_ll_12_h_mm, NL);
  gsl_spline * spl_wt_ll_22_h_mm = gsl_spline_alloc(L_SPLINE_TYPE,NL);
  status = gsl_spline_init(spl_wt_ll_22_h_mm, theta_arr, wt_ll_22_h_mm, NL);
  //printf("Splines for correlation done.\n");

  //Adjusting the range of thetas in case theta_in is wider than theta_arr
  int ii,istart=0,iend=nofl;
  if(theta_in[0]<theta_arr[0] || theta_in[nofl-1]>theta_arr[NL-1]){
    //printf("theta_in range: [%e,%e]\n",theta_in[0],theta_in[nofl-1]);
    //printf("theta_arr range: [%e,%e]\n",theta_arr[0],theta_arr[NL-1]);
    //printf("This code would crash because gsl will attempt to extrapolate.\n");
    //printf("Temporary solution: reducing the range for comparison to avoid extralpolation.\n");
    ii=0;
    while(theta_in[ii]<theta_arr[NL-1]){ii++;}
    iend=ii-1;
    ii=nofl-1;
    while(theta_in[ii]>theta_arr[0]){ii=ii-1;}
    istart=ii+1;
    //printf("Corrected theta_in range: [%e,%e]\n",theta_in[istart],theta_in[iend]);
    //printf("This correction avoids crash, but does not\n compare correlation in the full range of angles needed.\n");
  }
  
 
  //Now we are going to define the tolerance based on the 
  //absolute error, provided by EK. We will read and spline the tolerance.
  int nsig=15;
  double *sigwt_dd_11,*sigwt_dd_22;
  double *sigwt_ll_11_mm,*sigwt_ll_22_mm;
  double *sigwt_ll_11_pp,*sigwt_ll_22_pp;
  sigwt_dd_11=malloc(nsig*sizeof(double));
  sigwt_dd_22=malloc(nsig*sizeof(double));
  sigwt_ll_11_pp=malloc(nsig*sizeof(double));
  sigwt_ll_22_pp=malloc(nsig*sizeof(double));
  sigwt_ll_11_mm=malloc(nsig*sizeof(double));
  sigwt_ll_22_mm=malloc(nsig*sizeof(double));
  double sig_theta_in[nsig];
  FILE *fi_dd_sig,*fi_pp_sig,*fi_mm_sig;
  fi_dd_sig=fopen("tests/benchmark/cov_corr/sigma_clustering_Nbin5","r");
  fi_mm_sig=fopen("tests/benchmark/cov_corr/sigma_xi-_Nbin5","r");
  fi_pp_sig=fopen("tests/benchmark/cov_corr/sigma_xi+_Nbin5","r");
  fscanf(fi_dd_sig,"%*s %*s %*s %*s %*s\n");
  fscanf(fi_mm_sig,"%*s %*s %*s %*s %*s\n");
  fscanf(fi_pp_sig,"%*s %*s %*s %*s %*s\n");
  for(int ii=0;ii<nsig;ii++) {
    fscanf(fi_dd_sig,"%le %le %le %*le",&sig_theta_in[ii],&sigwt_dd_11[ii],&sigwt_dd_22[ii]);
    fscanf(fi_pp_sig,"%le %le %le %*le",&sig_theta_in[ii],&sigwt_ll_11_pp[ii],&sigwt_ll_22_pp[ii]);
    fscanf(fi_mm_sig,"%le %le %le %*le",&sig_theta_in[ii],&sigwt_ll_11_mm[ii],&sigwt_ll_22_mm[ii]);
    sig_theta_in[ii]=sig_theta_in[ii]/60.; //convert to deg
  }
  fclose(fi_dd_sig);
  fclose(fi_mm_sig);
  fclose(fi_pp_sig);
  //printf("Covariance of the correlation read.\n");

  //Spline
  gsl_spline * spl_sigwt_dd_11 = gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  status = gsl_spline_init(spl_sigwt_dd_11, sig_theta_in, sigwt_dd_11,nsig);
  gsl_spline * spl_sigwt_dd_22 = gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  status = gsl_spline_init(spl_sigwt_dd_22, sig_theta_in, sigwt_dd_22,nsig);
  gsl_spline * spl_sigwt_pp_11 = gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  status = gsl_spline_init(spl_sigwt_pp_11, sig_theta_in, sigwt_ll_11_pp,nsig);
  gsl_spline * spl_sigwt_pp_22 = gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  status = gsl_spline_init(spl_sigwt_pp_22, sig_theta_in, sigwt_ll_22_pp,nsig);
  gsl_spline * spl_sigwt_mm_11 = gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  status = gsl_spline_init(spl_sigwt_mm_11, sig_theta_in, sigwt_ll_11_mm,nsig);
  gsl_spline * spl_sigwt_mm_22 = gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  status = gsl_spline_init(spl_sigwt_mm_22, sig_theta_in, sigwt_ll_22_mm,nsig);
  //printf("Splines of the covariance done.\n");

  //Adjusting theta comparison range if theta_in is wider than sig_theta_in
  int istart2=0,iend2=nsig;
  if((theta_in[istart]<sig_theta_in[0]) || (theta_in[iend-1]>sig_theta_in[nsig-1])){
    //printf("sig_theta_in range: [%e,%e]\n",sig_theta_in[0],sig_theta_in[nsig-1]);
    ///printf("theta_in range: [%e,%e]\n",theta_in[istart],theta_in[iend-1]);
    //printf("This code would crash because gsl will attempt to extrapolate.\n");
    //printf("Temporary solution: reducing the range for comparison to avoid extralpolation.\n");
    ii=0;
    while(theta_in[ii]<sig_theta_in[nsig-1]){ii++;}
    iend2=ii-1;
    ii=nofl-1;
    while(theta_in[ii]>sig_theta_in[0]){ii=ii-1;}
    istart2=ii+1;
    if(istart2>istart) istart=istart2;
    if(iend2<iend) iend=iend2;
    //printf("Corrected theta_in range: [%e,%e]\n",theta_in[istart],theta_in[iend]);
    //printf("This correction avoids crash, but does not\n compare correlation in the full range of angles needed.\n");
  }

  double tmp,tmptol;
  FILE *output = fopen("tests/cc_test_corr_out.dat", "w");
  for(ii=istart;ii<iend;ii++) {
    tmp=gsl_spline_eval(spl_wt_dd_11_h, theta_in[ii], NULL);
    tmptol=gsl_spline_eval(spl_sigwt_dd_11, theta_in[ii], NULL);
    if(fabs(tmp-wt_dd_11[ii])>tmptol)
      fraction_failed++;
    fprintf(output,"%.10e %.10e %.10e",theta_in[ii],fabs(tmp-wt_dd_11[ii]),tmptol);

    tmp=gsl_spline_eval(spl_wt_dd_22_h, theta_in[ii], NULL);
    tmptol=gsl_spline_eval(spl_sigwt_dd_22, theta_in[ii], NULL);
    if(fabs(tmp-wt_dd_22[ii])>tmptol)
      fraction_failed++;
    fprintf(output," %.10e %.10e",fabs(tmp-wt_dd_22[ii]),tmptol);

    gsl_spline_eval_e(spl_wt_ll_11_h_pp, theta_in[ii], NULL,&tmp);
    tmptol=gsl_spline_eval(spl_sigwt_pp_11, theta_in[ii], NULL);
    if(fabs(tmp-wt_ll_11_pp[ii])>tmptol)
      fraction_failed++;
    fprintf(output," %.10e %.10e",fabs(tmp-wt_ll_11_pp[ii]),tmptol);

    gsl_spline_eval_e(spl_wt_ll_22_h_pp, theta_in[ii], NULL,&tmp);
    tmptol=gsl_spline_eval(spl_sigwt_pp_22, theta_in[ii], NULL);
    if(fabs(tmp-wt_ll_22_pp[ii])>tmptol)
      fraction_failed++;
    fprintf(output," %.10e %.10e",fabs(tmp-wt_ll_22_pp[ii]),tmptol);

    gsl_spline_eval_e(spl_wt_ll_11_h_mm, theta_in[ii], NULL,&tmp);
    tmptol=gsl_spline_eval(spl_sigwt_mm_11, theta_in[ii], NULL);
    if(fabs(tmp-wt_ll_11_mm[ii])>tmptol)
      fraction_failed++;
    fprintf(output," %.10e %.10e",fabs(tmp-wt_ll_11_mm[ii]),tmptol);

    gsl_spline_eval_e(spl_wt_ll_22_h_mm, theta_in[ii], NULL,&tmp);
    tmptol=gsl_spline_eval(spl_sigwt_mm_22, theta_in[ii], NULL);
    if(fabs(tmp-wt_ll_22_mm[ii])>tmptol)
      fraction_failed++;
    fprintf(output," %.10e %.10e \n",fabs(tmp-wt_ll_22_mm[ii]),tmptol);
  }
  fclose(output);

  free(sigwt_dd_11);
  free(sigwt_dd_22);
  free(sigwt_ll_11_pp);
  free(sigwt_ll_22_pp);
  free(sigwt_ll_11_mm);
  free(sigwt_ll_22_mm);
  gsl_spline_free(spl_sigwt_dd_11);
  gsl_spline_free(spl_sigwt_dd_22);
  gsl_spline_free(spl_sigwt_mm_11);
  gsl_spline_free(spl_sigwt_pp_11);
  gsl_spline_free(spl_sigwt_mm_22);
  gsl_spline_free(spl_sigwt_pp_22);
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

  fraction_failed/=6*nofl;
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
