#include "ccl.h"
#include "ctest.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define CORR_FRACTION_PASS 1E-3
#define CORR_ERROR_FRACTION 0.5
#define ELL_MAX_CL 3000

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

static void compare_corr(char *compare_type,int algorithm,struct corrs_data * data)
{
  int ii,status=0;

  /* Set up the CCL configuration for comparing to benchmarks
   * The benchmarks are of two types: those which use analytic
   * redshift distributions, and those which use histograms for
   * them. We will compare CCL correlations to the benchmarks
   * using estimated covariances from CosmoLike.
   */
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  ccl_parameters params = ccl_parameters_create_flat_lcdm(data->Omega_c,data->Omega_b,data->h,
							  data->sigma_8,data->n_s,&status);
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);

  /*Create arrays for redshift distributions in the case of analytic benchmarks*/
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
    for(ii=0;ii<nz;ii++) {
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
  else { /*Load arrays for redshift distributions in the case of histograms*/
    char *rtn;
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
    rtn=fgets(str,1024,fnz1);
    rtn=fgets(str,1024,fnz2);
    for(ii=0;ii<nz;ii++) {
      int stat;
      double z1,z2,nz1,nz2;
      stat=fscanf(fnz1,"%lf %lf",&z1,&nz1);
      stat=fscanf(fnz2,"%lf %lf",&z2,&nz2);
      zarr_1[ii]=z1; zarr_2[ii]=z2;
      pzarr_1[ii]=nz1; pzarr_2[ii]=nz2;
      bzarr[ii]=1.;
    }
  }

  /*For the same configuration as the benchmarks, we will produce CCL
   correlation functions starting by computing C_ells here: */
  char fname[256];
  FILE *fi_dd_11,*fi_dd_12,*fi_dd_22;
  FILE *fi_ll_11_pp,*fi_ll_12_pp,*fi_ll_22_pp;
  FILE *fi_ll_11_mm,*fi_ll_12_mm,*fi_ll_22_mm;
  int has_rsd=0,has_magnification=0, has_intrinsic_alignment=0;
  int status2=0;
  CCL_ClTracer *tr_nc_1=ccl_cl_tracer_number_counts_simple_new(cosmo,nz,zarr_1,pzarr_1,
							       nz,zarr_1,bzarr,&status2);
  ASSERT_NOT_NULL(tr_nc_1);
  CCL_ClTracer *tr_nc_2=ccl_cl_tracer_number_counts_simple_new(cosmo,nz,zarr_2,pzarr_2,
							       nz,zarr_2,bzarr,&status2);
  ASSERT_NOT_NULL(tr_nc_2);
  CCL_ClTracer *tr_wl_1=ccl_cl_tracer_lensing_simple_new(cosmo,nz,zarr_1,pzarr_1,&status2);
  ASSERT_NOT_NULL(tr_wl_1);
  CCL_ClTracer *tr_wl_2=ccl_cl_tracer_lensing_simple_new(cosmo,nz,zarr_2,pzarr_2,&status2);
  ASSERT_NOT_NULL(tr_wl_2);

  /* Read in the benchmark correlations*/
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
  double taper_cl_limits[4]={1,2,10000,15000};//{0,0,0,0};
  double wt_dd_11[nofl],wt_dd_12[nofl],wt_dd_22[nofl];
  double wt_ll_11_mm[nofl],wt_ll_12_mm[nofl],wt_ll_22_mm[nofl];
  double wt_ll_11_pp[nofl],wt_ll_12_pp[nofl],wt_ll_22_pp[nofl];
  double *wt_dd_11_h,*wt_dd_12_h,*wt_dd_22_h;
  double *wt_ll_11_h_mm,*wt_ll_12_h_mm,*wt_ll_22_h_mm;
  double *wt_ll_11_h_pp,*wt_ll_12_h_pp,*wt_ll_22_h_pp;
  double theta_in[nofl];

  for(ii=0;ii<nofl;ii++) {
    int stat;
    double dum;
    stat=fscanf(fi_dd_11,"%lf %lf",&theta_in[ii],&wt_dd_11[ii]);
    stat=fscanf(fi_dd_12,"%lf %lf",&dum,&wt_dd_12[ii]);
    stat=fscanf(fi_dd_22,"%lf %lf",&dum,&wt_dd_22[ii]);
    stat=fscanf(fi_ll_11_pp,"%lf %lf",&dum,&wt_ll_11_pp[ii]);
    stat=fscanf(fi_ll_12_pp,"%lf %lf",&dum,&wt_ll_12_pp[ii]);
    stat=fscanf(fi_ll_22_pp,"%lf %lf",&dum,&wt_ll_22_pp[ii]);
    stat=fscanf(fi_ll_11_mm,"%lf %lf",&dum,&wt_ll_11_mm[ii]);
    stat=fscanf(fi_ll_12_mm,"%lf %lf",&dum,&wt_ll_12_mm[ii]);
    stat=fscanf(fi_ll_22_mm,"%lf %lf",&dum,&wt_ll_22_mm[ii]);
  }
  fclose(fi_dd_11); fclose(fi_dd_12); fclose(fi_dd_22);
  fclose(fi_ll_11_pp); fclose(fi_ll_12_pp); fclose(fi_ll_22_pp);
  fclose(fi_ll_11_mm); fclose(fi_ll_12_mm); fclose(fi_ll_22_mm);

  /*Compute the correlation with CCL*/
  int il;
  double *clarr=malloc(ELL_MAX_CL*sizeof(double));
  double *larr=malloc(ELL_MAX_CL*sizeof(double));
  for(il=0;il<ELL_MAX_CL;il++)
    larr[il]=il;

  wt_dd_11_h=malloc(nofl*sizeof(double));
  for(il=0;il<ELL_MAX_CL;il++)
    clarr[il]=ccl_angular_cl(cosmo,il,tr_nc_1,tr_nc_1,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dd_11_h,CCL_CORR_GG,
		  0,taper_cl_limits,algorithm,&status);
  wt_dd_12_h=malloc(nofl*sizeof(double));
  for(il=0;il<ELL_MAX_CL;il++)
    clarr[il]=ccl_angular_cl(cosmo,il,tr_nc_1,tr_nc_2,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dd_12_h,CCL_CORR_GG,
		  0,taper_cl_limits,algorithm,&status);
  wt_dd_22_h=malloc(nofl*sizeof(double));
  for(il=0;il<ELL_MAX_CL;il++)
    clarr[il]=ccl_angular_cl(cosmo,il,tr_nc_2,tr_nc_2,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dd_22_h,CCL_CORR_GG,
		  0,taper_cl_limits,algorithm,&status);

  wt_ll_11_h_mm=malloc(nofl*sizeof(double));
  wt_ll_11_h_pp=malloc(nofl*sizeof(double));
  for(il=0;il<ELL_MAX_CL;il++)
    clarr[il]=ccl_angular_cl(cosmo,il,tr_wl_1,tr_wl_1,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ll_11_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ll_11_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);

  wt_ll_12_h_mm=malloc(nofl*sizeof(double));
  wt_ll_12_h_pp=malloc(nofl*sizeof(double));
  for(il=0;il<ELL_MAX_CL;il++)
    clarr[il]=ccl_angular_cl(cosmo,il,tr_wl_1,tr_wl_2,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ll_12_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ll_12_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);

  wt_ll_22_h_mm=malloc(nofl*sizeof(double));
  wt_ll_22_h_pp=malloc(nofl*sizeof(double));
  for(il=0;il<ELL_MAX_CL;il++)
    clarr[il]=ccl_angular_cl(cosmo,il,tr_wl_2,tr_wl_2,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ll_22_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ll_22_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);
  free(clarr);
  free(larr);

  /* With the CCL correlation already computed, we proceed to the
  * comparison. Here, we read in the benchmark covariances from CosmoLike, which
  * allow us to set our tolerance.
  */
  int nsig=15;
  double sigwt_dd_11[15], sigwt_dd_22[15];
  double sigwt_ll_11_mm[15], sigwt_ll_22_mm[15];
  double sigwt_ll_11_pp[15], sigwt_ll_22_pp[15];
  double sig_theta_in[15];

  char bs[1024];
  FILE *fi_dd_sig=fopen("tests/benchmark/cov_corr/sigma_clustering_Nbin5","r");
  FILE *fi_mm_sig=fopen("tests/benchmark/cov_corr/sigma_xi-_Nbin5","r");
  FILE *fi_pp_sig=fopen("tests/benchmark/cov_corr/sigma_xi+_Nbin5","r");
  if(fgets(bs,sizeof(bs),fi_dd_sig)==NULL) {
    fprintf(stderr,"Error reading file\n");
    exit(1);
  }
  if(fgets(bs,sizeof(bs),fi_mm_sig)==NULL) {
    fprintf(stderr,"Error reading file\n");
    exit(1);
  }
  if(fgets(bs,sizeof(bs),fi_pp_sig)==NULL) {
    fprintf(stderr,"Error reading file\n");
    exit(1);
  }
  for(int ii=0;ii<nsig;ii++) {
    int stat;
    double dum;
    stat=fscanf(fi_dd_sig,"%le %le %le %le",&sig_theta_in[ii],&sigwt_dd_11[ii],&sigwt_dd_22[ii],&dum);
    stat=fscanf(fi_pp_sig,"%le %le %le %le",&sig_theta_in[ii],&sigwt_ll_11_pp[ii],&sigwt_ll_22_pp[ii],&dum);
    stat=fscanf(fi_mm_sig,"%le %le %le %le",&sig_theta_in[ii],&sigwt_ll_11_mm[ii],&sigwt_ll_22_mm[ii],&dum);
    sig_theta_in[ii]=sig_theta_in[ii]/60.; //convert to deg
  }
  fclose(fi_dd_sig);
  fclose(fi_mm_sig);
  fclose(fi_pp_sig);

  /* Spline the covariances */
  gsl_spline *spl_sigwt_dd_11   =gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_dd_11   ,sig_theta_in,sigwt_dd_11   ,nsig);
  gsl_spline *spl_sigwt_dd_22   =gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_dd_22   ,sig_theta_in,sigwt_dd_22   ,nsig);
  gsl_spline *spl_sigwt_ll_11_pp=gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_ll_11_pp,sig_theta_in,sigwt_ll_11_pp,nsig);
  gsl_spline *spl_sigwt_ll_22_pp=gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_ll_22_pp,sig_theta_in,sigwt_ll_22_pp,nsig);
  gsl_spline *spl_sigwt_ll_11_mm=gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_ll_11_mm,sig_theta_in,sigwt_ll_11_mm,nsig);
  gsl_spline *spl_sigwt_ll_22_mm=gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_ll_22_mm,sig_theta_in,sigwt_ll_22_mm,nsig);

  /* Proceed to the comparison between benchmarks and CCL.
   * If DEBUG flag is set, then produce an output file.
   */
  
#ifdef _DEBUG
  FILE *output = fopen("cc_test_corr_out.dat", "w");
#endif //_DEBUG
  int npoints=0;
  for(ii=0;ii<nofl;ii++) {
    double tol;
    
    if((theta_in[ii]<sig_theta_in[0]) ||(theta_in[ii]>sig_theta_in[nsig-1]))
      continue;
    else
      npoints++;

    /*First time the tolerance is set. The tolerance is equal to the 
     *expected error bar times CORR_ERR_FRACTION=0.5 (default) */
    tol=gsl_spline_eval(spl_sigwt_dd_11,theta_in[ii],NULL);
    if(fabs(wt_dd_11_h[ii]-wt_dd_11[ii])>tol*CORR_ERROR_FRACTION)
      fraction_failed++;
#ifdef _DEBUG
    fprintf(output,"%.10e %.10e %.10e %.10e",theta_in[ii],wt_dd_11_h[ii],wt_dd_11[ii],tol);
#endif //_DEBUG
    //    tol=gsl_spline_eval(spl_sigwt_dd_12,theta_in[ii],NULL);
    //    if(fabs(wt_dd_12_h[ii]-wt_dd_12[ii])>tol*CORR_ERROR_FRACTION)
    //      fraction_failed++;
#ifdef _DEBUG
    //    fprintf(output," %.10e %.10e",wt_dd_12_h[ii],wt_dd_12[ii]);
#endif //_DEBUG
    tol=gsl_spline_eval(spl_sigwt_dd_22,theta_in[ii],NULL);
    if(fabs(wt_dd_22_h[ii]-wt_dd_22[ii])>tol*CORR_ERROR_FRACTION)
      fraction_failed++;
#ifdef _DEBUG
    fprintf(output," %.10e %.10e %.10e",wt_dd_22_h[ii],wt_dd_22[ii],tol);
#endif //_DEBUG

    tol=gsl_spline_eval(spl_sigwt_ll_11_pp,theta_in[ii],NULL);
    if(fabs(wt_ll_11_h_pp[ii]-wt_ll_11_pp[ii])>tol*CORR_ERROR_FRACTION)
      fraction_failed++;
#ifdef _DEBUG
    fprintf(output," %.10e %.10e %.10e",wt_ll_11_h_pp[ii],wt_ll_11_pp[ii],tol);
#endif //_DEBUG
    //    tol=gsl_spline_eval(spl_sigwt_ll_12_pp,theta_in[ii],NULL);
    //    if(fabs(wt_ll_12_h_pp[ii]-wt_ll_12_pp[ii])>tol*CORR_ERROR_FRACTION)
    //      fraction_failed++;
#ifdef _DEBUG
    //    fprintf(output," %.10e %.10e",wt_ll_12_h_pp[ii],wt_ll_12_pp[ii]);
#endif //_DEBUG
    tol=gsl_spline_eval(spl_sigwt_ll_22_pp,theta_in[ii],NULL);
    if(fabs(wt_ll_22_h_pp[ii]-wt_ll_22_pp[ii])>tol*CORR_ERROR_FRACTION)
      fraction_failed++;
#ifdef _DEBUG
    fprintf(output," %.10e %.10e %.10e",wt_ll_22_h_pp[ii],wt_ll_22_pp[ii],tol);
#endif //_DEBUG

    tol=gsl_spline_eval(spl_sigwt_ll_11_mm,theta_in[ii],NULL);
    if(fabs(wt_ll_11_h_mm[ii]-wt_ll_11_mm[ii])>tol*CORR_ERROR_FRACTION)
      fraction_failed++;
#ifdef _DEBUG
    fprintf(output," %.10e %.10e %.10e",wt_ll_11_h_mm[ii],wt_ll_11_mm[ii],tol);
#endif //_DEBUG
    //    tol=gsl_spline_eval(spl_sigwt_ll_12_mm,theta_in[ii],NULL);
    //    if(fabs(wt_ll_12_h_mm[ii]-wt_ll_12_mm[ii])>tol*CORR_ERROR_FRACTION)
    //      fraction_failed++;
#ifdef _DEBUG
    //    fprintf(output," %.10e %.10e",wt_ll_12_h_mm[ii],wt_ll_12_mm[ii]);
#endif //_DEBUG
    tol=gsl_spline_eval(spl_sigwt_ll_22_mm,theta_in[ii],NULL);
    if(fabs(wt_ll_22_h_mm[ii]-wt_ll_22_mm[ii])>tol*CORR_ERROR_FRACTION)
      fraction_failed++;
#ifdef _DEBUG
    fprintf(output," %.10e %.10e %.10e",wt_ll_22_h_mm[ii],wt_ll_22_mm[ii],tol);
    fprintf(output,"\n");
#endif //_DEBUG
  }
#ifdef _DEBUG
  fclose(output);
#endif //_DEBUG
  
  //Determine the fraction of points that failed the test
  fraction_failed/=6*npoints;
  printf("%lf %% ",fraction_failed*100);
  //Check is this fraction is larger than we allow
  ASSERT_TRUE((fraction_failed<CORR_FRACTION_PASS));
  
  //Free splines, cosmology and arrays
  gsl_spline_free(spl_sigwt_dd_11);
  gsl_spline_free(spl_sigwt_dd_22);
  gsl_spline_free(spl_sigwt_ll_11_pp);
  gsl_spline_free(spl_sigwt_ll_22_pp);
  gsl_spline_free(spl_sigwt_ll_11_mm);
  gsl_spline_free(spl_sigwt_ll_22_mm);
  free(wt_dd_11_h); free(wt_dd_12_h); free(wt_dd_22_h);
  free(wt_ll_11_h_pp); free(wt_ll_12_h_pp); free(wt_ll_22_h_pp);
  free(wt_ll_11_h_mm); free(wt_ll_12_h_mm); free(wt_ll_22_h_mm);
  free(zarr_1); free(zarr_2);
  free(pzarr_1); free(pzarr_2);
  free(bzarr);
  ccl_cosmology_free(cosmo);
}

CTEST2(corrs,analytic_fftlog) {
  compare_corr("analytic",CCL_CORR_FFTLOG,data);
}

CTEST2(corrs,analytic_bessel) {
  compare_corr("analytic",CCL_CORR_BESSEL,data);
}
