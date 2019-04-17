#include "ccl.h"
#include "ctest.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define CORR_ERROR_FRACTION 0.1
#define ELL_MAX_CL 10000
#define L_SPLINE_TYPE gsl_interp_akima
double fftlogfactor; //this is the factor by which FFTLog performs more weakly than the brute-force integration approach (Bessel)

CTEST_DATA(corrs) {
  double Omega_c;
  double Omega_b;
  double h;
  double n_s;
  double sigma8;
};

CTEST_SETUP(corrs) {
  data->Omega_c = 0.30;
  data->Omega_b = 0.00;
  data->h = 0.7;
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

static void get_cls_arr(ccl_cosmology *cosmo,
			CCL_ClTracer *tr1,CCL_ClTracer *tr2,
			double *cls,
			//double *ell_correct,
			int *status)
{
  //Create array of ells
  int nls=0;
  double l=0;
  double *ls=malloc(ELL_MAX_CL*sizeof(double));
  for(int ii=0;l<=ELL_MAX_CL;ii++) {
    ls[ii]=l;
    if(l<100)
      l+=1;
    else if(l<200)
      l+=5;
    else if(l<400)
      l+=10;
    else
      l+=20;
    nls++;
  }

  //Generate corresponding power spectra
  double *cls_interp=malloc(nls*sizeof(double));

  //Loop over ells
  for(int ii=0;ii<nls;ii++)
    cls_interp[ii]=ccl_angular_cl_limber(cosmo,tr1,tr2,NULL,ls[ii],status);

  //Interpolate
  gsl_spline *sp=gsl_spline_alloc(gsl_interp_cspline,nls);
  gsl_spline_init(sp,ls,cls_interp,nls);
  for(int ii=0;ii<ELL_MAX_CL;ii++) {
    int e=gsl_spline_eval_e(sp,(double)ii,NULL,&(cls[ii]));
    if(e) {
      fprintf(stderr,"Interpolation error\n");
      exit(1);
    }
  }
  free(ls);
  free(cls_interp);
  gsl_spline_free(sp);
}

static void compare_corr(char *compare_type,int algorithm,struct corrs_data * data)
{
  int ii,status=0;

  /* Set up the CCL configuration for comparing to benchmarks
   * The benchmarks are of two types: those which use analytic
   * redshift distributions, and those which use histograms for
   * them. We will compare CCL correlations to the benchmarks
   * from CosmoLSS using estimated covariances from CosmoLike.
   */
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  config.matter_power_spectrum_method = ccl_linear;
  ccl_parameters params = ccl_parameters_create_flat_lcdm(data->Omega_c,data->Omega_b,data->h,
							  data->sigma8,data->n_s,&status);
  params.T_CMB=2.7;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);

  if(!strcmp(compare_type,"histo")) { //This is needed for the histogrammed N(z) in order to pass the IA tests
    cosmo->gsl_params.INTEGRATION_LIMBER_EPSREL = 2.5E-5;
    cosmo->gsl_params.INTEGRATION_EPSREL = 2.5E-5;
    ccl_set_debug_policy(CCL_DEBUG_MODE_OFF);
  }

  /*Create arrays for redshift distributions in the case of analytic benchmarks*/
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
    for(ii=0;ii<nz;ii++) {
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
  else { /*Load arrays for redshift distributions in the case of histograms*/
    char *rtn;
    char str[1024];
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
    rtn=fgets(str,1024,fnz1);
    rtn=fgets(str,1024,fnz2);

    for(ii=0;ii<nz;ii++) {
      int stat;
      double z1,z2,nz1,nz2,zia1,zia2,aia1,aia2;
      stat=fscanf(fnz1,"%lf %lf",&z1,&nz1);
      stat=fscanf(fnz2,"%lf %lf",&z2,&nz2);
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

  /*For the same configuration as the benchmarks, we will produce CCL
   correlation functions starting by computing C_ells here: */
  char fname[256];
  FILE *fi_dd_11,*fi_dd_12,*fi_dd_22;
  FILE *fi_ll_11_pp,*fi_ll_12_pp,*fi_ll_22_pp;
  FILE *fi_ll_11_mm,*fi_ll_12_mm,*fi_ll_22_mm;
  FILE *fi_li_11_pp,*fi_li_12_pp,*fi_li_22_pp;
  FILE *fi_li_11_mm,*fi_li_12_mm,*fi_li_22_mm;
  FILE *fi_ii_11_pp,*fi_ii_12_pp,*fi_ii_22_pp;
  FILE *fi_ii_11_mm,*fi_ii_12_mm,*fi_ii_22_mm;
  FILE *fi_lltot_11_pp,*fi_lltot_12_pp,*fi_lltot_22_pp;
  FILE *fi_lltot_11_mm,*fi_lltot_12_mm,*fi_lltot_22_mm;
  FILE *fi_dl_11,*fi_dl_12,*fi_dl_21,*fi_dl_22;
  FILE *fi_di_11,*fi_di_12,*fi_di_21,*fi_di_22;
  FILE *fi_dltot_11,*fi_dltot_12,*fi_dltot_21,*fi_dltot_22;
  int has_rsd=0,has_magnification=0, has_intrinsic_alignment=0;
  int status2=0;
  CCL_ClTracer *tr_nc_1=ccl_cl_tracer_number_counts_simple(cosmo,nz,zarr_1,pzarr_1,
							       nz,zarr_1,bzarr,&status2);
  ASSERT_NOT_NULL(tr_nc_1);
  CCL_ClTracer *tr_nc_2=ccl_cl_tracer_number_counts_simple(cosmo,nz,zarr_2,pzarr_2,
							       nz,zarr_2,bzarr,&status2);
  ASSERT_NOT_NULL(tr_nc_2);
  CCL_ClTracer *tr_wl_1=ccl_cl_tracer_lensing_simple(cosmo,nz,zarr_1,pzarr_1,&status2);
  ASSERT_NOT_NULL(tr_wl_1);
  CCL_ClTracer *tr_wl_2=ccl_cl_tracer_lensing_simple(cosmo,nz,zarr_2,pzarr_2,&status2);
  ASSERT_NOT_NULL(tr_wl_2);
  CCL_ClTracer *tr_wli_1=ccl_cl_tracer_lensing(cosmo,1,1,nz,zarr_1,pzarr_1,nz,zarr_1,az1arr,nz,zarr_1,rz1arr,&status);
  ASSERT_NOT_NULL(tr_wli_1);
  CCL_ClTracer *tr_wli_2=ccl_cl_tracer_lensing(cosmo,1,1,nz,zarr_2,pzarr_2,nz,zarr_2,az2arr,nz,zarr_2,rz2arr,&status);
  ASSERT_NOT_NULL(tr_wli_2);

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
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_wt_li_pp.txt",compare_type);
  fi_li_11_pp=fopen(fname,"r"); ASSERT_NOT_NULL(fi_li_11_pp);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_wt_li_pp.txt",compare_type);
  fi_li_12_pp=fopen(fname,"r"); ASSERT_NOT_NULL(fi_li_12_pp);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_wt_li_pp.txt",compare_type);
  fi_li_22_pp=fopen(fname,"r"); ASSERT_NOT_NULL(fi_li_22_pp);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_wt_li_mm.txt",compare_type);
  fi_li_11_mm=fopen(fname,"r"); ASSERT_NOT_NULL(fi_li_11_mm);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_wt_li_mm.txt",compare_type);
  fi_li_12_mm=fopen(fname,"r"); ASSERT_NOT_NULL(fi_li_12_mm);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_wt_li_mm.txt",compare_type);
  fi_li_22_mm=fopen(fname,"r"); ASSERT_NOT_NULL(fi_li_22_mm);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_wt_ii_pp.txt",compare_type);
  fi_ii_11_pp=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ii_11_pp);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_wt_ii_pp.txt",compare_type);
  fi_ii_12_pp=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ii_12_pp);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_wt_ii_pp.txt",compare_type);
  fi_ii_22_pp=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ii_22_pp);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_wt_ii_mm.txt",compare_type);
  fi_ii_11_mm=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ii_11_mm);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_wt_ii_mm.txt",compare_type);
  fi_ii_12_mm=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ii_12_mm);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_wt_ii_mm.txt",compare_type);
  fi_ii_22_mm=fopen(fname,"r"); ASSERT_NOT_NULL(fi_ii_22_mm);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_wt_lltot_pp.txt",compare_type);
  fi_lltot_11_pp=fopen(fname,"r"); ASSERT_NOT_NULL(fi_lltot_11_pp);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_wt_lltot_pp.txt",compare_type);
  fi_lltot_12_pp=fopen(fname,"r"); ASSERT_NOT_NULL(fi_lltot_12_pp);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_wt_lltot_pp.txt",compare_type);
  fi_lltot_22_pp=fopen(fname,"r"); ASSERT_NOT_NULL(fi_lltot_22_pp);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_wt_lltot_mm.txt",compare_type);
  fi_lltot_11_mm=fopen(fname,"r"); ASSERT_NOT_NULL(fi_lltot_11_mm);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_wt_lltot_mm.txt",compare_type);
  fi_lltot_12_mm=fopen(fname,"r"); ASSERT_NOT_NULL(fi_lltot_12_mm);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_wt_lltot_mm.txt",compare_type);
  fi_lltot_22_mm=fopen(fname,"r"); ASSERT_NOT_NULL(fi_lltot_22_mm);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_wt_dl.txt",compare_type);
  fi_dl_11=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dl_11);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_wt_dl.txt",compare_type);
  fi_dl_12=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dl_12);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b1%s_log_wt_dl.txt",compare_type);
  fi_dl_21=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dl_21);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_wt_dl.txt",compare_type);
  fi_dl_22=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dl_22);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_wt_di.txt",compare_type);
  fi_di_11=fopen(fname,"r"); ASSERT_NOT_NULL(fi_di_11);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_wt_di.txt",compare_type);
  fi_di_12=fopen(fname,"r"); ASSERT_NOT_NULL(fi_di_12);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b1%s_log_wt_di.txt",compare_type);
  fi_di_21=fopen(fname,"r"); ASSERT_NOT_NULL(fi_di_21);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_wt_di.txt",compare_type);
  fi_di_22=fopen(fname,"r"); ASSERT_NOT_NULL(fi_di_22);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_wt_dltot.txt",compare_type);
  fi_dltot_11=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dltot_11);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_wt_dltot.txt",compare_type);
  fi_dltot_12=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dltot_12);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b1%s_log_wt_dltot.txt",compare_type);
  fi_dltot_21=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dltot_21);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_wt_dltot.txt",compare_type);
  fi_dltot_22=fopen(fname,"r"); ASSERT_NOT_NULL(fi_dltot_22);

  int nofl=15;
  double taper_cl_limits[4]={1,2,10000,15000};
  double wt_dd_11[nofl],wt_dd_12[nofl],wt_dd_22[nofl];
  double wt_ll_11_mm[nofl],wt_ll_12_mm[nofl],wt_ll_22_mm[nofl];
  double wt_ll_11_pp[nofl],wt_ll_12_pp[nofl],wt_ll_22_pp[nofl];
  double wt_li_11_mm[nofl],wt_li_12_mm[nofl],wt_li_22_mm[nofl];
  double wt_li_11_pp[nofl],wt_li_12_pp[nofl],wt_li_22_pp[nofl];
  double wt_ii_11_mm[nofl],wt_ii_12_mm[nofl],wt_ii_22_mm[nofl];
  double wt_ii_11_pp[nofl],wt_ii_12_pp[nofl],wt_ii_22_pp[nofl];
  double wt_lltot_11_mm[nofl],wt_lltot_12_mm[nofl],wt_lltot_22_mm[nofl];
  double wt_lltot_11_pp[nofl],wt_lltot_12_pp[nofl],wt_lltot_22_pp[nofl];
  double wt_dl_11[nofl],wt_dl_12[nofl],wt_dl_21[nofl],wt_dl_22[nofl];
  double wt_di_11[nofl],wt_di_12[nofl],wt_di_21[nofl],wt_di_22[nofl];
  double wt_dltot_11[nofl],wt_dltot_12[nofl],wt_dltot_21[nofl],wt_dltot_22[nofl];
  double *wt_dd_11_h,*wt_dd_12_h,*wt_dd_22_h;
  double *wt_ll_11_h_mm,*wt_ll_12_h_mm,*wt_ll_22_h_mm;
  double *wt_ll_11_h_pp,*wt_ll_12_h_pp,*wt_ll_22_h_pp;
  double *wt_li_11_h_mm,*wt_li_12_h_mm,*wt_li_22_h_mm;
  double *wt_li_11_h_pp,*wt_li_12_h_pp,*wt_li_22_h_pp;
  double *wt_ii_11_h_mm,*wt_ii_12_h_mm,*wt_ii_22_h_mm;
  double *wt_ii_11_h_pp,*wt_ii_12_h_pp,*wt_ii_22_h_pp;
  double *wt_lltot_11_h_mm,*wt_lltot_12_h_mm,*wt_lltot_22_h_mm;
  double *wt_lltot_11_h_pp,*wt_lltot_12_h_pp,*wt_lltot_22_h_pp;
  double *wt_dl_11_h,*wt_dl_12_h,*wt_dl_21_h,*wt_dl_22_h;
  double *wt_di_11_h,*wt_di_12_h,*wt_di_21_h,*wt_di_22_h;
  double *wt_dltot_11_h,*wt_dltot_12_h,*wt_dltot_21_h,*wt_dltot_22_h;
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
    stat=fscanf(fi_li_11_pp,"%lf %lf",&dum,&wt_li_11_pp[ii]);
    stat=fscanf(fi_li_12_pp,"%lf %lf",&dum,&wt_li_12_pp[ii]);
    stat=fscanf(fi_li_22_pp,"%lf %lf",&dum,&wt_li_22_pp[ii]);
    stat=fscanf(fi_li_11_mm,"%lf %lf",&dum,&wt_li_11_mm[ii]);
    stat=fscanf(fi_li_12_mm,"%lf %lf",&dum,&wt_li_12_mm[ii]);
    stat=fscanf(fi_li_22_mm,"%lf %lf",&dum,&wt_li_22_mm[ii]);
    stat=fscanf(fi_ii_11_pp,"%lf %lf",&dum,&wt_ii_11_pp[ii]);
    stat=fscanf(fi_ii_12_pp,"%lf %lf",&dum,&wt_ii_12_pp[ii]);
    stat=fscanf(fi_ii_22_pp,"%lf %lf",&dum,&wt_ii_22_pp[ii]);
    stat=fscanf(fi_ii_11_mm,"%lf %lf",&dum,&wt_ii_11_mm[ii]);
    stat=fscanf(fi_ii_12_mm,"%lf %lf",&dum,&wt_ii_12_mm[ii]);
    stat=fscanf(fi_ii_22_mm,"%lf %lf",&dum,&wt_ii_22_mm[ii]);
    stat=fscanf(fi_lltot_11_pp,"%lf %lf",&dum,&wt_lltot_11_pp[ii]);
    stat=fscanf(fi_lltot_12_pp,"%lf %lf",&dum,&wt_lltot_12_pp[ii]);
    stat=fscanf(fi_lltot_22_pp,"%lf %lf",&dum,&wt_lltot_22_pp[ii]);
    stat=fscanf(fi_lltot_11_mm,"%lf %lf",&dum,&wt_lltot_11_mm[ii]);
    stat=fscanf(fi_lltot_12_mm,"%lf %lf",&dum,&wt_lltot_12_mm[ii]);
    stat=fscanf(fi_lltot_22_mm,"%lf %lf",&dum,&wt_lltot_22_mm[ii]);
    stat=fscanf(fi_dl_11,"%lf %lf",&dum,&wt_dl_11[ii]);
    stat=fscanf(fi_dl_12,"%lf %lf",&dum,&wt_dl_12[ii]);
    stat=fscanf(fi_dl_21,"%lf %lf",&dum,&wt_dl_21[ii]);
    stat=fscanf(fi_dl_22,"%lf %lf",&dum,&wt_dl_22[ii]);
    stat=fscanf(fi_di_11,"%lf %lf",&dum,&wt_di_11[ii]);
    stat=fscanf(fi_di_12,"%lf %lf",&dum,&wt_di_12[ii]);
    stat=fscanf(fi_di_21,"%lf %lf",&dum,&wt_di_21[ii]);
    stat=fscanf(fi_di_22,"%lf %lf",&dum,&wt_di_22[ii]);
    stat=fscanf(fi_dltot_11,"%lf %lf",&dum,&wt_dltot_11[ii]);
    stat=fscanf(fi_dltot_12,"%lf %lf",&dum,&wt_dltot_12[ii]);
    stat=fscanf(fi_dltot_21,"%lf %lf",&dum,&wt_dltot_21[ii]);
    stat=fscanf(fi_dltot_22,"%lf %lf",&dum,&wt_dltot_22[ii]);
  }
  fclose(fi_dd_11); fclose(fi_dd_12); fclose(fi_dd_22);
  fclose(fi_ll_11_pp); fclose(fi_ll_12_pp); fclose(fi_ll_22_pp);
  fclose(fi_ll_11_mm); fclose(fi_ll_12_mm); fclose(fi_ll_22_mm);
  fclose(fi_li_11_pp); fclose(fi_li_12_pp); fclose(fi_li_22_pp);
  fclose(fi_li_11_mm); fclose(fi_li_12_mm); fclose(fi_li_22_mm);
  fclose(fi_ii_11_pp); fclose(fi_ii_12_pp); fclose(fi_ii_22_pp);
  fclose(fi_ii_11_mm); fclose(fi_ii_12_mm); fclose(fi_ii_22_mm);
  fclose(fi_lltot_11_pp); fclose(fi_lltot_12_pp); fclose(fi_lltot_22_pp);
  fclose(fi_lltot_11_mm); fclose(fi_lltot_12_mm); fclose(fi_lltot_22_mm);
  fclose(fi_dl_11); fclose(fi_dl_12); fclose(fi_dl_21); fclose(fi_dl_22);
  fclose(fi_di_11); fclose(fi_di_12); fclose(fi_di_21); fclose(fi_di_22);
  fclose(fi_dltot_11); fclose(fi_dltot_12); fclose(fi_dltot_21); fclose(fi_dltot_22);

  /*Compute the correlation with CCL*/
  double *clarr=malloc(ELL_MAX_CL*sizeof(double));
  double *clarr1=malloc(ELL_MAX_CL*sizeof(double));
  double *clarr2=malloc(ELL_MAX_CL*sizeof(double));
  double *clarr3=malloc(ELL_MAX_CL*sizeof(double));
  double *clarr4=malloc(ELL_MAX_CL*sizeof(double));
  double *larr=malloc(ELL_MAX_CL*sizeof(double));
  int *ells=malloc(ELL_MAX_CL*sizeof(int)); // ccl_angular_cls needs int
  for(int il=0;il<ELL_MAX_CL;il++){
    larr[il]=il;
    ells[il]=il;
  }

  //Here, we are degrading CORR_ERROR_FRACTION by fftlogfactor (only deviates from 1 for FFTLog, i.e. this factor is only applied when using FFTLog for integration)
  fftlogfactor=1.0;
  if(algorithm==1002){
    fftlogfactor = 2.0;
  }

  /*Use Limber computation*/
  wt_dd_11_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_1,tr_nc_1,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dd_11_h,CCL_CORR_GG,
		  0,taper_cl_limits,algorithm,&status);
  wt_dd_12_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_1,tr_nc_2,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dd_12_h,CCL_CORR_GG,
		  0,taper_cl_limits,algorithm,&status);
  wt_dd_22_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_2,tr_nc_2,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dd_22_h,CCL_CORR_GG,
		  0,taper_cl_limits,algorithm,&status);

  wt_ll_11_h_mm=malloc(nofl*sizeof(double));
  wt_ll_11_h_pp=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_wl_1,tr_wl_1,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ll_11_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ll_11_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);
  wt_ll_12_h_mm=malloc(nofl*sizeof(double));
  wt_ll_12_h_pp=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_wl_1,tr_wl_2,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ll_12_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ll_12_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);
  wt_ll_22_h_mm=malloc(nofl*sizeof(double));
  wt_ll_22_h_pp=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_wl_2,tr_wl_2,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ll_22_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ll_22_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);

  wt_li_11_h_mm=malloc(nofl*sizeof(double));
  wt_li_11_h_pp=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_wl_1,tr_wli_1,clarr1,&status);
  get_cls_arr(cosmo,tr_wl_1,tr_wl_1,clarr2,&status);
  for(int il=0;il<ELL_MAX_CL;il++){
    clarr[il]=2*(clarr1[il]-clarr2[il]);
  }
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_li_11_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_li_11_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);
  wt_li_12_h_mm=malloc(nofl*sizeof(double));
  wt_li_12_h_pp=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_wl_1,tr_wli_2,clarr1,&status);
  get_cls_arr(cosmo,tr_wli_1,tr_wl_2,clarr2,&status);
  get_cls_arr(cosmo,tr_wl_1,tr_wl_2,clarr3,&status);
  for(int il=0;il<ELL_MAX_CL;il++){
    clarr[il]=clarr1[il]+clarr2[il]-2*clarr3[il];
  }
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_li_12_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_li_12_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);
  wt_li_22_h_mm=malloc(nofl*sizeof(double));
  wt_li_22_h_pp=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_wl_2,tr_wli_2,clarr1,&status);
  get_cls_arr(cosmo,tr_wl_2,tr_wl_2,clarr2,&status);
  for(int il=0;il<ELL_MAX_CL;il++){
    clarr[il]=2*(clarr1[il]-clarr2[il]);
  }
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_li_22_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_li_22_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);

  wt_ii_11_h_mm=malloc(nofl*sizeof(double));
  wt_ii_11_h_pp=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_wli_1,tr_wli_1,clarr1,&status);
  get_cls_arr(cosmo,tr_wl_1,tr_wl_1,clarr2,&status);
  get_cls_arr(cosmo,tr_wl_1,tr_wli_1,clarr3,&status);
  for(int il=0;il<ELL_MAX_CL;il++){
    clarr[il]=clarr1[il]+clarr2[il]-2*clarr3[il];
  }
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ii_11_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ii_11_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);
  wt_ii_12_h_mm=malloc(nofl*sizeof(double));
  wt_ii_12_h_pp=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_wli_1,tr_wli_2,clarr1,&status);
  get_cls_arr(cosmo,tr_wl_1,tr_wl_2,clarr2,&status);
  get_cls_arr(cosmo,tr_wl_1,tr_wli_2,clarr3,&status);
  get_cls_arr(cosmo,tr_wli_1,tr_wl_2,clarr4,&status);
  for(int il=0;il<ELL_MAX_CL;il++){
    clarr[il]=clarr1[il]+clarr2[il]-clarr3[il]-clarr4[il];
  }
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ii_12_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ii_12_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);
  wt_ii_22_h_mm=malloc(nofl*sizeof(double));
  wt_ii_22_h_pp=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_wli_2,tr_wli_2,clarr1,&status);
  get_cls_arr(cosmo,tr_wl_2,tr_wl_2,clarr2,&status);
  get_cls_arr(cosmo,tr_wl_2,tr_wli_2,clarr3,&status);
  for(int il=0;il<ELL_MAX_CL;il++){
    clarr[il]=clarr1[il]+clarr2[il]-2*clarr3[il];
  }
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ii_22_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_ii_22_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);

  wt_lltot_11_h_mm=malloc(nofl*sizeof(double));
  wt_lltot_11_h_pp=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_wli_1,tr_wli_1,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_lltot_11_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_lltot_11_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);
  wt_lltot_12_h_mm=malloc(nofl*sizeof(double));
  wt_lltot_12_h_pp=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_wli_1,tr_wli_2,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_lltot_12_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_lltot_12_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);
  wt_lltot_22_h_mm=malloc(nofl*sizeof(double));
  wt_lltot_22_h_pp=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_wli_2,tr_wli_2,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_lltot_22_h_pp,CCL_CORR_LP,
		  0,taper_cl_limits,algorithm,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_lltot_22_h_mm,CCL_CORR_LM,
		  0,taper_cl_limits,algorithm,&status);

  wt_dl_11_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_1,tr_wl_1,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dl_11_h,CCL_CORR_GL,
		  0,taper_cl_limits,algorithm,&status);
  wt_dl_12_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_1,tr_wl_2,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dl_12_h,CCL_CORR_GL,
		  0,taper_cl_limits,algorithm,&status);
  wt_dl_21_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_2,tr_wl_1,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dl_21_h,CCL_CORR_GL,
		  0,taper_cl_limits,algorithm,&status);
  wt_dl_22_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_2,tr_wl_2,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dl_22_h,CCL_CORR_GL,
		  0,taper_cl_limits,algorithm,&status);

  wt_dltot_11_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_1,tr_wli_1,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dltot_11_h,CCL_CORR_GL,
		  0,taper_cl_limits,algorithm,&status);
  wt_dltot_12_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_1,tr_wli_2,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dltot_12_h,CCL_CORR_GL,
		  0,taper_cl_limits,algorithm,&status);
  wt_dltot_21_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_2,tr_wli_1,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dltot_21_h,CCL_CORR_GL,
		  0,taper_cl_limits,algorithm,&status);
  wt_dltot_22_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_2,tr_wli_2,clarr,&status);
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_dltot_22_h,CCL_CORR_GL,
		  0,taper_cl_limits,algorithm,&status);

  wt_di_11_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_1,tr_wl_1,clarr1,&status);
  get_cls_arr(cosmo,tr_nc_1,tr_wli_1,clarr2,&status);
  for(int il=0;il<ELL_MAX_CL;il++){
    clarr[il]=clarr2[il]-clarr1[il];
  }
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_di_11_h,CCL_CORR_GL,
		  0,taper_cl_limits,algorithm,&status);
  wt_di_12_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_1,tr_wl_2,clarr1,&status);
  get_cls_arr(cosmo,tr_nc_1,tr_wli_2,clarr2,&status);
  for(int il=0;il<ELL_MAX_CL;il++){
    clarr[il]=clarr2[il]-clarr1[il];
  }
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_di_12_h,CCL_CORR_GL,
		  0,taper_cl_limits,algorithm,&status);
  wt_di_21_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_2,tr_wl_1,clarr1,&status);
  get_cls_arr(cosmo,tr_nc_2,tr_wli_1,clarr2,&status);
  for(int il=0;il<ELL_MAX_CL;il++){
    clarr[il]=clarr2[il]-clarr1[il];
  }
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_di_21_h,CCL_CORR_GL,
		  0,taper_cl_limits,algorithm,&status);
  wt_di_22_h=malloc(nofl*sizeof(double));
  get_cls_arr(cosmo,tr_nc_2,tr_wl_2,clarr1,&status);
  get_cls_arr(cosmo,tr_nc_2,tr_wli_2,clarr2,&status);
  for(int il=0;il<ELL_MAX_CL;il++){
    clarr[il]=clarr2[il]-clarr1[il];
  }
  ccl_correlation(cosmo,ELL_MAX_CL,larr,clarr,nofl,theta_in,wt_di_22_h,CCL_CORR_GL,
		  0,taper_cl_limits,algorithm,&status);

  free(clarr);
  free(clarr1);
  free(clarr2);
  free(clarr3);
  free(clarr4);
  free(larr);

  /* With the CCL correlation already computed, we proceed to the
  * comparison. Here, we read in the benchmark covariances from CosmoLike, which
  * allow us to set our tolerance.
  */
  int nsig=15;
  double sigwt_dd_11[15], sigwt_dd_22[15];
  double sigwt_dl_11[15], sigwt_dl_12[15], sigwt_dl_21[15], sigwt_dl_22[15];
  double sigwt_ll_12_mm[15], sigwt_ll_12_pp[15];
  double sigwt_ll_11_mm[15], sigwt_ll_22_mm[15];
  double sigwt_ll_11_pp[15], sigwt_ll_22_pp[15];
  double sig_theta_in[15];

  char bs[1024];
  FILE *fi_dl_sig=fopen("tests/benchmark/cov_corr/sigma_ggl_Nbin5","r");
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
  if(fgets(bs,sizeof(bs),fi_dl_sig)==NULL) {
    fprintf(stderr,"Error reading file\n");
    exit(1);
  }
  for(int ii=0;ii<nsig;ii++) {
    int stat;
    double dum;
    stat=fscanf(fi_dd_sig,"%le %le %le %le",&sig_theta_in[ii],&sigwt_dd_11[ii],&sigwt_dd_22[ii],&dum);
    stat=fscanf(fi_dl_sig,"%le %le %le %le %le",&sig_theta_in[ii],&sigwt_dl_12[ii],&sigwt_dl_11[ii],&sigwt_dl_22[ii],&sigwt_dl_21[ii]);
    stat=fscanf(fi_pp_sig,"%le %le %le %le",&sig_theta_in[ii],&sigwt_ll_11_pp[ii],&sigwt_ll_22_pp[ii],&sigwt_ll_12_pp[ii]);
    stat=fscanf(fi_mm_sig,"%le %le %le %le",&sig_theta_in[ii],&sigwt_ll_11_mm[ii],&sigwt_ll_22_mm[ii],&sigwt_ll_12_mm[ii]);
    sig_theta_in[ii]=sig_theta_in[ii]/60.; //convert to deg
  }
  fclose(fi_dd_sig);
  fclose(fi_mm_sig);
  fclose(fi_pp_sig);
  fclose(fi_dl_sig);
  /* Spline the covariances */
  gsl_spline *spl_sigwt_dd_11   =gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_dd_11   ,sig_theta_in,sigwt_dd_11   ,nsig);
  gsl_spline *spl_sigwt_dd_22   =gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_dd_22   ,sig_theta_in,sigwt_dd_22   ,nsig);
  gsl_spline *spl_sigwt_ll_11_pp=gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_ll_11_pp,sig_theta_in,sigwt_ll_11_pp,nsig);
  gsl_spline *spl_sigwt_ll_22_pp=gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_ll_22_pp,sig_theta_in,sigwt_ll_22_pp,nsig);
  gsl_spline *spl_sigwt_ll_12_pp=gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_ll_12_pp,sig_theta_in,sigwt_ll_12_pp,nsig);
  gsl_spline *spl_sigwt_ll_11_mm=gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_ll_11_mm,sig_theta_in,sigwt_ll_11_mm,nsig);
  gsl_spline *spl_sigwt_ll_22_mm=gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_ll_22_mm,sig_theta_in,sigwt_ll_22_mm,nsig);
  gsl_spline *spl_sigwt_ll_12_mm=gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_ll_12_mm,sig_theta_in,sigwt_ll_12_mm,nsig);
  gsl_spline *spl_sigwt_dl_11   =gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_dl_11,sig_theta_in,sigwt_dl_11 ,nsig);
  gsl_spline *spl_sigwt_dl_12   =gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_dl_12,sig_theta_in,sigwt_dl_12 ,nsig);
  gsl_spline *spl_sigwt_dl_21   =gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_dl_21,sig_theta_in,sigwt_dl_21 ,nsig);
  gsl_spline *spl_sigwt_dl_22   =gsl_spline_alloc(L_SPLINE_TYPE,nsig);
  gsl_spline_init(spl_sigwt_dl_22,sig_theta_in,sigwt_dl_22 ,nsig);

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
    ASSERT_TRUE(fabs(wt_dd_11_h[ii]-wt_dd_11[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    //The dd_12 term commented out below because do not currently have the covariance.
    //    tol=gsl_spline_eval(spl_sigwt_dd_12,theta_in[ii],NULL);
    //    ASSERT_TRUE(fabs(wt_dd_12_h_pp[ii]-wt_dd_12_pp[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_dd_22,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_dd_22_h[ii]-wt_dd_22[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);

    //Only considering the GG covariance since do not have one with intrinsic alignments included.
    //Also assuming covariance approximately the same for analytic and histogram n(z).
    tol=gsl_spline_eval(spl_sigwt_ll_11_pp,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_ll_11_h_pp[ii]-wt_ll_11_pp[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_12_pp,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_ll_12_h_pp[ii]-wt_ll_12_pp[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_22_pp,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_ll_22_h_pp[ii]-wt_ll_22_pp[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_11_mm,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_ll_11_h_mm[ii]-wt_ll_11_mm[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_12_mm,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_ll_12_h_mm[ii]-wt_ll_12_mm[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_22_mm,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_ll_22_h_mm[ii]-wt_ll_22_mm[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_11_pp,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_li_11_h_pp[ii]-wt_li_11_pp[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_12_pp,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_li_12_h_pp[ii]-wt_li_12_pp[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_22_pp,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_li_22_h_pp[ii]-wt_li_22_pp[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_11_mm,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_li_11_h_mm[ii]-wt_li_11_mm[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_12_mm,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_li_12_h_mm[ii]-wt_li_12_mm[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_22_mm,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_li_22_h_mm[ii]-wt_li_22_mm[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_11_pp,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_ii_11_h_pp[ii]-wt_ii_11_pp[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_12_pp,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_ii_12_h_pp[ii]-wt_ii_12_pp[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_22_pp,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_ii_22_h_pp[ii]-wt_ii_22_pp[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_11_mm,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_ii_11_h_mm[ii]-wt_ii_11_mm[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_12_mm,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_ii_12_h_mm[ii]-wt_ii_12_mm[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_22_mm,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_ii_22_h_mm[ii]-wt_ii_22_mm[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_11_pp,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_lltot_11_h_pp[ii]-wt_lltot_11_pp[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_12_pp,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_lltot_12_h_pp[ii]-wt_lltot_12_pp[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_22_pp,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_lltot_22_h_pp[ii]-wt_lltot_22_pp[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_11_mm,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_lltot_11_h_mm[ii]-wt_lltot_11_mm[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_12_mm,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_lltot_12_h_mm[ii]-wt_lltot_12_mm[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_ll_22_mm,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_lltot_22_h_mm[ii]-wt_lltot_22_mm[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);

    //GGL terms. Analogous to cosmic shear, only considering the gG covariance since do not have one with intrinsic alignments included.
    tol=gsl_spline_eval(spl_sigwt_dl_11,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_dl_11_h[ii]-wt_dl_11[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_dl_12,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_dl_12_h[ii]-wt_dl_12[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_dl_21,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_dl_21_h[ii]-wt_dl_21[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_dl_22,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_dl_22_h[ii]-wt_dl_22[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_dl_11,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_di_11_h[ii]-wt_di_11[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_dl_12,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_di_12_h[ii]-wt_di_12[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_dl_21,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_di_21_h[ii]-wt_di_21[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_dl_22,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_di_22_h[ii]-wt_di_22[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_dl_11,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_dltot_11_h[ii]-wt_dltot_11[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_dl_12,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_dltot_12_h[ii]-wt_dltot_12[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_dl_21,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_dltot_21_h[ii]-wt_dltot_21[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
    tol=gsl_spline_eval(spl_sigwt_dl_22,theta_in[ii],NULL);
    ASSERT_TRUE(fabs(wt_dltot_22_h[ii]-wt_dltot_22[ii])<tol*CORR_ERROR_FRACTION*fftlogfactor);
  }

  //Free splines, cosmology and arrays
  gsl_spline_free(spl_sigwt_dd_11);
  gsl_spline_free(spl_sigwt_dd_22);
  gsl_spline_free(spl_sigwt_dl_11);
  gsl_spline_free(spl_sigwt_dl_12);
  gsl_spline_free(spl_sigwt_dl_21);
  gsl_spline_free(spl_sigwt_dl_22);
  gsl_spline_free(spl_sigwt_ll_11_pp);
  gsl_spline_free(spl_sigwt_ll_22_pp);
  gsl_spline_free(spl_sigwt_ll_12_pp);
  gsl_spline_free(spl_sigwt_ll_11_mm);
  gsl_spline_free(spl_sigwt_ll_22_mm);
  gsl_spline_free(spl_sigwt_ll_12_mm);
  free(wt_dd_11_h); free(wt_dd_12_h); free(wt_dd_22_h);
  free(wt_ll_11_h_pp); free(wt_ll_12_h_pp); free(wt_ll_22_h_pp);
  free(wt_ll_11_h_mm); free(wt_ll_12_h_mm); free(wt_ll_22_h_mm);
  free(wt_li_11_h_pp); free(wt_li_12_h_pp); free(wt_li_22_h_pp);
  free(wt_li_11_h_mm); free(wt_li_12_h_mm); free(wt_li_22_h_mm);
  free(wt_ii_11_h_pp); free(wt_ii_12_h_pp); free(wt_ii_22_h_pp);
  free(wt_ii_11_h_mm); free(wt_ii_12_h_mm); free(wt_ii_22_h_mm);
  free(wt_lltot_11_h_pp); free(wt_lltot_12_h_pp); free(wt_lltot_22_h_pp);
  free(wt_lltot_11_h_mm); free(wt_lltot_12_h_mm); free(wt_lltot_22_h_mm);
  free(wt_dl_11_h); free(wt_dl_12_h); free(wt_dl_21_h); free(wt_dl_22_h);
  free(wt_di_11_h); free(wt_di_12_h); free(wt_di_21_h); free(wt_di_22_h);
  free(wt_dltot_11_h); free(wt_dltot_12_h); free(wt_dltot_21_h); free(wt_dltot_22_h);
  free(zarr_1); free(zarr_2);
  free(pzarr_1); free(pzarr_2);
  free(bzarr);
  free(az1arr); free(az2arr);
  free(rz1arr); free(rz2arr);
  ccl_cosmology_free(cosmo);
  if(!strcmp(compare_type,"histo")) {
    ccl_set_debug_policy(CCL_DEBUG_MODE_WARNING);
  }
}

CTEST2(corrs,analytic_fftlog) {
  compare_corr("analytic",CCL_CORR_FFTLOG,data);
}

CTEST2(corrs,histo_fftlog) {
  compare_corr("histo",CCL_CORR_FFTLOG,data);
}

CTEST2(corrs,analytic_bessel) {
  compare_corr("analytic",CCL_CORR_BESSEL,data);
}

CTEST2(corrs,histo_bessel) {
  compare_corr("histo",CCL_CORR_BESSEL,data);
}
