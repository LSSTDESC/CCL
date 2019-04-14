#include "ccl.h"
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

static double *read_cell(char *fname)
{
  double *cls=malloc(NELLS*sizeof(double));
  FILE *fi=fopen(fname,"r");
  ASSERT_NOT_NULL(fi);
  for(int ii=0;ii<NELLS;ii++) {
    int l,stat;
    stat=fscanf(fi,"%d %lf\n",&l,&(cls[ii]));
    if(stat!=2) {
      fprintf(stderr,"Error reading bechmark file %s\n",fname);
      exit(1);
    }
  }
  return cls;
}

//Wrapper that computes array of power spectra 
static void compare_cls_arr(ccl_cosmology *cosmo,CCL_ClTracer *tr1,CCL_ClTracer *tr2,
			    int nl,int *ls,double *cls_bm,
			    double *cl_a1b1,double *cl_a1b2,double *cl_a2b1,double *cl_a2b2,
			    double *ell_correct,int *status)
{
  //Loop over ells
  for(int ii=0;ii<nl;ii++) {
    int l=ls[ii];
    double cl_here=ccl_angular_cl_limber(cosmo,tr1,tr2,NULL,(double)l,status)*ell_correct[ii];
    double el_here=ELS_TOLERANCE*sqrt((cl_a1b1[l]*cl_a2b2[l]+cl_a2b1[l]*cl_a1b2[l])/(2*l+1.));
    ASSERT_TRUE(fabs(cl_here-cls_bm[ls[ii]])<el_here);
  }
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
  params.T_CMB=2.7;
  params.Omega_n_rel=0;
  params.Omega_l = 0.7;
  params.sigma8=data->sigma8;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  ASSERT_NOT_NULL(cosmo);

  double epsrel_save;
  if(!strcmp(compare_type,"histo")) { //This is needed for the histogrammed N(z) in order to pass the IA tests
    epsrel_save = cosmo->gsl_params.INTEGRATION_LIMBER_EPSREL;
    cosmo->gsl_params.INTEGRATION_LIMBER_EPSREL = 2.5E-5;
    cosmo->gsl_params.INTEGRATION_EPSREL = 2.5E-5;
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
  FILE *fi_li_11,*fi_li_12,*fi_li_22;
  CCL_ClTracer *tr_nc_1=ccl_cl_tracer_number_counts_simple(cosmo,nz,zarr_1,pzarr_1,
							   nz,zarr_1,bzarr,&status);
  ASSERT_NOT_NULL(tr_nc_1);
  CCL_ClTracer *tr_nc_2=ccl_cl_tracer_number_counts_simple(cosmo,nz,zarr_2,pzarr_2,
							   nz,zarr_2,bzarr,&status);
  ASSERT_NOT_NULL(tr_nc_2);
  CCL_ClTracer *tr_wl_1=ccl_cl_tracer_lensing_simple(cosmo,nz,zarr_1,pzarr_1,&status);
  ASSERT_NOT_NULL(tr_wl_1);
  CCL_ClTracer *tr_wl_2=ccl_cl_tracer_lensing_simple(cosmo,nz,zarr_2,pzarr_2,&status);
  ASSERT_NOT_NULL(tr_wl_2);
  CCL_ClTracer *tr_ia_1=ccl_cl_tracer_lensing(cosmo,0,1,nz,zarr_1,pzarr_1,
					       nz,zarr_1,az1arr,nz,zarr_1,rz1arr,&status);
  ASSERT_NOT_NULL(tr_ia_1);
  CCL_ClTracer *tr_ia_2=ccl_cl_tracer_lensing(cosmo,0,1,nz,zarr_2,pzarr_2,
					       nz,zarr_2,az2arr,nz,zarr_2,rz2arr,&status);
  ASSERT_NOT_NULL(tr_ia_2);
  CCL_ClTracer *tr_cl=ccl_cl_tracer_cmblens(cosmo,zlss,&status);
  ASSERT_NOT_NULL(tr_cl);

  //Read benchmark power spectra
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_dd.txt",compare_type);
  double *cls_dd_11_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_dd.txt",compare_type);
  double *cls_dd_12_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_dd.txt",compare_type);
  double *cls_dd_22_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_dl.txt",compare_type);
  double *cls_dl_11_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_dl.txt",compare_type);
  double *cls_dl_12_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b1%s_log_cl_dl.txt",compare_type);
  double *cls_dl_21_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_dl.txt",compare_type);
  double *cls_dl_22_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_di.txt",compare_type);
  double *cls_di_11_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_di.txt",compare_type);
  double *cls_di_12_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b1%s_log_cl_di.txt",compare_type);
  double *cls_di_21_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_di.txt",compare_type);
  double *cls_di_22_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_dc.txt",compare_type);
  double *cls_dc_1_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_dc.txt",compare_type);
  double *cls_dc_2_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_ll.txt",compare_type);
  double *cls_ll_11_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_ll.txt",compare_type);
  double *cls_ll_12_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_ll.txt",compare_type);
  double *cls_ll_22_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_li.txt",compare_type);
  double *cls_li_11_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_li.txt",compare_type);
  double *cls_li_12_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_li.txt",compare_type);
  double *cls_li_22_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_lc.txt",compare_type);
  double *cls_lc_1_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_lc.txt",compare_type);
  double *cls_lc_2_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b1%s_log_cl_ii.txt",compare_type);
  double *cls_ii_11_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b1b2%s_log_cl_ii.txt",compare_type);
  double *cls_ii_12_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_b2b2%s_log_cl_ii.txt",compare_type);
  double *cls_ii_22_b=read_cell(fname);
  sprintf(fname,"tests/benchmark/codecomp_step2_outputs/run_log_cl_cc.txt");
  double *cls_cc_b=read_cell(fname);

  //Calculate array of ells and ell corrections
  int nls=639;
  int *ells=malloc(nls*sizeof(int));
  double *ell_correct_one=malloc(nls*sizeof(double));
  double *ell_correct_dl=malloc(nls*sizeof(double));
  double *ell_correct_ll=malloc(nls*sizeof(double));
  double *ell_correct_lc=malloc(nls*sizeof(double));
  double *ell_correct_li=malloc(nls*sizeof(double));
  for(int ii=0;ii<50;ii++)
    ells[ii]=2+ii;
  for(int ii=50;ii<nls;ii++)
    ells[ii]=ells[ii-1]+5;
  for(int ii=0;ii<nls;ii++) {
    double l=(double)(ells[ii]);
    ell_correct_one[ii]=1;
    ell_correct_dl[ii]=(l+0.5)*(l+0.5)/sqrt((l+2.)*(l+1.)*l*(l-1.));
    ell_correct_ll[ii]=ell_correct_dl[ii]*ell_correct_dl[ii];
    ell_correct_lc[ii]=l*(l+1.)/sqrt((l+2.)*(l+1.)*l*(l-1.));
    ell_correct_li[ii]=2*ell_correct_dl[ii];
  }

  //Now compute all power spectra and compare with benchmarks
  //NC1-NC1
  compare_cls_arr(cosmo,tr_nc_1,tr_nc_1,nls,ells,cls_dd_11_b,
		  cls_dd_11_b,cls_dd_11_b,cls_dd_11_b,cls_dd_11_b,
		  ell_correct_one,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //NC1-NC2
  compare_cls_arr(cosmo,tr_nc_1,tr_nc_2,nls,ells,cls_dd_12_b,
		  cls_dd_11_b,cls_dd_12_b,cls_dd_12_b,cls_dd_22_b,
		  ell_correct_one,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //NC2-NC2
  compare_cls_arr(cosmo,tr_nc_2,tr_nc_2,nls,ells,cls_dd_22_b,
		  cls_dd_22_b,cls_dd_22_b,cls_dd_22_b,cls_dd_22_b,
		  ell_correct_one,&status);
  if (status) printf("%s\n",cosmo->status_message);

  //NC1-WL1
  compare_cls_arr(cosmo,tr_nc_1,tr_wl_1,nls,ells,cls_dl_11_b,
		  cls_dd_11_b,cls_dl_11_b,cls_dl_11_b,cls_ll_11_b,
		  ell_correct_dl,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //NC1-WL2
  compare_cls_arr(cosmo,tr_nc_1,tr_wl_2,nls,ells,cls_dl_12_b,
		  cls_dd_11_b,cls_dl_12_b,cls_dl_12_b,cls_ll_22_b,
		  ell_correct_dl,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //NC2-WL1
  compare_cls_arr(cosmo,tr_nc_2,tr_wl_1,nls,ells,cls_dl_21_b,
		  cls_dd_22_b,cls_dl_21_b,cls_dl_21_b,cls_ll_11_b,
		  ell_correct_dl,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //NC2-WL2
  compare_cls_arr(cosmo,tr_nc_2,tr_wl_2,nls,ells,cls_dl_22_b,
		  cls_dd_22_b,cls_dl_22_b,cls_dl_22_b,cls_ll_22_b,
		  ell_correct_dl,&status);
  if (status) printf("%s\n",cosmo->status_message);

  //NC1-IA1
  compare_cls_arr(cosmo,tr_nc_1,tr_ia_1,nls,ells,cls_di_11_b,
		  cls_dd_11_b,cls_di_11_b,cls_di_11_b,cls_ii_11_b,
		  ell_correct_dl,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //NC1-IA2
  compare_cls_arr(cosmo,tr_nc_1,tr_ia_2,nls,ells,cls_di_12_b,
		  cls_dd_11_b,cls_di_12_b,cls_di_12_b,cls_ii_22_b,
		  ell_correct_dl,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //NC2-IA1
  compare_cls_arr(cosmo,tr_nc_2,tr_ia_1,nls,ells,cls_di_21_b,
		  cls_dd_22_b,cls_di_21_b,cls_di_21_b,cls_ii_11_b,
		  ell_correct_dl,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //NC2-IA2
  compare_cls_arr(cosmo,tr_nc_2,tr_ia_2,nls,ells,cls_di_22_b,
		  cls_dd_22_b,cls_di_22_b,cls_di_22_b,cls_ii_22_b,
		  ell_correct_dl,&status);
  if (status) printf("%s\n",cosmo->status_message);

  //IA1-WL1
  compare_cls_arr(cosmo,tr_wl_1,tr_ia_1,nls,ells,cls_li_11_b,
  		  cls_ii_11_b,cls_li_11_b,cls_li_11_b,cls_ll_11_b,
		  ell_correct_li,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //IA2-WL2
  compare_cls_arr(cosmo,tr_wl_2,tr_ia_2,nls,ells,cls_li_22_b,
  		  cls_ii_22_b,cls_li_22_b,cls_li_22_b,cls_ll_22_b,
		  ell_correct_li,&status);
  if (status) printf("%s\n",cosmo->status_message);

  //WL1-WL1
  compare_cls_arr(cosmo,tr_wl_1,tr_wl_1,nls,ells,cls_ll_11_b,
		  cls_ll_11_b,cls_ll_11_b,cls_ll_11_b,cls_ll_11_b,
		  ell_correct_ll,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //WL1-WL2
  compare_cls_arr(cosmo,tr_wl_1,tr_wl_2,nls,ells,cls_ll_12_b,
		  cls_ll_11_b,cls_ll_12_b,cls_ll_12_b,cls_ll_22_b,
		  ell_correct_ll,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //WL2-WL2
  compare_cls_arr(cosmo,tr_wl_2,tr_wl_2,nls,ells,cls_ll_22_b,
		  cls_ll_22_b,cls_ll_22_b,cls_ll_22_b,cls_ll_22_b,
		  ell_correct_ll,&status);
  if (status) printf("%s\n",cosmo->status_message);

  //IA1-IA1
  compare_cls_arr(cosmo,tr_ia_1,tr_ia_1,nls,ells,cls_ii_11_b,
		  cls_ii_11_b,cls_ii_11_b,cls_ii_11_b,cls_ii_11_b,
		  ell_correct_ll,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //IA1-IA2
  compare_cls_arr(cosmo,tr_ia_1,tr_ia_2,nls,ells,cls_ii_12_b,
		  cls_ii_11_b,cls_ii_12_b,cls_ii_12_b,cls_ii_22_b,
		  ell_correct_ll,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //IA2-IA2
  compare_cls_arr(cosmo,tr_ia_2,tr_ia_2,nls,ells,cls_ii_22_b,
		  cls_ll_22_b,cls_ll_22_b,cls_ll_22_b,cls_ll_22_b,
		  ell_correct_ll,&status);
  if (status) printf("%s\n",cosmo->status_message);

  //NC1-CMBL
  compare_cls_arr(cosmo,tr_nc_1,tr_cl,nls,ells,cls_dc_1_b,
		  cls_dd_11_b,cls_dc_1_b,cls_dc_1_b,cls_cc_b,
		  ell_correct_one,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //NC2-CMBL
  compare_cls_arr(cosmo,tr_nc_2,tr_cl,nls,ells,cls_dc_2_b,
		  cls_dd_22_b,cls_dc_2_b,cls_dc_2_b,cls_cc_b,
		  ell_correct_one,&status);
  if (status) printf("%s\n",cosmo->status_message);

  //WL1-CMBL
  compare_cls_arr(cosmo,tr_wl_1,tr_cl,nls,ells,cls_lc_1_b,
		  cls_ll_11_b,cls_lc_1_b,cls_lc_1_b,cls_cc_b,
		  ell_correct_lc,&status);
  if (status) printf("%s\n",cosmo->status_message);
  //WL2-CMBL
  compare_cls_arr(cosmo,tr_wl_2,tr_cl,nls,ells,cls_lc_2_b,
		  cls_ll_22_b,cls_lc_2_b,cls_lc_2_b,cls_cc_b,
		  ell_correct_lc,&status);
  if (status) printf("%s\n",cosmo->status_message);

  //CMBL-CMBL
  compare_cls_arr(cosmo,tr_cl,tr_cl,nls,ells,cls_cc_b,
		  cls_cc_b,cls_cc_b,cls_cc_b,cls_cc_b,
		  ell_correct_one,&status);
  if (status) printf("%s\n",cosmo->status_message);

  if(!strcmp(compare_type,"histo")) {
    cosmo->gsl_params.INTEGRATION_EPSREL = epsrel_save;
    cosmo->gsl_params.INTEGRATION_LIMBER_EPSREL = epsrel_save;
    ccl_set_debug_policy(CCL_DEBUG_MODE_WARNING);
  }

  free(cls_dd_11_b); free(cls_dd_12_b); free(cls_dd_22_b);
  free(cls_dl_12_b);free(cls_dl_21_b);free(cls_dl_11_b);free(cls_dl_22_b);
  free(cls_di_12_b);free(cls_di_21_b);free(cls_di_11_b);free(cls_di_22_b);
  free(cls_dc_1_b); free(cls_dc_2_b);
  free(cls_ll_11_b); free(cls_ll_12_b); free(cls_ll_22_b);
  free(cls_li_11_b); free(cls_li_12_b); free(cls_li_22_b);
  free(cls_ii_11_b); free(cls_ii_12_b); free(cls_ii_22_b);
  free(cls_lc_1_b); free(cls_lc_2_b);
  free(cls_cc_b);
  free(ells);
  free(ell_correct_one);
  free(ell_correct_dl);
  free(ell_correct_ll);
  free(ell_correct_lc);
  free(ell_correct_li);

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
  ccl_cl_tracer_free(tr_ia_1);
  ccl_cl_tracer_free(tr_ia_2);
  ccl_cl_tracer_free(tr_cl);
  ccl_cosmology_free(cosmo);
}

CTEST2(cls,analytic) {
  compare_cls("analytic",data);
}

CTEST2(cls,histo) {
  compare_cls("histo",data);
}
