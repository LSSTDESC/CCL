#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"

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

int main(int argc, char * argv[])
{
  double Omega_c = 0.25;
  double Omega_b = 0.05;
  double h = 0.7;
  double A_s = 2.1e-9;
  double n_s = 0.96;
  ccl_configuration config = default_config;
  config.transfer_function_method = ccl_bbks;
  ccl_parameters params = ccl_parameters_create_flat_lcdm(Omega_c, Omega_b, h, A_s, n_s);
  params.sigma_8=0.8;
  ccl_cosmology * cosmo = ccl_cosmology_create(params, config);
  
  //  ccl_cosmology_compute_distances(cosmo,&status);
  //  ccl_cosmology_compute_growth(cosmo,&status);
  //  ccl_cosmology_compute_power(cosmo, &status);

  //Create arrays for N(z)
  int nz=256;
  double zmean=1.0,sigz=0.1;
  double *zarr,*pzarr,*bzarr;
  zarr=malloc(nz*sizeof(double));
  pzarr=malloc(nz*sizeof(double));
  bzarr=malloc(nz*sizeof(double));
  for(int ii=0;ii<nz;ii++) {
    double z=zmean-5*sigz+10*sigz*(ii+0.5)/nz;
    double pz=exp(-0.5*((z-zmean)*(z-zmean)/(sigz*sigz)));
    zarr[ii]=z;
    pzarr[ii]=pz;
    bzarr[ii]=1.;
  }

  //Create tracers
  ClTracer *tr_nc=ccl_tracer_new(cosmo,CL_TRACER_NC,nz,zarr,pzarr,nz,zarr,bzarr); //Number counts tracer
  ClTracer *tr_wl=ccl_tracer_new(cosmo,CL_TRACER_WL,nz,zarr,pzarr,nz,NULL,NULL);  //Lensing tracer

  FILE *fi=fopen("lj_test_cl_dd.txt","r");
  int nl=linecount(fi); rewind(fi);
  for(int ii=0;ii<nl;ii++) {
    int l;
    double cl_lj,cl_ccl;
    int stat=fscanf(fi,"%d %lf",&l,&cl_lj);
    if(stat!=2) {
      fprintf(stderr,"error reading file\n");
      exit(1);
    }
    cl_ccl=ccl_angular_cl(cosmo,l,tr_nc,tr_nc);
    if(l>0) {
      if(fabs(cl_ccl/cl_lj-1)>1E-4)
	printf("%d %lE\n",l,cl_ccl/cl_lj-1);
    }
  }
  fclose(fi);

  fi=fopen("lj_test_cl_d1l2.txt","r");
  nl=linecount(fi); rewind(fi);
  for(int ii=0;ii<nl;ii++) {
    int l;
    double cl_lj,cl_ccl;
    int stat=fscanf(fi,"%d %lf",&l,&cl_lj);
    if(stat!=2) {
      fprintf(stderr,"error reading file\n");
      exit(1);
    }
    cl_ccl=ccl_angular_cl(cosmo,l,tr_nc,tr_wl);
    if(l>0) {
      if(fabs(cl_ccl/cl_lj-1)>1E-4)
	printf("%d %lE\n",l,cl_ccl/cl_lj-1);
    }
  }
  fclose(fi);

  fi=fopen("lj_test_cl_ll.txt","r");
  nl=linecount(fi); rewind(fi);
  for(int ii=0;ii<nl;ii++) {
    int l;
    double cl_lj,cl_ccl;
    int stat=fscanf(fi,"%d %lf",&l,&cl_lj);
    if(stat!=2) {
      fprintf(stderr,"error reading file\n");
      exit(1);
    }
    cl_ccl=ccl_angular_cl(cosmo,l,tr_wl,tr_wl);
    if(l>0) {
      if(fabs(cl_ccl/cl_lj-1)>1E-4)
	printf("%d %lE\n",l,cl_ccl/cl_lj-1);
    }
  }
  fclose(fi);

  ccl_tracer_free(tr_nc);
  ccl_tracer_free(tr_wl);
  free(zarr);
  free(pzarr);
  free(bzarr);

  //This checks that a(chi(a))=a
  for(double a=0.2;a<1;a+=0.02) {
    double chi=ccl_comoving_radial_distance(cosmo,a);
    double a_rec=ccl_scale_factor_of_chi(cosmo,chi);
    if(fabs(a_rec-a)>1E-6) {
      printf("FAILED\n");
      exit(1);
    }
  }
  printf("SUCCESS\n");

  return 0;
}
