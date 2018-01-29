#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

/*
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
#define PS 0.1 
#define NREL 3.046
#define NMAS 0
#define MNU 0.0
*/
#define ZD 0.5
#define NZ 1024
#define Z0_GC 1.0 
#define SZ_GC 0.02
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 499

#define CLS_PRECISION 1E-2 // with respect to cosmic variance


CTEST_DATA(angpow) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double N_nu_rel;
  double N_nu_mass;
  double mnu;
  double Omega_n;
  double Omega_v;
  double Omega_k;
  double w_0;
  double w_a;
};




// Set up the cosmological parameters to be used 
CTEST_SETUP(angpow){
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->Omega_n = 0.0;
  data->Omega_v = 0;
  data->N_nu_rel=3.046;
  data->N_nu_mass=0;
  data->mnu=0;
  data->w_0     = -1;
  data->w_a    = 0;
  data->Omega_k = 0;
}





static void test_angpow_precision(struct angpow_data * data)
{
  // Status flag
  int status =0;
  
  // Initialize cosmological parameters

  ccl_configuration ccl_config=default_config;
  ccl_config.transfer_function_method=ccl_boltzmann_class;
  ccl_config.matter_power_spectrum_method=ccl_linear;
  //ccl_parameters ccl_params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,NORMPS,NS,-1,NULL,NULL);
  ccl_parameters ccl_params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k, data->N_nu_rel, data->N_nu_mass, data->mnu, data->w_0, data->w_a, data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  //  ccl_parameters ccl_params=ccl_parameters_create(data->Omega_c, data->Omega_b, 						data->Omega_k, data->Omega_n, 						data->w_0, data->w_a,						data->h, data->A_s, data->n_s,-1,NULL,NULL);

  // Initialize cosmology object given cosmo params
  ccl_cosmology *ccl_cosmo=ccl_cosmology_create(ccl_params,ccl_config);

  // Create tracers for angular power spectra
  double z_arr_gc[NZ],nz_arr_gc[NZ],bz_arr[NZ],sz_arr[NZ];
  for(int i=0;i<NZ;i++)
    {
      z_arr_gc[i]=Z0_GC-5*SZ_GC+10*SZ_GC*(i+0.5)/NZ;
      nz_arr_gc[i]=exp(-0.5*pow((z_arr_gc[i]-Z0_GC)/SZ_GC,2));
      bz_arr[i]=1;//+z_arr_gc[i];
      sz_arr[i]=exp(-0.5*pow((z_arr_gc[i]-Z0_GC)/SZ_GC,2));
    }
  
  // Galaxy clustering tracer
  bool has_rsd = true;
  bool has_magnification = false;
  CCL_ClTracer *ct_gc_A=ccl_cl_tracer_number_counts(ccl_cosmo,has_rsd,has_magnification,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr,-1,NULL,NULL, &status);
  CCL_ClTracer *ct_gc_B=ccl_cl_tracer_number_counts(ccl_cosmo,has_rsd,has_magnification,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr,-1,NULL,NULL, &status);
  
  int *ells=malloc(NL*sizeof(int));
  double *cells_gg_angpow=malloc(NL*sizeof(double));
  double *cells_gg_native=malloc(NL*sizeof(double));
  for(int ii=0;ii<NL;ii++)
    ells[ii]=ii;


  // Workspaces
  double linstep = 40;
  double logstep = 1.15;
  double dchi = (ct_gc_A->chimax-ct_gc_A->chimin)/200.; 
  double dlk = 0.003;
  double zmin = 0.05;
  CCL_ClWorkspace *wnl=ccl_cl_workspace_default(NL+1,2*ells[NL-1],CCL_NONLIMBER_METHOD_NATIVE,logstep,linstep,dchi,dlk,zmin,&status);
  CCL_ClWorkspace *wap=ccl_cl_workspace_default(NL+1,2*ells[NL-1],CCL_NONLIMBER_METHOD_ANGPOW,logstep,linstep,dchi,dlk,zmin,&status);

  
  // Compute C_ell
  ccl_angular_cls(ccl_cosmo,wnl,ct_gc_B,ct_gc_B,NL,ells,cells_gg_native,&status);
  ccl_angular_cls(ccl_cosmo,wap,ct_gc_A,ct_gc_A,NL,ells,cells_gg_angpow,&status);
  double rel_precision = 0.;
  for(int ii=2;ii<NL;ii++) {
    int l = ells[ii];
    double cl_gg_nl=cells_gg_native[ii];
    double cl_gg_ap=cells_gg_angpow[ii];
    double ratio = fabs(cl_gg_nl-cl_gg_ap)/cl_gg_nl;
    rel_precision += ratio / sqrt(2./(2*l+1));
    //printf("%d %.3g %.3g %.3g %.3g\n",l,cl_gg_nl,cl_gg_ap,ratio,ratio / sqrt(2./(2*l+1)));
  }
  rel_precision /= NL;
  //printf("precision %.3g\n",rel_precision);

  ASSERT_TRUE((rel_precision < CLS_PRECISION));

  /*
  {// Save the Cls in text file for tests
    Angpow::Parameters para = Angpow::Param::Instance().GetParam();
    std::fstream ofs;
    std::string outName = para.output_dir + para.common_file_tag + "cl.txt";
    ofs.open(outName, std::fstream::out);
    for(int l=0; l<LMAX; l++){
      ofs << std::setprecision(20) << l << " " << spline_eval(l,spl_cl) << std::endl;
    }
    ofs.close();
  }
  */
  
  /* {//save ctheta */

  /*   Angpow::CTheta ct(clout,para.apod); */

  /*   std::fstream ofs; */
  /*   std::string outName = para.output_dir + para.common_file_tag + "ctheta.txt"; */
  /*   //define theta values */
  /*   const int Npts=100; */
  /*   const double theta_max=para.theta_max*M_PI/180; */
  /*   double step=theta_max/(Npts-1); */
      
  /*   ofs.open(outName, std::fstream::out); */
  /*   for (size_t i=0;i<Npts;i++){ */
  /*     double t=i*step; */
  /*     ofs << std::setprecision(20) << t << " " << ct(t) << std::endl; */
  /*   } */
  /*   ofs.close(); */
    
  /*   outName = para.output_dir + para.common_file_tag + "apod_cl.txt"; */
  /*   ct.WriteApodCls(outName); */
  /* } */
  
  
  //Free up tracers
  ccl_cl_tracer_free(ct_gc_A);
  ccl_cl_tracer_free(ct_gc_B);
  free(ells);
  ccl_cl_workspace_free(wap);
  ccl_cl_workspace_free(wnl);
  free(cells_gg_angpow);
  free(cells_gg_native);
 /*
  free(zarr_1);
  free(zarr_2);
  free(pzarr_1);
  free(pzarr_2);
  free(bzarr);
  */
  ccl_cosmology_free(ccl_cosmo);  
}

CTEST2(angpow,precision) {
  test_angpow_precision(data);
}
