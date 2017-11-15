#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

#define NZ 1024
#define Z0_GC 1.0 
#define SZ_GC 0.02
#define Z0_SH 0.65
#define SZ_SH 0.05
#define LMAX 499

#define CLS_PRECISION 1E-2 // with respect to cosmic variance

CTEST_DATA(angpow) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double Omega_n;
  double Omega_v[5];
  double Omega_k[5];
  double w_0[5];
  double w_a[5];
  
  double z[6];
  double gf[5][6];
};


// Set up the cosmological parameters to be used 
CTEST_SETUP(growth){

  // Values that are the same for all 5 models
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->Omega_n = 0.0;
  data->Omega_v[i] = 0;
  data->w_0[i]     = -1;
  data->w_a[i]     = 0;
  data->Omega_k[i] = 0;
}




static void test_angpow_precision(data)
{
  // Status flag
  int status =0;
  
  // Initialize cosmological parameters

  ccl_configuration ccl_config=default_config;
  ccl_config.transfer_function_method=ccl_boltzmann_class;
  ccl_parameters ccl_params=ccl_parameters_create(data->Omega_c, data->Omega_b, 
						data->Omega_k, data->Omega_n, 
						data->w_0, data->w_a,
						data->h, data->A_s, data->n_s,-1,NULL,NULL);

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
  CCL_ClTracer *clt_gc1=ccl_cl_tracer_number_counts_new(ccl_cosmo,has_rsd,has_magnification,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr,NZ,z_arr_gc,sz_arr, &status);
  CCL_ClTracer *clt_gc2=ccl_cl_tracer_number_counts_new(ccl_cosmo,has_rsd,has_magnification,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr,NZ,z_arr_gc,sz_arr, &status);

  // Workspaces
  double linstep = 40;
  double logstep = 1.15;
  double dchi = (ct_gc_A->chimax-ct_gc_A->chimin)/1000.; // must be below 3 to converge toward limber computation at high ell
  double dlk = 0.003;
  double zmin = 0.05;
  CCL_ClWorkspace *wnl=ccl_cl_workspace_new(NL+1,2*ells[NL-1],CCL_NONLIMBER_METHOD_NATIVE,logstep,linstep,dchi,dlk,zmin,&status);
  CCL_ClWorkspace *wap=ccl_cl_workspace_new(NL+1,2*ells[NL-1],CCL_NONLIMBER_METHOD_ANGPOW,logstep,linstep,dchi,dlk,zmin,&status);

  
  // Compute C_ell
  ccl_angular_cls(cosmo,wnl,ct_gc_B,ct_gc_B,NL,ells,cells_gg_native,&status);
  ccl_angular_cls(cosmo,wap,ct_gc_A,ct_gc_A,NL,ells,cells_gg_angpow,&status);
  double rel_precision = 0.;
  for(int ii=0;ii<NL;ii++) {
    int l = ells[ii]
    double cl_gg_nl=cells_gg_native[ii];
    double cl_gg_ap=cells_gg_angpow[ii];
    double ratio = abs(cl_gg_nl-cl_gg_ap)/cl_gg_nl;
    rel_precision += ratio / sqrt(2./(2*l+1));
  }
  rel_precision /= NL;

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
  ccl_cl_tracer_free(clt_gc1);
  ccl_cl_tracer_free(clt_gc2);
  free(ells);
  free(zarr_1);
  free(zarr_2);
  free(pzarr_1);
  free(pzarr_2);
  free(bzarr);
  ccl_cosmology_free(cosmo);  
}

CTEST1(angpow,precision) {
  angpow_precision(data)
}
