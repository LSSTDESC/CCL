#include "ccl_angpow_interface.h"

#define OC 0.2582
#define OB 0.0483
#define OK 0.00
#define ON 0.00
#define HH 0.679
#define W0 -1.0
#define WA 0.00
#define NS 1.0
#define NORMPS 2.215e-9
#define ZD 0.5
#define NZ 128
#define Z0_GC 1.0 
#define SZ_GC 0.02
#define Z0_SH 0.65
#define SZ_SH 0.05
#define LMAX 500




//------------------------------
// Exemple of processing from P(k) to Cl
//------------------------------
int main(int argc,char **argv){

  int rc=0;
  try {
  // Status flag
  int status =0;
  
  // Initialize cosmological parameters

  ccl_configuration ccl_config=default_config;
  ccl_config.transfer_function_method=ccl_boltzmann_class;
  ccl_parameters ccl_params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,NORMPS,NS,-1,NULL,NULL);

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

  // Compute C_ell
  SplPar * spl_cl = ccl_angular_cls_angpow(ccl_cosmo, LMAX, clt_gc1, clt_gc2, &status);
  
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


  }//try
    catch (std::exception& sex) {
      std::cerr << "\n job std::exception :"  << (std::string)typeid(sex).name() 
         << "\n msg= " << sex.what() << std::endl;
    rc = 78;
  }
    catch ( std::string str ) {
    std::cerr << "job Exception raised: " << str << std::endl;
  }
  catch (...) {
    std::cerr << " job catched unknown (...) exception  " << std::endl; 
    rc = 79; 
  } 

  std::cout << ">>>> job ------- END ----------- RC=" << rc << std::endl;
  return rc;
  
}
