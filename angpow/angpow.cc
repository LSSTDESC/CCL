#include <iostream>

#include "Angpow/walltimer.h"          //profiling 

#include "Angpow/angpow_exceptions.h"  //exceptions
#include "Angpow/angpow_numbers.h"     //r_8... def

#include "Angpow/angpow_parameters.h"  //control parameters 

#include "Angpow/angpow_powspec.h"     //power spectrum IMPLEMENTATION
#include "Angpow/angpow_integrand.h"   //f_ell(k,z) integrand functions
#include "Angpow/angpow_radial.h"      //radial window
#include "Angpow/angpow_utils.h"       //utility functions
#include "Angpow/angpow_pk2cl.h"       //utility class that produce the Cl (auto/cross corr)
#include "Angpow/angpow_clbase.h"      //class to manage Clbase structure (ell, val)
#include "Angpow/angpow_ctheta.h"      //class function to manage CTheta values
#include "Angpow/angpow_tools.h"

#ifdef _OPENMP
#include<omp.h>
#endif


//------------------------------------------------------

namespace Angpow {


//------------------------------
// Exemple of processing from P(k) to Cl
//------------------------------
void process() {
  
  //Get the pointer to the job processing user parameters
  Parameters para = Param::Instance().GetParam();
  
  int Lmax = para.Lmax; //ell in [0, Lmax-1]

  tstack_push("Processing....");

  //Radial (redshift) selection windows
  RadSelectBase* Z1win = 0;
  switch(para.wtype1) {
  case Parameters::GaussGal:
    {
      r_8 zmean  = para.mean1;
      r_8 zsigma = para.width1;
      r_8 zmin   = zmean - para.n_sigma_cut * zsigma;
      r_8 zmax   = zmean + para.n_sigma_cut * zsigma;
      std::cout << "Galaxy density + Gaussian window Z1: mean/sigma/min/max: " 
		<< zmean << "/"
		<< zsigma << "/"
		<< zmin << "/"
		<< zmax
		<< std::endl;
      Z1win = new RadModifiedGaussSelect(zmean, zsigma, zmin, zmax);
      break;
    }
  case Parameters::Gauss:
    {
      r_8 zmean  = para.mean1;
      r_8 zsigma = para.width1;
      r_8 zmin   = zmean - para.n_sigma_cut * zsigma;
      r_8 zmax   = zmean + para.n_sigma_cut * zsigma;
      std::cout << "Gaussian window Z1: mean/sigma/min/max: " 
		<< zmean << "/"
		<< zsigma << "/"
		<< zmin << "/"
		<< zmax
		<< std::endl;
      Z1win = new RadGaussSelect(zmean, zsigma, zmin, zmax);      
      break;
    }
  case Parameters::Dirac:
    {
      r_8 zmin   = 0; //not used
      r_8 zmax   = 0; //not used
      r_8 zmean  = para.mean1;
      std::cout << "Dirac window Z1: mean: " << zmean << std::endl;
      Z1win = new DiracSelect(zmean, zmin, zmax);
      break;
    }
  case Parameters::TopHat:
    {
      r_8 zmean  = para.mean1;
      r_8 zwidth = para.width1; //note 1/2 full width
      r_8 smooth_edges = para.smooth_edges;
      r_8 nSigmaCut = para.n_sigma_cut;
      r_8 zmin = zmean - (1. + nSigmaCut*smooth_edges) * zwidth;
      r_8 zmax = zmean + (1. + nSigmaCut*smooth_edges) * zwidth;
      std::cout << "TopHas window Z1: mean, width, min, max: " <<
	zmean << " " << zwidth << " " << zmin << " " << zmax << std::endl;
      Z1win = new RadTopHatSmoothSelect(zmin,zmax,zmean,zwidth,smooth_edges);
      break;
    }
  default:
    throw AngpowError("Unknown Select 1 type");
    break;
  }//sw
  if(Z1win == 0) throw AngpowError("process: FATAL ERROR: unknown Z1 selection function");

  RadSelectBase* Z2win = 0;
  switch(para.wtype2) {
  case Parameters::GaussGal:
    {
      r_8 zmean  = para.mean2;
      r_8 zsigma = para.width2;
      r_8 zmin   = zmean - para.n_sigma_cut * zsigma;
      r_8 zmax   = zmean + para.n_sigma_cut * zsigma;
      std::cout << "Galaxy density + Gaussian window Z2: mean/sigma/min/max: " 
		<< zmean << "/"
		<< zsigma << "/"
		<< zmin << "/"
		<< zmax
		<< std::endl;
      Z2win = new RadModifiedGaussSelect(zmean, zsigma, zmin, zmax);
      break;
    }
  case Parameters::Gauss:
    {
      r_8 zmean  = para.mean2;
      r_8 zsigma = para.width2;
      r_8 zmin   = zmean - para.n_sigma_cut * zsigma;
      r_8 zmax   = zmean + para.n_sigma_cut * zsigma;
      std::cout << "Gaussian window Z2: mean/sigma/min/max: " 
		<< zmean << "/"
		<< zsigma << "/"
		<< zmin << "/"
		<< zmax
		<< std::endl;
      Z2win = new RadGaussSelect(zmean, zsigma, zmin, zmax);      
      break;
    }
  case Parameters::Dirac:
    {
      r_8 zmin   = 0; //not used
      r_8 zmax   = 0; //not used
      r_8 zmean  = para.mean2;
      std::cout << "Dirac window Z2: mean: " << zmean << std::endl;
      Z2win = new DiracSelect(zmean, zmin, zmax);
      break;
    }
  case Parameters::TopHat:
    {
      r_8 zmean  = para.mean2;
      r_8 zwidth = para.width2; //note 1/2 full width
      r_8 smooth_edges = para.smooth_edges;
      r_8 nSigmaCut = para.n_sigma_cut;
      r_8 zmin = zmean - (1. + nSigmaCut*smooth_edges) * zwidth;
      r_8 zmax = zmean + (1. + nSigmaCut*smooth_edges) * zwidth;
      std::cout << "TopHas window Z2: mean, width, min, max: " <<
	zmean << " " << zwidth << " " << zmin << " " << zmax << std::endl;
      Z2win = new RadTopHatSmoothSelect(zmin,zmax,zmean,zwidth,smooth_edges);
    }
    break;
  default:
    throw AngpowError("Unknown Select 2 type");
    break;
  }//sw
  if(Z2win == 0) throw AngpowError("process: FATAL ERROR: unknown Z2 selection function");


  
  //The cosmological distance tool 
  CosmoCoord cosmo(para.cosmo_zmin, para.cosmo_zmax, para.cosmo_npts, para.cosmo_precision);
  

  //Usage of an external file of P(k) and use of an internal Growth function
  //If k unit is h/Mpc and P(k) unit is [Mpc/h]^3 then set  use_h_rescaling = true
  //If the P(k) is defined at zref=0 then no growth rescaling is needed. 
  //
  std::string pwName = para.power_spectrum_input_dir + para.power_spectrum_input_file;
  r_8 pw_kmin = para.pw_kmin;
  r_8 pw_kmax = para.pw_kmax;
  r_8 zref = 0.;
  bool use_h_rescaling = true;
  bool use_growth_rescaling= false;
  PowerSpecFile pws(cosmo,pwName,zref,pw_kmin,pw_kmax,use_h_rescaling,use_growth_rescaling);

  
  // JN 20/04/2017
  r_8 bias = para.bias;
  bool include_rsd = para.include_rsd;
  Integrand int1(&pws, &cosmo, bias, include_rsd);
  Integrand int2(&pws, &cosmo, bias, include_rsd);

  //Initialize the Cl with parameters to select the ell set which is interpolated after the processing
  Clbase clout(Lmax,para.linearStep, para.logStep);

  //Main class
  Pk2Cl pk2cl; //Default: the user parameters are used in the Constructor 
  pk2cl.PrintParam();
  pk2cl.Compute(int1, int2, cosmo, Z1win, Z2win, Lmax, clout);
  //pk2cl.Compute(pws, cosmo, Z1win, Z2win, Lmax, clout);


  tstack_pop("Processing....");
  tstack_report("Processing....");


  
  {//save the Cls
    std::fstream ofs;
    std::string outName = para.output_dir + para.common_file_tag + "cl.txt";
    ofs.open(outName, std::fstream::out);
    for(int index_l=0;index_l<clout.Size();index_l++){
      ofs << setprecision(20) << clout[index_l].first << " " << clout[index_l].second << endl;
    }
    ofs.close();
  }
  
  {//save ctheta

    CTheta ct(clout,para.apod);

    std::fstream ofs;
    std::string outName = para.output_dir + para.common_file_tag + "ctheta.txt";
    //define theta values
    const int Npts=100;
    const double theta_max=para.theta_max*M_PI/180;
    double step=theta_max/(Npts-1);
      
    ofs.open(outName, std::fstream::out);
    for (size_t i=0;i<Npts;i++){
      double t=i*step;
      ofs << setprecision(20) << t << " " << ct(t) << endl;
    }
    ofs.close();
    
    outName = para.output_dir + para.common_file_tag + "apod_cl.txt";
    ct.WriteApodCls(outName);
}


  //clear
  pws.ExplicitDestroy();


  if(Z1win) delete Z1win; Z1win = 0;
  if(Z2win) delete Z2win; Z2win = 0;

  std::cout << "End process......" << std::endl;


}//process



}//namespace


//------------------------------------------------------

int main(int narg, char* argv[]) {


  using namespace Angpow; 
  //Default return code
  int rc = 0;

  try {

    //Default Initialization 
    if(narg<2) {
      throw AngpowError("usage: angpow <init-file>");
    }
    std::string fileIni = argv[1];
  
#ifdef _OPENMP
    cout << "using OpenMP with max threads=" << omp_get_max_threads() << endl;
    cout << "dynamic adjustment is: "<< omp_get_dynamic()<<endl;
#endif    
    
    Param::Instance().ReadParam(fileIni);
    {
      Parameters para = Param::Instance().GetParam();
      std::fstream ofs;
      std::string outName = para.output_dir + para.common_file_tag + "used-param.txt";
      ofs.open(outName, std::fstream::out);
      Param::Instance().WriteParam(ofs);
    }
    
    
    //Go to processing
    process();
  }
  
  catch (std::exception& sex) {
    cerr << "\n angpow.cc std::exception :"  << (string)typeid(sex).name() 
         << "\n msg= " << sex.what() << endl;
    rc = 78;
  }
  catch ( string str ) {
    cerr << "angpow.cc Exception raised: " << str << endl;
  }
  catch (...) {
    cerr << " angpow.cc catched unknown (...) exception  " << endl; 
    rc = 79; 
  } 

  cout << ">>>> angpow.cc ------- END ----------- RC=" << rc << endl;
  return rc;
      
}//main




