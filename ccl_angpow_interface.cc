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
#define Z0_GC 0.50
#define SZ_GC 0.05
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 512

namespace Angpow {

void init_windows(Parameters para, RadSelectBase*& Z1win, RadSelectBase*& Z2win) {
  
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

}

}//namespace



//------------------------------
// Exemple of processing from P(k) to Cl
//------------------------------
int main(int argc,char **argv){
  //status flag
  //int status =0;
  // Initialize cosmological parameters

  ccl_configuration ccl_config=default_config;
  ccl_config.transfer_function_method=ccl_boltzmann_class;
  ccl_parameters ccl_params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,NORMPS,NS,-1,NULL,NULL);

  // Initialize cosmology object given cosmo params
  ccl_cosmology *ccl_cosmo=ccl_cosmology_create(ccl_params,ccl_config);
  
  // Initialize the Angpow parameters
  //Angpow::Param::Instance().SetToDefault();
  Angpow::Parameters para = Angpow::Param::Instance().GetParam();
  para.wtype1 = Angpow::Parameters::Dirac; para.wtype2 = Angpow::Parameters::Dirac;
  para.mean1 = 1.0; para.mean2 = 1.0;
  para.cosmo_zmax = 9.0;
  para.cosmo_npts = 1000;
  para.chebyshev_order_1 = 9;
  para.chebyshev_order_2 = 9;

  // Initialize the radial selection windows
  Angpow::RadSelectBase* Z1win = 0;
  Angpow::RadSelectBase* Z2win = 0;
  Angpow::init_windows(para, Z1win, Z2win);

  //The cosmological distance tool 
  Angpow::CosmoCoordCCL cosmo(ccl_cosmo, para.cosmo_zmin, para.cosmo_zmax, para.cosmo_npts); //, para.cosmo_precision);

  // Define Pk
  double kmin = 1e-5; //para.pw_kmin;
  double kmax = 10; //para.pw_kmax;
  int nk = 1000;
  Angpow::PowerSpecCCL pws(ccl_cosmo, kmin, kmax, nk);

  //Initialize the Cl with parameters to select the ell set which is interpolated after the processing
  int Lmax = 500; //para.Lmax; //ell in [0, Lmax-1]
  Angpow::Clbase clout(Lmax,para.linearStep, para.logStep);

  //Main class
  Angpow::Pk2Cl pk2cl; //Default: the user parameters are used in the Constructor 
  pk2cl.PrintParam();
  //  pk2cl.Compute(pws, Z1win, Z2win, Lmax, clout);
  pk2cl.Compute(pws, cosmo, Z1win, Z2win, Lmax, clout);

  {//save the Cls
    std::fstream ofs;
    std::string outName = para.output_dir + para.common_file_tag + "cl.txt";
    ofs.open(outName, std::fstream::out);
    for(int index_l=0;index_l<clout.Size();index_l++){
      ofs << std::setprecision(20) << clout[index_l].first << " " << clout[index_l].second << std::endl;
    }
    ofs.close();
  }
  
  {//save ctheta

    Angpow::CTheta ct(clout,para.apod);

    std::fstream ofs;
    std::string outName = para.output_dir + para.common_file_tag + "ctheta.txt";
    //define theta values
    const int Npts=100;
    const double theta_max=para.theta_max*M_PI/180;
    double step=theta_max/(Npts-1);
      
    ofs.open(outName, std::fstream::out);
    for (size_t i=0;i<Npts;i++){
      double t=i*step;
      ofs << std::setprecision(20) << t << " " << ct(t) << std::endl;
    }
    ofs.close();
    
    outName = para.output_dir + para.common_file_tag + "apod_cl.txt";
    ct.WriteApodCls(outName);
  }

  
}
