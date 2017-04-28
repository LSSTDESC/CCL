#include <iostream>

#include "Angpow/walltimer.h"          //profiling 

#include "Angpow/angpow_exceptions.h"  //exceptions
#include "Angpow/angpow_numbers.h"     //r_8... def

#include "Angpow/angpow_parameters.h"  //control parameters 

#include "Angpow/angpow_powspec.h"     //power spectrum IMPLEMENTATION
#include "Angpow/angpow_radial.h"      //radial window
#include "Angpow/angpow_utils.h"       //utility functions
#include "Angpow/angpow_pk2cl.h"       //utility class that produce the Cl (auto/cross corr)
#include "Angpow/angpow_clbase.h"      //class to manage Clbase structure (ell, val)

#include "Angpow/angpow_quadinteg.h"   //Quuadrature

#ifdef _OPENMP
#include<omp.h>
#endif


//------------------------------------------------------

namespace Angpow {


//------------------------------
// Exemple of processing from P(k) to Cl
//------------------------------

//Experimental Limber approx
class LimberApp : public ClassFunc1D {
public:
  LimberApp(int ell, PowerSpecBase* pws, CosmoCoord* cosmo, 
	    RadSelectBase* W1, RadSelectBase* W2): 
    ell_(ell), pws_(pws), cosmo_(cosmo), W1_(W1), W2_(W2) {
    
    norm_ = 2./(2.*ell+1.);
  }
  virtual ~LimberApp() {}
  virtual r_8 operator()(r_8 k) const {
    r_8 zlk = cosmo_->z((ell_+0.5)/k);
    
    //Verif r(z) <-> z(r) OK pass 18/11/16    
//     std::cout << std::setprecision(20) << k << " " << (ell_+0.5) << " " << zlk << " " << W1_->operator()(zlk) << " " << W2_->operator()(zlk)<< std::endl; 

    r_8 Ezlk = cosmo_->EzMpcm1(zlk);
    pws_->Init(ell_,zlk); //JEC 12/12/16
    return norm_
      * (W1_->operator()(zlk)) 
      * (W2_->operator()(zlk)) 
      * Ezlk*Ezlk 
      * (pws_->operator()(ell_,k,zlk));
  }
private:
  int ell_;
  PowerSpecBase* pws_;
  CosmoCoord* cosmo_;
  RadSelectBase* W1_;
  RadSelectBase* W2_;
  r_8 norm_;
};


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
  
  std::cout << "Hubble Length (Mpc) = " << cosmo.Ez(1.)/cosmo.EzMpcm1(1.) << std::endl;


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
  PowerSpecFile pws(cosmo,pwName,zref,pw_kmin,pw_kmax,
		    use_h_rescaling,
		    use_growth_rescaling);
  
  //Initialize the Cl with parameters to select the ell set which is interpolated after the processing
  std::cout << "debug: Lmax, linear, log: " << Lmax << ", " << para.linearStep<< ", "<< para.logStep
	    << std::endl;

  Clbase clout(Lmax,para.linearStep, para.logStep);

  r_8 Rmax=1.;
  {
    std::vector<r_8> ztmp(2);
    ztmp[0] = Z1win->GetZMax();
    ztmp[1] = Z2win->GetZMax();
    std::vector<r_8> rtmp;
    RedShift2Radius(ztmp,rtmp,cosmo);
    Rmax = rtmp[0]<rtmp[1] ? rtmp[1] : rtmp[0];
  }
  

  //Experiment Limber computation START
  {
    
    typedef GaussKronrodQuadrature<r_8> Integrator_t;
    Integrator_t Integrator(40,para.quadrature_rule_ios_dir+"/CosmoRuleData.txt",true);
    r_8 precision = 1e-10;

    
    //Normalization of selection functions
    r_8 normW1 = 1.;
    {
      Integrator.SetFuncBounded(*Z1win,Z1win->GetZMin(),Z1win->GetZMax());      
      Quadrature<r_8,Integrator_t>::values_t integ_val 
	= Integrator.GlobalAdapStrat(precision);
      normW1 = integ_val.first;
    }

    r_8 normW2 = 1.;
    {
      Integrator.SetFuncBounded(*Z2win,Z2win->GetZMin(),Z2win->GetZMax());      
      Quadrature<r_8,Integrator_t>::values_t integ_val 
	= Integrator.GlobalAdapStrat(precision);
      normW2 = integ_val.first;
    }
    
    std::cout << "norm W1, W2 " << normW1 << ", " << normW2 << std::endl;
    

    for(int index_l=0; index_l<clout.Size(); index_l++){
      
      int l=clout[index_l].first;
      
      //      r_8 lowKBnd = (BesselJImp::Xmin(l))/Rmax; 
      r_8 lowKBnd = 0.;
      r_8 highKBnd =  para.cl_kmax;
    
      std::cout << "ell, kmin, kmax: " << l << ", "<< lowKBnd << ", " << highKBnd << std::endl;

      LimberApp f(l,&pws,&cosmo,Z1win,Z2win);
      Integrator.SetFuncBounded(f,lowKBnd,highKBnd);
      Quadrature<r_8,Integrator_t>::values_t integ_val 
	= Integrator.GlobalAdapStrat(precision);

      std::cout << "l, val " << l << " " << integ_val.first << std::endl;

      clout[index_l].second = integ_val.first/(normW1*normW2);

      

    }//l-loop
  
  }//Experiment Limber computation END

  //interpolation
  clout.Interpolate();
  
  {//save the Cls
    std::fstream ofs;
    std::string outName = para.output_dir + para.common_file_tag + "cl.txt";
    ofs.open(outName, std::fstream::out);
    for(int index_l=0;index_l<clout.Size();index_l++){
      ofs << setprecision(20) << clout[index_l].first << " " << clout[index_l].second << endl;
    }
    ofs.close();
  }

  tstack_pop("Processing....");
  tstack_report("Processing....");

  //clear
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




