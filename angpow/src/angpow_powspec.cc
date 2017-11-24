#include "Angpow/angpow_powspec.h"

namespace Angpow {

////////////
//Growth Factor from SimLSS (Magneville & Ansari)
////////////

// From Eisenstein & Hu ApJ 496:605-614 1998 April 1
// Pour avoir D(z) = 1/(1+z) faire: OmegaMatter0=1 OmegaLambda0=0
  GrowthEisenstein::GrowthEisenstein(double OmegaMatter0,double OmegaLambda0)
    
    : O0_(OmegaMatter0) , Ol_(OmegaLambda0) {
  if(OmegaMatter0==0.) {
    std::cout<<"GrowthEisenstein::GrowthEisenstein_Error: bad OmegaMatter0  value : "
	     <<OmegaMatter0<<std::endl;
    throw AngpowError("GrowthEisenstein::GrowthEisenstein_Error:  bad OmegaMatter0  value");
  }
  
  Ok_ = 1. - O0_ - Ol_;

  // Calcul de la normalisation (pour z=0 -> growth=1.)
  double D1z0 = pow(O0_,4./7.) - Ol_ + (1.+O0_/2.)*(1.+Ol_/70.);
  D1z0 = 2.5*O0_ / D1z0;
  invD1z0_ = 1./D1z0;
}//Ctor



void GrowthEisenstein::SetParTo(double OmegaMatter0,double OmegaLambda0)
{
 if(OmegaMatter0>0.) O0_ = OmegaMatter0;
 Ol_ = OmegaLambda0;
}

bool GrowthEisenstein::SetParTo(double OmegaMatter0)
// idem precedent sans changer OmegaLambda0
{
 if(OmegaMatter0<=0.) return false;
 O0_ = OmegaMatter0;
 return true;
}

//////////////////////////////////////////////////////////////////

PowerSpecFile::PowerSpecFile(const CosmoCoord& su, std::string inpkname, r_8 zref, 
			     r_8 kmin, r_8 kmax, 
			     bool h_rescale, 
			     bool growth_rescale) {
  //minimal Cosmology
  r_8 h100 = su.h();
  r_8 Om0 = su.OmegaMatter();
  r_8 Ol0 = su.OmegaLambda();
  mypGE_ = new GrowthEisenstein(Om0,Ol0); 
  cout<<"GrowthFactor: D1(0)="<<(*mypGE_)(0.)<<" D1(z=1)= "<<(*mypGE_)(1.) <<" D1("<<zref<<") = "<<(*mypGE_)(zref)<<endl;
  
  cout << "Reading input Pk from file " << inpkname << " at zref="<<zref<< endl;

  std::vector<r_8> ks;
  std::vector<r_8> Pks;

  std::ifstream ifs(inpkname.c_str(), std::ifstream::in);
  if (!ifs) throw AngpowError("Fail reading "+inpkname);
 

  std::string line;
  int iline=1;
  while(std::getline(ifs, line)){
    if (line.find("#",0) != string::npos) { //# as comments
    }else{
      std::stringstream ss(line);
      double k;
      double Pk;
      ss >> k >> Pk;
      ks.push_back(k);
      Pks.push_back(Pk);
      iline++;
    }//look for header
  }//while

  ifs.close();

  std::vector<r_8> vx;
  std::vector<r_8> vy;


  for(int i=0;i<(int)ks.size();i++){
    if(h_rescale) {
      ks[i] *= h100; 
      Pks[i] /= h100*h100*h100; //JEC 15/9/16 in case k in h/Mpc & Pk in (Mpc/h)^3
    }
    if(growth_rescale){
       r_8 Dz = (*mypGE_)(zref);
       Pks[i] /= Dz*Dz; //if zref =/= 0 rescale P(k) such that P(k,zref) = P(k,z=0)*D^2(zref) (D(0)=1 by default)
    }
    if(ks[i]<kmin||ks[i]>kmax) continue;
    vx.push_back(ks[i]);
    vy.push_back(Pks[i]);
  }
  //JEC 20/10/16 choose regularly spaced grid (faster interpolation) START
  r_8 vxmin = *(std::min_element(vx.begin(), vx.end()));
  r_8 vxmax = *(std::max_element(vx.begin(), vx.end()));
  //  std::cout << "PowerSpecFile: Pk interpol vxmin/max: " << vxmin<<", "<<vxmax << ", " << 10*vx.size() <<std::endl;
//   Pk_ = new SLinInterp1D();
//   Pk_->DefinePoints(vx,vy,vxmin,vxmax,100*vx.size());
  Pk_ = new SLinInterp1D(vx,vy,vxmin,vxmax,0); //JEC 15/2/17

  //Pk_.DefinePoints(vx,vy);
  //-----------------------------------------  END
  //  npt_ = ks.size(); lkmin = log10(kmin); lkmax = log10(kmax); dlk_=(lkmax-lkmin)/npt_;

  //UN USED JEC 14/2/17  
//   r_8 pmin = *(std::min_element(Pks.begin(), Pks.end()));
//   r_8 pmax = *(std::max_element(Pks.begin(), Pks.end()));

 
}//Ctor



}//namespace
