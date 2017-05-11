#include <iostream>
#include <fstream> 
#include <iomanip>  
#include <iterator>

#include "Angpow/angpow_pk2cl.h"
#include "Angpow/angpow_cosmo_base.h"
#include "Angpow/angpow_exceptions.h"
#include "Angpow/angpow_parameters.h"
#include "Angpow/angpow_radint.h" 
#include "Angpow/angpow_powspec_base.h"
#include "Angpow/angpow_integrand_base.h"
#include "Angpow/angpow_kinteg.h"
#include "Angpow/angpow_clbase.h"

namespace Angpow {

//Ctor
 Pk2Cl::Pk2Cl(){

   Parameters para = Param::Instance().GetParam();
   nRadOrder1_ = para.radial_order_1;
   nRadOrder2_ = para.radial_order_2;
   iOrdFunc1_  = para.chebyshev_order_1;
   iOrdFunc2_  = para.chebyshev_order_2;
   nRootPerInt_ = para.n_bessel_roots_per_interval;
   kMax_ = para.cl_kmax;

}//Ctor

void Pk2Cl::PrintParam(){
  using namespace std;
  cout << "Pk2Cl parameters ...." << endl; 
  cout << "  radial quadrature order z1: " << nRadOrder1_ << endl;
  cout << "  radial quadrature order z2: " << nRadOrder2_ << endl;
  cout << "  kMax : " <<  kMax_ << endl;
  cout << "  order of Chebyshev transform for 1st funct : " <<  iOrdFunc1_  << endl;
  cout << "  order of Chebyshev transform for 2nd funct : " <<  iOrdFunc2_  << endl;
  cout << "  number of Bessel roots per k-subintegrals: " <<  nRootPerInt_ << endl;
  
}//PrintParam



//-----------
//Cross correlation P(k) => Cl(z1,z2) with simple optimized cartesian sampling
//-----------
void Pk2Cl::Compute(PowerSpecBase& pws, CosmoCoordBase& coscoord,
		    RadSelectBase* Z1win, RadSelectBase* Z2win, int Lmax, 
		    Clbase& clout){

  


  Radial2DIntegrator theRadInt(nRadOrder1_,nRadOrder2_);
  theRadInt.SetQuadrature(Z1win, Z2win);

  int NRvalI = theRadInt.GetNRvalI();
  int NRvalJ = theRadInt.GetNRvalJ();

  std::vector<r_8>winW;
  theRadInt.ComputeSuperW(winW);


  std::vector<r_8>zInodes(theRadInt.GetZInodes());
  std::vector<r_8>zJnodes(theRadInt.GetZJnodes());
  std::vector<r_8>rInodes;
  RedShift2Radius(zInodes, rInodes, coscoord);
  std::vector<r_8>rJnodes;
  RedShift2Radius(zJnodes, rJnodes, coscoord);
  
    
  KIntegrator kinteg(rInodes, rJnodes, 
		     zInodes,zJnodes, 
		     winW,
		     NRvalI, NRvalJ, Lmax,
		     iOrdFunc1_, iOrdFunc2_, 
		     nRootPerInt_, kMax_);  

  kinteg.Compute(pws, clout);
  
  auto init=clout.InitialElls();
  std::cout << init.size() << " Cl values computed at l=" ;
  std::copy(init.begin(),init.end(),std::ostream_iterator<int>(std::cout,"/"));
  std::cout<< std::endl;
  clout.Interpolate();
}



//-----------
//Cross correlation P(k) => Cl(z1,z2) with simple optimized cartesian sampling
//-----------
void Pk2Cl::Compute(IntegrandBase& int1, IntegrandBase& int2,
		    CosmoCoordBase& coscoord,
		    RadSelectBase* Z1win, RadSelectBase* Z2win, int Lmax, 
		    Clbase& clout){

  


  Radial2DIntegrator theRadInt(nRadOrder1_,nRadOrder2_);
  theRadInt.SetQuadrature(Z1win, Z2win);

  int NRvalI = theRadInt.GetNRvalI();
  int NRvalJ = theRadInt.GetNRvalJ();

  std::vector<r_8>winW;
  theRadInt.ComputeSuperW(winW);


  std::vector<r_8>zInodes(theRadInt.GetZInodes());
  std::vector<r_8>zJnodes(theRadInt.GetZJnodes());
  std::vector<r_8>rInodes;
  RedShift2Radius(zInodes, rInodes, coscoord);
  std::vector<r_8>rJnodes;
  RedShift2Radius(zJnodes, rJnodes, coscoord);
  
    
  KIntegrator kinteg(rInodes, rJnodes, 
		     zInodes,zJnodes, 
		     winW,
		     NRvalI, NRvalJ, Lmax,
		     iOrdFunc1_, iOrdFunc2_, 
		     nRootPerInt_, kMax_);  

  kinteg.Compute(int1, int2 , clout);
  
  //  auto init=clout.InitialElls();
  //  std::cout << init.size() << " Cl values computed at l=" ;
  //  std::copy(init.begin(),init.end(),std::ostream_iterator<int>(std::cout,"/"));
  //  std::cout<< std::endl;
  //  clout.Interpolate();
}

}//namespace
