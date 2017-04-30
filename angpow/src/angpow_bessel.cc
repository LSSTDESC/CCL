#include "Angpow/angpow_bessel.h"
#include "Angpow/angpow_exceptions.h"
#include "Angpow/angpow_parameters.h"
#include <boost/exception/diagnostic_information.hpp> 
#include <boost/exception_ptr.hpp> 
#include <boost/math/policies/policy.hpp> //test
#include <boost/mpl/bool.hpp> //test devra etre enleve

namespace Angpow {

void BesselRoot::CmpRoots() {
  typedef boost::math::policies::policy<boost::math::policies::digits10<5> > pol;

    try{
      for (int lcur =0; lcur<lmax_; lcur++){
	//j(l,x) ~ J(l+1/2,x)/sqrt(x). 
	r_4 order = (r_4)(lcur + 0.5);
	std::vector<r_8> zeros(nroots_);
	int keepFirstRoots = 0; //was 2 JEC 9/10/16
	for(int i=0; i<nroots_; i++) {
	  int rIdx = (i<keepFirstRoots) 
	    ? i 
	    : keepFirstRoots-1+(i-keepFirstRoots+1)*step_;
	  // zeros[i] = boost::math::cyl_bessel_j_zero(order,rIdx+1,pol()); //1-based index of zero
	  //JEC 20mai16: use the approximated roots given by boost inside the
	  //boost::math::cyl_bessel_j_zero. It means do not use newton_raphson_iterate process
	  zeros[i] = boost::math::detail::bessel_zero::cyl_bessel_j_zero_detail::initial_guess(order, rIdx+1, pol());
	}
	std::copy(zeros.begin(),zeros.end(),qln_.begin() + lcur*nroots_);
      }
    }
    catch (boost::exception& ex) {
      // error handling
      std::cerr << boost::diagnostic_information(ex);
    }

}// BesselRoot::CmpRoots

  
std::vector<r_8> MakeBesselJImpXmin(int Lmax=2000, r_8 cut=5e-10) {
  BesselJImp jl;

  std::vector<r_8>vec(Lmax);

  for(int el=0;el<Lmax;el++){
    if(el==0){//j_0(x)=sin(x)/x
      vec[el]=0;
      continue;
    }
    r_8 xmin=0;
    r_8 xmax=el+0.5;
    //JEC 10/11/16    r_8 fmax = boost::math::sph_bessel(el,xmax);
    r_8 fmax;
    jl.bessel_j(el,xmax,&fmax);

    if(fmax < cut)
      throw AngpowError("BesselJImp::Xmin cut not appropriate. FATAL");
    
    r_8 eps=0.5e-10;
    r_8 xmiddle=(xmax+xmin)*0.5;
    r_8 fmiddle;
    while((xmax-xmin)/(xmax+xmin)> eps){
      xmiddle = (xmax+xmin)*0.5;
      //JEC 10/11/16      fmiddle = boost::math::sph_bessel(el,xmiddle);
      jl.bessel_j(el,xmax,&fmiddle);
      if(fmiddle>cut)
	xmax = xmiddle;
      else
	xmin = xmiddle;
    }//while

    vec[el]=xmiddle;

  }//el-loop
  return vec;
}

std::vector<r_8> BesselJImp::xmin_ = MakeBesselJImpXmin(Param::Instance().GetParam().Lmax_for_xmin,
							Param::Instance().GetParam().jl_xmin_cut);

}//namespace 

