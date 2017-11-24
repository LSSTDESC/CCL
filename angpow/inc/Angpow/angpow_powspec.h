#ifndef ANGPOW_POWSPEC_SEEN
#define ANGPOW_POWSPEC_SEEN
/*
 *  This file is part of Angpow.
 *
 *  Angpow is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  Angpow is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Angpow; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  Angpow is being developed at the Linear Accelerateur Laboratory (LAL)
 *  91898 ORSAY CEDEX - FRANCE
 *  main author: J.E Campagne
 *    co-authors: J. Neveu, S. Plaszczynski
 */

#include <iostream>
#include <fstream> 
#include <string> 
#include <vector>
#include <numeric> 
#include <math.h>


#include "angpow_numbers.h"
#include "angpow_func.h"
#include "angpow_tools.h"
#include "angpow_cosmo.h"
#include "angpow_powspec_base.h"

//--------------
//Example of Implementation in the context of
// Computation of P(k) Growth(z)^2 from an external file to get P(k) at z=0 and
// the Growth-factor from a simple implementation of Eisenstein & Hu parametrization
//--------------

namespace Angpow {

class GrowthBase {
 public:
  GrowthBase() {}
  virtual ~GrowthBase() {}
  virtual inline r_8 operator()(const r_8& zz) const {return 1;};
  virtual inline r_8 LogGrowthDeriv(r_8 z) {
    r_8 h=0.0001;
    r_8 Gz = (*this)(z);
    r_8 Gzph = (*this)(z+h);
    r_8 Gzmh = (*this)(z-h);
    return (1.+z)/Gz*(Gzph-Gzmh)/(2.*h);  
  }
};

class GrowthEisenstein : public GrowthBase {
public:
  GrowthEisenstein(r_8 OmegaMatter0,r_8 OmegaLambda0);
  GrowthEisenstein(GrowthEisenstein& d1) :  O0_(d1.O0_) , Ol_(d1.Ol_),  Ok_(d1.Ok_), invD1z0_(d1.invD1z0_){}  
  virtual ~GrowthEisenstein() {}
  virtual inline r_8 operator()(const r_8& zz) const {

    // see Formulae A4 + A5 + A6 page 614
    r_8 z=zz+1;
    r_8 z2 = z*z, z3 = z2*z;
    
    // Calcul du growthfactor pour z
    r_8 den = Ol_ + Ok_*z2 + O0_*z3;
    r_8 o0z = O0_ *z3 / den;
    r_8 olz = Ol_ / den;
    
    r_8 D1z = pow(o0z,4./7.) - olz + (1.+o0z/2.)*(1.+olz/70.);
    D1z = 2.5*o0z / z / D1z;
    
    return D1z * invD1z0_;
  }//operator



  void SetParTo(r_8 OmegaMatter0,r_8 OmegaLambda0);
  bool SetParTo(r_8 OmegaMatter0);
protected:
  r_8 O0_;
  r_8 Ol_;
  r_8 Ok_;        //JEC 27/11/16 precompute
  r_8 invD1z0_;   //"
};




////////////////////
//Power spectrum classes
////////////////////
 




/*!
  Load a power spectrum from a two column k P(k) ascii file
  But it is not satisfactory as one needs also at least the "comov. distance function(z)"
  from the cosmological parameters used by the external tool
*/
class PowerSpecFile : public PowerSpecBase {
 public:
  //! Constructor
  PowerSpecFile(const CosmoCoord& su,std::string inpkname, r_8 zref, 
		r_8 kmin, r_8 kmax, 
		bool h_rescale=true, 
		bool growth_rescale=false);
  
  //! Destructor
  virtual ~PowerSpecFile() { 
    //DO NOT DESTROY THE POINTER on GrowthEisenstein* and SLinInterp1D*  
  }

  //! Used to delete explicitly the local pointers
  virtual void ExplicitDestroy() { 
    if(mypGE_) delete mypGE_;
    if(Pk_) delete Pk_;
  }

  /*! Explicit to get a clone of the primary object via shallow copy
    using the Copy Ctor
   */
  virtual PowerSpecFile* clone() const {
    return new PowerSpecFile(static_cast<const PowerSpecFile&>(*this));
  }
  
  /*! called by angpow_kinteg.cc to fix the value of some function
    at fixed z value (and l too if necessayr)
   */
  void Init(r_8 z) {r_8 tmp= mypGE_->operator()(z); growth2_ = tmp*tmp;}

  //Main operator
  virtual r_8 operator()(r_8 k, r_8 z) {
     return growth2_*(Pk_->operator()(k));
  }


  //JEC
  GrowthEisenstein* Growth() { return mypGE_;}
  GrowthEisenstein* Growth() const { return mypGE_;}

    
private:

  SLinInterp1D* Pk_;          //!< access to  Pk(k)
  GrowthEisenstein* mypGE_;   //!< access tp  D(z)
  r_8 growth2_;               //!< D(zi)^2

  //forbid for the time beeing the assignment operator
  PowerSpecFile& operator=(const PowerSpecFile& copy);
  
  //Minimal copy to allow Main operator(int, r_8, r_8) to work
  PowerSpecFile(const PowerSpecFile& copy) :
    Pk_(copy.Pk_), mypGE_(copy.mypGE_), growth2_(copy.growth2_) {}
};




class PowGrowth0 : public ClassFunc1D {
public:
  PowGrowth0(PowerSpecFile* PSpec, r_8 zref):  
    PSpec_(PSpec), zref_(zref) {ell_=0;}
  virtual ~PowGrowth0() {}
  virtual r_8 operator()(r_8 k) const {
    return k*k*(*PSpec_)(k,zref_);
  }
private:
  PowerSpecFile* PSpec_;  //no ownership
  r_8 zref_;
  int ell_; //dummy
  
};


}//end namespace
#endif //ANGPOW_POWSPEC_SEEN
