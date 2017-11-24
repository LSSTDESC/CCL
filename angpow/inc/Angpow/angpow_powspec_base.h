#ifndef ANGPOW_POWSPEC_BASE_SEEN
#define ANGPOW_POWSPEC_BASE_SEEN
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

#include "angpow_bessel.h"
#include "angpow_cosmo_base.h"

namespace Angpow {

  
class PowerSpecBase {
public:
  //PowerSpecBase(bool has_rsd=false, r_8 fz=0.): has_rsd_(has_rsd), fz_(fz) {}
  PowerSpecBase() {}
  virtual ~PowerSpecBase() {}
  /*!
    clone operator that must be like
    return new Derived(static_cast<const Derived&>(*this));
    The Copy Constructor should perform shadow copy (ie not deep copy of pointer object)
  */ 
  virtual PowerSpecBase* clone() const =0;
  //! main operator
  //virtual r_8 operator()(int ell, r_8 k, r_8 z) =0; //P_l(k,z)
  virtual r_8 operator()(r_8 k, r_8 z) =0; //P(k,z)
  //! function which is called before k-sampling so may be used to setup constant values/fucntions
  //virtual void Init(int ell, r_8 z) {} 
  virtual void Init(r_8 z) {} 
  //! function that should be used to free pointer not freed by the shadow copy of the clone operator.
  virtual void ExplicitDestroy() {}
};


  // OLD VERSION
  //k j_l(k ri) Sqrt[P_l(k,zi)]
class PowSqrtJBess : public ClassFunc1D {
public:
  //Use pointer to avoid Copy Ctor
  PowSqrtJBess(PowerSpecBase* PSpec, JBess1* jfunc, int ell, r_8 z): 
   PSpec_(PSpec), jfunc_(jfunc), ell_(ell), z_(z) {}
  virtual void Init(int ell, r_8 z) {ell_=ell, z_=z;} 
  virtual r_8 operator()(r_8 k) const {
    return k*sqrt((*PSpec_)(k,z_))*(*jfunc_)(k);
  }
  virtual ~PowSqrtJBess() {}
  /*! Explicit to get a clone of the primary object via shallow copy
    using the Copy Ctor
   */
  virtual PowSqrtJBess* clone() const {
    return new PowSqrtJBess(static_cast<const PowSqrtJBess&>(*this));
  }
private:
  PowerSpecBase* PSpec_;  //no ownership
  JBess1* jfunc_;         //"
  int ell_;
  r_8 z_;
  //Minimal copy to allow Main operator(int, r_8, r_8) to work
  //PowSqrtJBess(const PowSqrtJBess& copy) :
  //SqrtIntegrandBase(z_).copy.SqrtIntegrandBase(z_), PSpec_(copy.PSpec_), jfunc_(copy.jfunc_), ell_(copy.ell_) {} 
}; //PowSqrtJBess



}//end namespace
#endif //ANGPOW_POWSPEC_BASE_SEEN
