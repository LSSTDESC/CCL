#ifndef ANGPOW_INTEGRAND_SEEN
#define ANGPOW_INTEGRAND_SEEN
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
#include "angpow_powspec_base.h"
#include "angpow_powspec.h"
#include "angpow_integrand_base.h"

namespace Angpow {



// JN 20/04/2017 Class to compute
//     k j_l(k ri) Sqrt[P_m(k,zi)]
// with bias = 1.
class Integrand : public IntegrandBase {
public:
  //Use pointer to avoid Copy Ctor
 Integrand(PowerSpecFile* PSpec, CosmoCoordBase* cosmo, r_8 bias, bool include_rsd): 
  PSpec_(PSpec), cosmo_(cosmo), bias_(bias), include_rsd_(include_rsd) {}

  void Init(int ell, r_8 z) {
    ell_ = ell;
    z_ = z;
    R_ = cosmo_->r(z);
    jlR_ = new JBess1(ell,R_);
    PSpec_->Init(z);
    if(include_rsd_) {
      jlp1R_ = new JBess1(ell+1,R_);
      fz_ = PSpec_->Growth()->LogGrowthDeriv(z);
    }
  } 
  virtual r_8 operator()(r_8 k) const {
    r_8 x = k*R_;
    r_8 jlRk = (*jlR_)(k);
    r_8 delta = bias_*jlRk; // density term with bias
    if(include_rsd_){ // RSD term
      r_8 jlRksecond = 0.;
      if(x<1e-40) { // compute second derivative j"_ell(r(z)*k)
    	if(ell_==0) {
    	  jlRksecond = -1./3. + x*x/10.;
    	} else if(ell_==2) {
    	  jlRksecond = 2./15. - 2*x*x/35.;
    	} else {
    	  jlRksecond = 0.;
    	}
      } else {
    	jlRksecond = 2.*(*jlp1R_)(k)/x + (ell_*(ell_-1.)/(x*x) - 1.)*jlRk;
      }
      delta += fz_*jlRksecond;
    }
    return k*sqrt((*PSpec_)(k,z_))*delta;
  }
  virtual ~Integrand() {}
  /*! Explicit to get a clone of the primary object via shallow copy
    using the Copy Ctor
   */
  virtual Integrand* clone() const {
    return new Integrand(static_cast<const Integrand&>(*this));
  }
  virtual void ExplicitDestroy() { 
    if(jlR_) delete jlR_;
  }

private:
  PowerSpecFile* PSpec_;  //no ownership
  CosmoCoordBase* cosmo_;  //no ownership
  JBess1* jlR_;  // j_ell(k*R)
  JBess1* jlp1R_;   // j_(ell+1)(k*R)
  int ell_;    // multipole ell
  r_8 z_;      // redshift z
  r_8 R_;   // radial comoving distance r(z)
  r_8 fz_;  // growth rate f(z)
  r_8 bias_;   // bias b(z)
  bool include_rsd_;  // include RSD effects or not

  //Minimal copy to allow Main operator(int, r_8, r_8) to work
  //JEC 22/4/17 use cloning of PowerSpectrum
  Integrand(const Integrand& copy) : cosmo_(copy.cosmo_), 
    jlR_(0), jlp1R_(0),  bias_(copy.bias_), include_rsd_(copy.include_rsd_) {
    PSpec_ = (copy.PSpec_)->clone();
  } 
  
  //forbid for the time beeing the assignment operator
  Integrand& operator=(const Integrand& copy);

}; //Integrand



}//end namespace
#endif //ANGPOW_INTEGRAND_SEEN
