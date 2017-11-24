#ifndef ANGPOW_INTEGRAND_BASE_SEEN
#define ANGPOW_INTEGRAND_BASE_SEEN
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

namespace Angpow {


//! Base class for the integrand functions
class IntegrandBase : public ClassFunc1D {
 public:
  IntegrandBase()  {}
  virtual ~IntegrandBase() {}
  virtual r_8 operator()(r_8 k) const =0;
  virtual void Init(int ell, r_8 z) {} 
  virtual IntegrandBase* clone() const  =0;
  virtual void ExplicitDestroy() {}

};//IntegrandBase


}//end namespace
#endif //ANGPOW_INTEGRAND_BASE_SEEN
