#ifndef ANGPOW_RADIAL_BASE_SEEN
#define ANGPOW_RADIAL_BASE_SEEN
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

#include "angpow_numbers.h"
#include "angpow_func.h"


namespace Angpow {
  //! Base class of the selection fucntion
class RadSelectBase: public ClassFunc1D {
 public:
    RadSelectBase(r_8 zmin, r_8 zmax): zmin_(zmin), zmax_(zmax) {}
  virtual ~RadSelectBase() {}
  virtual r_8 operator()(r_8 z) const =0;

  r_8 GetZMin() const {return zmin_;}
  r_8 GetZMax() const {return zmax_;}
protected:
  r_8 zmin_;
  r_8 zmax_;  
  
};//RadSelectBase

}//end namespace
#endif //ANGPOW_RADIAL_BASE_SEEN
