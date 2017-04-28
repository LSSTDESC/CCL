#ifndef ANGPOW_COSMO_BASE_SEEN
#define ANGPOW_COSMO_BASE_SEEN
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

#include <vector>
#include <algorithm>
#include <functional>

#include "angpow_numbers.h"


namespace Angpow {

  //! base class for Comological Coordinate computations used in generic algorithms
class CosmoCoordBase {
public: 
  CosmoCoordBase() {}
  virtual ~CosmoCoordBase() {}
  //! r(z): radial comoving distance (Mpc)
  virtual r_8 r(r_8 z) const = 0;
  /*! z(r): redshift as the inverse of radial comoving distance (distance in Mpc) 
    [not used in pure angpow application] 
  */
  virtual r_8 z(r_8 r) const = 0;
};//CosmoCoordBase


//! Conversion between redshift and comoving distance (1D)
inline void RedShift2Radius(const std::vector<r_8>& zNodes, 
			    std::vector<r_8>& rNodes, const  CosmoCoordBase& ccb) {
  rNodes.resize(zNodes.size());
  std::function<r_8(r_8)> zTor = std::bind(&CosmoCoordBase::r,&ccb, std::placeholders::_1); //JEC 17/11/16
  std::transform(zNodes.begin(), zNodes.end(), rNodes.begin(), zTor);

 }//RedShift2Radius

}//end namespace
#endif //ANGPOW_COSMO_BASE_SEEN
