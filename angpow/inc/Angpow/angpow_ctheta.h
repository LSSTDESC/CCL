#ifndef ANGPOW_CTHETA_SEEN
#define ANGPOW_CTHETA_SEEN
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
#include "angpow_clbase.h"
#include<vector>
#include<string>

namespace Angpow {

  /*!
Class function to compute efficiently C(theta) from a Clbase
   */

class CTheta {

public:
  //no default ctor
  CTheta() = delete;

  //Ctor from Clbase
  // cl values will be apodized by exp(-(l/ls)^2) where ls=apod*lmax
  CTheta(const Clbase& cl,r_8 apod=0.4);


  //function value at theta (in radians) 
  r_8 operator()(const r_8 & theta);
  
  //lowest achievable precision (sigma) given the apodization: in radians
  inline r_8 Resolution() const {return 1/l_apod;};
  // dump apodized (2l+1)*Cls' into named file
  void WriteApodCls(const std::string & filename) const;

protected:

  std::vector<r_8> _cl,_p;
  double l_apod;

};//CTheta

}//end namespace


#endif //ANGPOW_CLBASE_SEEN
