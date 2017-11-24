#ifndef ANGPOW_PK2CL_SEEN
#define ANGPOW_PK2CL_SEEN
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
#include "angpow_numbers.h"     //r_8... def

namespace Angpow {

  /*!
    Hub class to perform the computation of the Cl from the P(k) 
    notice that Autocorrelation for large redshift band is a particular casr of 
    Crosscorrelation
   */

  
  class PowerSpecBase;
  class CosmoCoordBase;
  class IntegrandBase;
  class RadSelectBase;
  class Clbase;
  

class Pk2Cl {
public:

  //!Ctor
  Pk2Cl();
  //! Dtor
  virtual ~Pk2Cl() {}

  //! fix order of the radial integration if necessary
  void SetRadOrder(int ord1, int ord2) { nRadOrder1_ = ord1;  nRadOrder2_ = ord2;}
  void SetOrdFunc(int ord1, int ord2) { iOrdFunc1_ = ord1; iOrdFunc2_ = ord2;}
  void SetNRootPerInt(int n) { nRootPerInt_ = n; }
  void SetKmax(r_8 kMax) { kMax_ = kMax; }

  void PrintParam();

  
  //! Correlation P(k) => Cl(z,z')  Version  optimized  wrt sampling/FFT
  void Compute(PowerSpecBase& pws, CosmoCoordBase& coscoord,
	       RadSelectBase* Z1win, RadSelectBase* Z2win, int Lmax, 
	       Clbase& clout);
  void Compute(IntegrandBase& int1, IntegrandBase& int2,
	       CosmoCoordBase& coscoord,
	       RadSelectBase* Z1win, RadSelectBase* Z2win, int Lmax, 
	       Clbase& clout);
  

protected:

  int nRadOrder1_;     //<! order of radial integration 1D OR first for 2D
  int nRadOrder2_;     //<! order of radial integration second for 2D
  int iOrdFunc1_;      //<! Order of the Chebyshev Transform of 1st Function
  int iOrdFunc2_;      //<! Order of the Chebyshev Transform of 2nd Function
  int nRootPerInt_;    //<! number of Bessel roots per sub-k-integration 
  r_8 kMax_;           //<! direct cut on the highest k-value. 
  
};

}//end namespace
#endif //ANGPOW_PK2CL_SEEN
