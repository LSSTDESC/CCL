#ifndef ANGPOW_KINTEG_SEEN
#define ANGPOW_KINTEG_SEEN
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
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#ifdef PROFILING
#include "walltimer.h" //profiling 
#endif

#include "angpow_numbers.h"
#include "angpow_chebyshevInt.h"


namespace Angpow {

  class PowerSpecBase;
  class IntegrandBase;
  class ChebyshevInt;
  class Clbase;

//! Cartesian z1,z2 sampling minimal call Gauss
class KIntegrator {
public:

  //8/11/16 new interface to avoid splitting
  KIntegrator(const std::vector<r_8>& radiusI, const std::vector<r_8>& radiusJ,
	      const std::vector<r_8>& zI, const std::vector<r_8>& zJ,
	      const std::vector<r_8>& winW,
	      int NRvalI, int NRvalJ, int Lmax, 
	      int iOrd1, int iOrd2, int nRootPerInt=100, 
	      r_8 kMax=10);

  
  virtual ~KIntegrator() {
    for (size_t i=0;i<cheInts_.size();i++) delete cheInts_[i];
  }
  
  virtual void Compute(PowerSpecBase& pws, Clbase& clout); //nouvelle interface 10/10/16
  virtual void Compute(IntegrandBase& int1, IntegrandBase& int2, Clbase& clout); //nouvelle interface JN 04/04/2017
  
protected:

  std::vector<ChebyshevInt*> cheInts_;  //<! Chebyshev Transform and integration


  int nRootPerInt_;       //<! nomber of Bessel roots per sub-k-integration 
  r_8 kMax_;              //<! direct cut on the highest k-value. 

  int NRvalI_;            //<! number of redshift points in selection I
  int NRvalJ_;            //<! number of redshift points in selection J
  int NRvals_;            //<! total number of redshift points
  int Lmax_;              //<! Lmax
  
  
  std::vector<r_8> Ri_;   //<! values of r(zi) Mpc
  std::vector<r_8> Rj_;   //<! values of r(zj) Mpc
  std::vector<r_8> zi_;   //<! redshift zi
  std::vector<r_8> zj_;   //<! redshift zj
  bool same_sampling_;    //<! trigger true if the sampling on selection I is the same for selection J

  std::vector<r_8> winW_; //!< total weights of the 2D quadrature * the product of the selection I & J

}; //KIntegrator


}//end namespace

#endif //ANGPOW_KINTEG_SEEN
