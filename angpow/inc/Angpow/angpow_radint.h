#ifndef ANGPOW_RADINT_SEEN
#define ANGPOW_RADINT_SEEN
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
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>  
#include <stdlib.h>

#include "angpow_numbers.h"    //r_8... def
#include "angpow_exceptions.h" //exceptions
#include "angpow_radial.h"    //radial window

namespace Angpow {

class Quadrature1D {
 public:
  Quadrature1D(int norder=10):  
    norder_(norder), NRval_(-1), DeltaZ_(0) 
    {
      std::stringstream ss; ss << norder; 
      quadFile_ = "./data/ClenshawCurtisRuleData-" + ss.str() + ".txt";  //Todo parametrize "data" directory
      //      quadFile_ = "./data/GaussKronrodRuleData-" + ss.str() + ".txt";  //Todo parametrize "data" directory
    }
  virtual ~Quadrature1D() {}

  void SetQuadrature(RadSelectBase* win);
  int GetNRval()  const {return NRval_;}
  r_8 GetDeltaZ() const {return DeltaZ_;}
  std::vector<r_8> GetNodes()   const { return xQuad_; }
  std::vector<r_8> GetWeights() const { return wQuad_; }

 private:
  int norder_;
  int NRval_;
  r_8 DeltaZ_;
  std::string quadFile_;

  std::vector<r_8> xQuad_;
  std::vector<r_8> wQuad_;
  
};//Quadrature1D

class Quadrature2D {
public:

  typedef std::pair<r_8, r_8> Nodes_t;
  
  Quadrature2D(int norderI = 5, int norderJ = 5):
    norderI_(norderI),
    norderJ_(norderJ),    
    NRvalI_(-1), 
    NRvalJ_(-1), 
    DeltaZI_(0), 
    DeltaZJ_(0) {  
  }
  virtual ~Quadrature2D() {}
  
  void SetQuadrature(RadSelectBase* winI, RadSelectBase* winJ);
  int GetNRvalI()  const {return NRvalI_;}
  int GetNRvalJ()  const {return NRvalJ_;}
  r_8 GetDeltaZI() const {return DeltaZI_;}
  r_8 GetDeltaZJ() const {return DeltaZJ_;}

  //8/11/16 indiv nodes
  std::vector<r_8> GetInodes() const { return xI_; }
  std::vector<r_8> GetJnodes() const { return xJ_; }

  //joined quad
  std::vector<r_8> GetWeights()   const { return wQuad_; }
  

private:
  int norderI_;
  int norderJ_;
  int NRvalI_;
  int NRvalJ_;
  r_8 DeltaZI_;
  r_8 DeltaZJ_;

  std::vector<r_8> wQuad_;
  
  std::vector<r_8> xI_;
  std::vector<r_8> xJ_;
  
}; //Qudrature2D


class Radial2DIntegrator {
public:
  Radial2DIntegrator(int norderI = 5, 
		     int norderJ = 5) :
    norderI_(norderI),
    norderJ_(norderJ),  
    NRvalI_(-1),
    NRvalJ_(-1),
    DeltaZI_(0),
    DeltaZJ_(0),
    winI_(0),
    winJ_(0)  {}
  
  virtual ~Radial2DIntegrator() {}
  
  
  int GetNRvalI()  const { return NRvalI_; }
  int GetNRvalJ()  const { return NRvalJ_; }
  
  std::vector<r_8> GetZInodes() const {
    if(winI_ == 0) throw AngpowError("Radial2DIntegrator: unset I-window");
    return xI_;
  }
  std::vector<r_8> GetZJnodes() const {
    if(winJ_ == 0) throw AngpowError("Radial2DIntegrator: unset J-window");
    return xJ_;
  }
  
  void SetQuadrature(RadSelectBase* winI, RadSelectBase* winJ); 
  
  void ComputeSuperW(std::vector<r_8>& winW);
  
  
private:
  int norderI_;
  int norderJ_;
  
  int NRvalI_;
  int NRvalJ_;
  
  
  std::vector<r_8> wQuad_;

  //indiv quadrature nodes 8/11/16
  std::vector<r_8> xI_;
  std::vector<r_8> xJ_;
  
  r_8 DeltaZI_;
  r_8 DeltaZJ_;
  
  RadSelectBase* winI_; //not the owner 
  RadSelectBase* winJ_; //not the owner 
};//Radial2DIntegrator


}//end namespace
#endif //ANGPOW_RADINT_SEEN
