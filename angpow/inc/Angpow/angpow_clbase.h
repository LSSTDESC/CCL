#ifndef ANGPOW_CLBASE_SEEN
#define ANGPOW_CLBASE_SEEN
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
#include <utility>
#include "angpow_numbers.h"

namespace Angpow {

  /*!
    Class manipulation of pairs <ell, cl>
    generation of ell list according to sampling scheme in the range [0, Lmax-1]
    interpolation of the Cls
   */

class Clbase {
public:
  typedef std::pair<int, r_8> Acl;
  //Ctor
  Clbase(int Lmax, int l_linstep=40, r_8 l_logstep=1.15);
  //Get
  std::vector<Acl> GetCls() const { return cls_; }
  int  Size() const {return cls_.size(); }
  //Op.
  Acl& operator[](int index_l) { return cls_[index_l]; }
  //Interpolation over ells the Cl using spline 
  void Interpolate();

  inline std::vector<int> InitialElls() const {return ells_;}

protected:
  int Lmax_;               //<! all ells are in [0, Lmax-1]
  std::vector<Acl> cls_;   //<! vector of couple <ell, cl>
  std::vector<int> ells_;  //<! keep track of the initial ells generated
  std::vector<int> ellsAll_;  //<! All possible ells in the range [0, Lmax-1]
  bool maximal_;             //<! true if ellsAll_ = ells_

};//Clbase

}//end namespace


#endif //ANGPOW_CLBASE_SEEN
