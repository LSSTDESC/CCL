#ifndef ANGPOWEXCEPTIONS_SEEN
#define ANGPOWEXCEPTIONS_SEEN
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

#include <exception>
#include <string>

namespace Angpow {


class AngpowError : public std::exception {
public:    
  explicit AngpowError(const std::string& m) :  msg_(m) { }
  explicit AngpowError(const char* m): msg_(m) { }
  virtual ~AngpowError() throw() {}
  //! Implementation of std::exception what() method, returning the exception message
  virtual const char* what() const throw() { return msg_.c_str(); }
  
private:
  std::string msg_;
  
};//AngpowError


#define angpow_assert(testval,msg) \
do { if (testval); else throw AngpowError(msg); } while(0)


}//end namespace

#endif //ANGPOWEXCEPTIONS_SEEN
