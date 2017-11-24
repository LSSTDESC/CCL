#ifndef ANGPOWNUMBERS_SEEN
#define ANGPOWNUMBERS_SEEN
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
 *  The file originates from M. Reinecke who kindly has provided it to J.E.C
 */

// Template magic to select the proper data types. These templates
// should not be used outside this file.

namespace Angpow {

template <typename T, bool equalSize> struct sizeChooserHelper__
  { typedef void TYPE; };

template <typename T> struct sizeChooserHelper__<T,true>
  { typedef T TYPE; };

template <typename T1, typename T2, typename T3> struct sizeChooserHelper2__
  { typedef T1 TYPE; };

template <typename T2, typename T3> struct sizeChooserHelper2__ <void, T2, T3>
  { typedef T2 TYPE; };

template <typename T3> struct sizeChooserHelper2__ <void, void, T3>
  { typedef T3 TYPE; };

template <> struct sizeChooserHelper2__ <void, void, void>
  { };

template <int sz, typename T1, typename T2=char, typename T3=char>
  struct sizeChooser__
  {
  typedef typename sizeChooserHelper2__
    <typename sizeChooserHelper__<T1,sizeof(T1)==sz>::TYPE,
     typename sizeChooserHelper__<T2,sizeof(T2)==sz>::TYPE,
     typename sizeChooserHelper__<T3,sizeof(T3)==sz>::TYPE >::TYPE TYPE;
  };

typedef signed char int_1; // correct by definition
typedef unsigned char uint_1; // correct by definition
typedef sizeChooser__<2, short, int>::TYPE int_2;
typedef sizeChooser__<2, unsigned short, unsigned int>::TYPE uint_2;
typedef sizeChooser__<4, int, long, short>::TYPE int_4;
typedef sizeChooser__<4, unsigned int, unsigned long, unsigned short>::TYPE uint_4;
typedef sizeChooser__<8, long, long long>::TYPE int_8;
typedef sizeChooser__<8, unsigned long, unsigned long long>::TYPE uint_8;

typedef sizeChooser__<4, float, double>::TYPE r_4;
typedef sizeChooser__<8, double, long double>::TYPE r_8;
typedef sizeChooser__<16, double, long double>::TYPE r_16;

}//namespace

#endif //ANGPOWNUMBERS_SEEN
