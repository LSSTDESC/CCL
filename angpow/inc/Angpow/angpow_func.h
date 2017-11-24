#ifndef UTIL_H_SEEN
#define UTIL_H_SEEN
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
 *  along with libsharp; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  Angpow is being developed at the Linear Accelerateur Laboratory (LAL)
 *  91898 ORSAY CEDEX - FRANCE
 *  main author: J.E Campagne
 *    co-authors: J. Neveu, S. Plaszczynski
 */

namespace Angpow {


/* 1D function Class
*/

/*!
  \class GenericFunction 
  \brief Base class definition for objects representing functions 
*/
class GenericFunction {
public:
  GenericFunction() { }
  virtual ~GenericFunction() { }
};

/*! 
  \class ClassFunc1D 
  \brief Abstract interface definition for 1 D real functions  viewed as classes

  This class represents real function of a single real argument : double f(double x)
  Inheriting classes should define the double operator()(double x)
  For convenience, ClassFunc has been defined using a typedef as an alias
*/
class ClassFunc1D : public GenericFunction {
public:
  virtual double operator()(double x) const =0;
};

//! typedef definition of ClassFunc as ClassFunc1D for convenience
typedef ClassFunc1D ClassFunc ;
  

/*! 
  \class Function1D 
  \brief ClassFunc constructed from a a function:simple forward to function
  For convenience, ClassFunc has been defined using a typedef as an alias
*/
//ClassFunc constructed from a a function:simple forward to function
class Function1D : public ClassFunc1D {
 private:
  double (*f)(double);
 public:
  Function1D(double (*g)(double)):f(g){}
  virtual double operator()(double x) const { return f(x);}
};

typedef Function1D Function; 

/*! 
  \class ClassFunc2D 
  \brief Abstract interface definition for 2 D real functions  viewed as classes

  This class represents real function of a two real arguments : double f(double x, double y)
  Inheriting classes should define the double operator()(double x, double y)
  For convenience, ClassFunc has been defined using a typedef as an alias
*/
class ClassFunc2D : public GenericFunction {
public:
  virtual double operator()(double x, double y) const =0;
};

/*! 
  \class Function2D 
  \brief ClassFunc constructed from a a function:simple forward to function
  For convenience, ClassFunc has been defined using a typedef as an alias
*/
//ClassFunc constructed from a a function:simple forward to function
class Function2D : public ClassFunc2D {
 private:
  double (*f)(double, double);
 public:
 Function2D(double (*g)(double, double)):f(g){}
  virtual double operator()(double x, double y) const { return f(x,y);}
};

}//end namespace
#endif //UTIL_H_SEEN
