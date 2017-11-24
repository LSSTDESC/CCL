#ifndef ANGPOW_TOOLS_SEEN
#define ANGPOW_TOOLS_SEEN
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
#include <utility> //pair..
#include <functional> //less
#include <iostream>     // std::cout, std::ostream, std::ios
#include <fstream>      // std::filebuf

#include "angpow_func.h"


namespace Angpow {

/*!
  Algorithm adapted from CLASSgal code (class-code.net)
  The CLASSgal code for Relativistic Cosmological Large Scale Structure, 
  by E. Di Dio, F. Montanari, J. Lesgourgues and R. Durrer, JCAP 1311 (2013) 044

  Compute a liste of l values from [l_min, l_max] (included) using the
  stepping strategies given by the linear step (linstep) and logarithmic step
  (logstep) values. Code adapted from CLASSgal code (http://cosmology.unige.ch/content/classgal)
  
  \input l_min minimal l value
  \input l_max maximal l value
  \input l_linstep linear step
  \input l_logstep logarithmic step [if = 0 => linear stepping]
  \output vector of l values provided by the user
  
*/
void getLlist(int l_min, int l_max, int l_linstep, double l_logstep, std::vector<int>&l);

/*!
  Sorting a pair of objects according to predicate apply on first argument
 */
template <class T1, class T2, class Pred = std::less<T1> >
struct sort_pair_first {
    bool operator()(const std::pair<T1,T2>&left, const std::pair<T1,T2>&right) {
        Pred p;
        return p(left.first, right.first);
    }
};

/*!
  Sorting a pair of objects according to predicate apply on second argument
 */
template <class T1, class T2, class Pred = std::less<T2> >
struct sort_pair_second {
  bool operator()(const std::pair<T1,T2>&left, const std::pair<T1,T2>&right) {
    Pred p;
    return p(left.second, right.second);
  }
};

/*! Sort pairs according to the first argument which in turn is a pair: choose the first arg then the second
 */  
template <class T1, class T2, class Pred = std::less<T1> >
struct sort_pairpair_first {
  bool operator()(const std::pair<std::pair<T1,T1>,T2>&left, const std::pair<std::pair<T1,T1>,T2>&right) {
    Pred p;
    if(left.first.first != right.first.first)
      return p(left.first.first,right.first.first);
    return p(left.first.second, right.first.second);
  }
};
  
/*! Sort pairs according to the first argument which in turn is a pair: choose the second arg then the first
 */  
template <class T1, class T2, class Pred = std::less<T1> >
struct sort_pairpair_second {
  bool operator()(const std::pair<std::pair<T1,T1>,T2>&left, const std::pair<std::pair<T1,T1>,T2>&right) {
    Pred p;
    if(left.first.second != right.first.second)
      return p(left.first.second,right.first.second);
    return p(left.first.first, right.first.first);
  }
};


///////////////////////////////////////////////////////////////////
// Class CSpline et CSpline2 adapted from SOPHYA library (www.sophya.org)
// R. Ansari, J.E Campagne & Ch Magneville
///////////////////////////////////////////////////////////////////

//! Spline fit to a set of points Y=f(X) (as in Numerical Receipes).
//JEC 25/9/16 AutoDeriv: les derivees aux extremes calculees avec les elements du tableau
//            use std::sort facility
class CSpline {

public:

  friend class CSpline2;

  enum { AutoDeriv=4, NaturalAll = 3, NoNatural = 0, Natural1 = 1, NaturalN = 2 };

  explicit CSpline(int n,double* x,double* y,double yp1=0.,double ypn=0.
         ,int natural=NaturalAll,bool order=true);
  

  CSpline(double yp1=0.,double ypn=0.,int natural=NaturalAll);

  virtual ~CSpline();

  void SetNewTab(int n,double* x,double* y
                ,bool order=true,bool force=false);

  void SetBound1er(double yp1 = 0.,double yp2 = 0.);

  //!  Pour changer le type de contraintes sur les derivees 2sd
  inline void SetNaturalCSpline(int type = NaturalAll)
                 { Natural = type;}

  //!  Pour liberer la place tampon qui ne sert que dans ComputeCSpline() et pas dans CSplineInt
  inline void Free_Tmp()
  { 
    if(tmp != NULL) {delete [] tmp; tmp=NULL;}
  }
  
  void ComputeCSpline();

  double CSplineInt(double x) const;

  double operator()(double x) const {return CSplineInt(x); }

protected:


  void DelTab();

  // nombre d elements dans les tableaux X et Y
  int Nel;
  // true si les tableaux ont ete changes
  // et qu il faut recalculer ComputeCSpline()
  bool corrupt_Y2;
  // true si les tableaux X,Y ont ete alloues
  bool XY_Created;
  // type de contraintes sur la derivee 2sd
  int Natural;
  // valeurs imposees de la derivee 1ere aux limites
  double YP1, YPn;

  // tableaux rellement alloues si "order=true" ou seulement
  // connectes aux tableaux externes si "order=false"
  double* X;
  double* Y;

  // tableau des coeff permettant l interpolation,
  // remplis par ComputeCSpline()
  double* Y2;

  // tableau tampon utilise dans ComputeCSpline()
  double* tmp;
  //  int_4*  ind;

};


///////////////////////////////////////////////////////////////////
//! 2D Spline fit to a set of points Y=f(X1,X2) (as in Numerical Receipes).
class CSpline2  {

public:
  CSpline2(int n1,double* x1,int n2,double* x2,double* y
          ,int natural=CSpline::NaturalAll,bool order=true);

  CSpline2(int natural=CSpline::NaturalAll);

  virtual ~CSpline2();

  void SetNewTab(int n1,double* x1,int n2,double* x2,double* y
                ,bool order=true,bool force=false);

//!  Pour changer le type de contraintes sur les derivees 2sd
  inline void SetNaturalCSpline(int type = CSpline::NaturalAll)
                 { Natural = type;}

  void ComputeCSpline();

  double CSplineInt(double x1,double x2);

  double operator()(double x1, double x2) { return CSplineInt(x1,x2); }


protected:

  void DelTab();

  int Nel1, Nel2;
  bool corrupt_Y2;
  bool XY_Created;
  int Natural;

  double* X1;
  double* X2;
  double* Y;
  double* Y2;

  // nombre d elements alloues dans S
  int Nel_S;
  // tableau de CSpline pour interpolation selon X1
  CSpline** S;       // S[0->n2]
  CSpline*  Sint;

  // tableau tampon utilise dans CSplineInt()
  double* tmp;  // tmp[max(n1,n2)]
  //  int_4*  ind;
};

////////////////
// adapted from SOPHYA library (www.sophya.org)
// R. Ansari, J.E Campagne & Ch Magneville
////////////////
class SLinInterp1D : public ClassFunc1D {
public :
  //! Default constructor - represent the function y=x
  SLinInterp1D(); 
  // Regularly spaced points in X with Y values defined by yreg 	
  SLinInterp1D(double xmin, double xmax, std::vector<double>& yreg);
  //  Interpolate to a finer regularly spaced grid, from xmin to xmax with npt points if (npt>0)
  SLinInterp1D(std::vector<double>& xs, std::vector<double>& ys, double xmin=1., double xmax=-1., size_t npt=0); 

  virtual ~SLinInterp1D() { }
        
  double XMin() const { return xmin_; }
  double XMax() const { return xmax_; }
  double DeltaX()  { return dx_; }
  inline double X(long i) const {return xmin_ + i*dx_;}  // returns x value for index i

  // --------------------------------------------------------------
  //! Return the interpolated Y value as a function of X 
  double YInterp(double x) const ;
  //! Return the interpolated Y value as a function of X 
  virtual inline double operator()(double x) const {  return YInterp(x); }
  // --------------------------------------------------------------
        
  // Define the interpolation points through a set of regularly spaced points on X, between xmin and xmax 
  void DefinePoints(double xmin, double xmax, std::vector<double>& yreg);
  // Interpolate to a finer regularly spaced grid, from xmin to xmax with npt points 
  void DefinePoints(std::vector<double>& xs, std::vector<double>& ys, double xmin=1., double xmax=-1., size_t npt=0); 

  // Read  Y's (one/line) for regularly spaced X's from file and call DefinePoints(xmin, xmax, yreg)
  size_t ReadYFromFile(std::string const& filename, double xmin, double xmax, char clm='#');
  // Read pairs of X Y (one pair/line) from file and call DefinePoints(xs, ys ...)
  size_t ReadXYFromFile(std::string const& filename, double xmin=1., double xmax=-1., size_t npt=0, char clm='#');
  
  std::vector<double>& GetVX()  { return xs_; }
  std::vector<double>& GetVY()  { return ys_; }

  //! Print the object (interpolation points) on "ostream". lev=0,1
  std::ostream& Print(std::ostream& os,int lev=0) const ;
  //! Print the object (interpolation points) on "cout". lev=0,1
  inline std::ostream& Print(int lev=0) const { return Print(std::cout, lev); } 

protected:
  std::vector<double> yreg_, xs_, ys_;  // interpolated y value for regularly spaced x 
  double xmin_, xmax_, dx_;        // dx is spacing of finer grid of x's
  size_t ksmx_;                    // Maximum index value in xs_, ys_
  size_t npoints_;                 // Number of regularly spaced points, xmax not included 
};

/*! operator << overloading - Prints the interpolator object on the stream \b s */
inline std::ostream& operator << (std::ostream& s, SLinInterp1D const& a) 
{ a.Print(s,0);  return s; }



}//end namespace
#endif //ANGPOW_TOOLS_SEEN
