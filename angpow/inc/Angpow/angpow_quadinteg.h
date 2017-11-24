#ifndef QUADINTEG_H_SEEN
#define QUADINTEG_H_SEEN
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

// include standard c/c++
#include <cmath>
#include <cfloat>
#include <stdlib.h>

#include <vector>
#include <list>
#include <utility>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>

#include <limits> //for precision features
#include <iterator> //ostream_iterator

#include "angpow_func.h"

#define DUMPLEV 0

using namespace std; //See if better on each routine

namespace Angpow {

/*!
  \class Quadrature
  \brief quadrature defined in [0,1]
*/
template <typename T, typename TDerived>
class Quadrature {
    
public:
  typedef pair<T,T> values_t; //{intgral value, error}
  typedef pair<T,T> bounds_t; //{a,b} bounds
  typedef pair< bounds_t , values_t > result_t; // { {a,b},{int,err} }
  typedef list< result_t > colresults_t; // collection of result_t
  
  /*!
    Utility  Sort funtion according to error (largest first)  
   */
  struct Comparator {
    Quadrature* m;
    Comparator(Quadrature* p): m(p) {}
    bool operator()(const result_t& r1, const result_t& r2) {
      return (r1.second).second > (r2.second).second;
    }
  };

  /*! Class for intermediate recursive result
   */
  struct ResultClass {
    T integral;
    T error;
    vector<T> scaledNodes;

    ResultClass(T integ, T err, const vector<T>& vec): 
      integral(integ), error(err), scaledNodes(vec) {}

    ResultClass(const ResultClass& other):
      integral(other.integral),error(other.error),scaledNodes(other.scaledNodes){}
    
    ResultClass& operator=(const ResultClass& other){
      if(this == &other) return *this;
      integral = other.integral;
      error    = other.error;
      scaledNodes = other.scaledNodes;
      return *this;
    }
    

    void Print() {
      cout << "Integral = " << integral << " +/- " << error <<endl;
      cout << "Nodes <" << setprecision(10); 
      copy(scaledNodes.begin(), scaledNodes.end(), ostream_iterator<T>(cout, ", "));
      cout << ">" << setprecision(6)<<endl;
    }
    
  };

public:  
    /*!
      Default Ctor
    */
  Quadrature(string name="",size_t npts=0,string fName="", bool init=false):
    name_(name),
    npts_(npts),
    func_(0){

    if(init){
      //compute abscissa, weights, error weights and save into fName
      //      cout << "Init for " << name << " with " << npts << " pts saved in " << fName << endl;
      TDerived& tDerivedObj = (TDerived&)*this;
      try {

	//	tDerivedObj.ComputeAbsWeights(fName);
	tDerivedObj.ComputeAbsWeights();

	if(fName.length()!=0) {//save
	  ofstream fout(fName.c_str(), ofstream::out);
	  int size = absc_.size();
	  for (int i=0; i<size;i++){
	    fout <<fixed << showpoint << setprecision(30) << absc_[i] << "\t" << absw_[i] << "\t" << errw_[i] << endl;
	  }
	  fout.close();
	}//save

      } catch (exception& ex) {
	string msg = "Init ";
	msg += name + ": ERROR ";
	msg += ex.what();
	cerr << msg << endl;
	exit(EXIT_FAILURE);
      } catch (string msg) {
	cerr << msg << endl;
	exit(EXIT_FAILURE);
      } catch (...) {
	cerr << "Init " << name << " unexpected error " << endl;
	exit(EXIT_FAILURE);
      }
    } else {
      //read abscissa, weights, error weights from fName
      ifstream inFile;
      inFile.exceptions ( ifstream::eofbit | ifstream::failbit | ifstream::badbit );
      absc_.resize(npts);
      absw_.resize(npts);
      errw_.resize(npts);
      try {
	inFile.open(fName.c_str());
	for(size_t i=0;i<npts;i++){
	  inFile >> absc_[i]>> absw_[i]  >> errw_[i];
	}
	inFile.close();
      } catch (ifstream::failure e) {
	cerr << name <<" Exception opening/reading/closing file <"<<fName<<">"<<endl;
	exit(EXIT_FAILURE);
      } //try/catch read

    }//initialisation
 
  }//Ctor
  /*!
    Dtor: care cannot be virtual because of template
   */
  ~Quadrature() {}
    
   
  /*!
    Rescale abscissa and weights from [xInmin, xInmax] to [xOutmin, xOutmax]
    Typical use-case the abs & weights are computed for (-1,1) range
    and we need to translate them to (0,1) or vice-versa
  */
  void RescaleAbsWeights(T xInmin = -1.0, T xInmax = 1.0,
			  T xOutmin=0.0, T xOutmax=1.0) {
     
    T deltaXIn = xInmax-xInmin;
    T deltaXOut= xOutmax-xOutmin;
    T scale = deltaXOut/deltaXIn;
    for(size_t i=0; i<npts_;i++){
      absc_[i] = ((absc_[i]-xInmin)*xOutmax-(absc_[i]-xInmax)*xOutmin)/deltaXIn;
      absw_[i] *= scale;
      errw_[i] *= scale;      
    }
  }//RescaleAbsWeights
  
   /*!
     Set function
   */
  void SetFunction(ClassFunc1D& aFunc) { func_ = &aFunc;}
  /*!
    Set the integration bounds
  */
  void SetBounds(T xmin, T xmax) { initBounds_ = make_pair(xmin,xmax); }
  /*!
    Set both Function & bounds
  */
  void SetFuncBounded(ClassFunc1D& aFunc, T xmin, T xmax) {
    SetFunction(aFunc);
    SetBounds(xmin,xmax);
  }
  
  /*!
    Get abcissa, abscissa weights, error weights 
  */
  vector<T> GetAbscissa()  const {return absc_;}
  vector<T> GetAbscissaW() const {return absw_;}
  vector<T> GetErrorW()    const {return errw_;}

  /*!
     Get number of points used
   */
  size_t GetNPoints() const {return npts_;}
  size_t GetOrder()   const {return 0;} //should be reimplemented

  /*!
    Get Name
   */
  string    GetName()      const {return name_;}

  
  /*!
    Debug
  */
   void ToOut(ostream& out, const result_t& aResult) {
     out << "{" << aResult.first.first << ",," 
	  <<  aResult.first.second << "}, "
	  <<  aResult.second.first << ", "
	  << aResult.second.second << "}";
   }//ToOut
  
  void ToOut(ostream& out, const colresults_t& aColRes) {
    typename colresults_t::const_iterator iRes,iEnd;
    iEnd = aColRes.end();
    out << "{";
    for(iRes = aColRes.begin();iRes!=iEnd;++iRes){
      ToOut(out,*iRes); cout << ",";
    }
    out << "}" << endl;
  }//ToOut
  
   /*!
     Compute func integral between the current bounds
     The abscissa/weights of the Quadratures are defined for (0,1) interval.
   */
  ResultClass IntEval(const bounds_t& curBounds) {
    T integral=0.;
    T error   = 0.;
    T boundsDist = curBounds.second - curBounds.first;
    vector<T>xi(npts_);
    
    for (size_t i=0;i<npts_;i++){
      xi[i] = curBounds.first + absc_[i]*boundsDist;
      T fi = (*func_)(xi[i]);
      integral += absw_[i] * fi;
      error    += errw_[i] * fi;
    }
    return ResultClass(integral*boundsDist,fabs((double)error*boundsDist),xi);
  }//IntEval
  

  /*! Recursive computation of integral on a finite range
    \input curBounds = {x0,x1}
    \input cut: threshold to be compared to the integral error to stop or not
    \output : {integral, error}
  */
  ResultClass LocalAdapStratRecur(const bounds_t& curBounds, T cut, int maxdepth, int depth=0) {

#if DUMPLEV > 2
    cout << "depth ["<<depth<<"]";
    for(int i=0;i<depth;i++) cout << "...";
#endif

    ResultClass curResult = IntEval(curBounds); //{int, err, nodes}

    T error    = curResult.error;
  
    if (error <= cut || depth >= maxdepth){
#if DUMPLEV > 2
      cout << "Int[" <<curBounds.first << ", " <<curBounds.second << "]: "
	   << curResult.integral << " +/- " << curResult.error << ".... Ok" << endl;
#endif
      if(depth == maxdepth) cout << "Warning, max recursive reached" <<endl;
      return curResult;
    }  
    int nnodes = curResult.scaledNodes.size();
    int lastNodes=nnodes-1;
    T integral = (T)0.;
    error      = (T)0.;
    for(int i=0;i<lastNodes;i++){
      ResultClass locres = 
	LocalAdapStratRecur(make_pair(curResult.scaledNodes[i],curResult.scaledNodes[i+1]),cut,maxdepth,depth+1);
      integral += locres.integral;
      error    += locres.error;
    }

    return ResultClass(integral, error, curResult.scaledNodes);
        
  }//LocalAdapStratRecur

  /*!
    Local Adaptative Strategy
   */
  values_t LocalAdapStrat(T tol = (T)1.0e-6, 
			  size_t MaxNumRecurssion = 1000
			  ) {
    
    bounds_t curBounds = initBounds_; // {a,b}
    ResultClass curResult =  IntEval(curBounds); //{int, err, nodes}
    
    T integral = curResult.integral;
    T error    = curResult.error;
   
    T cut =  tol*fabs((double)integral);
    if (error <= cut) return make_pair(curResult.integral, curResult.error);  //Mathematica exemple T cut =  50*tol*fabs((double)integral);
    
    ResultClass tmp= LocalAdapStratRecur(curBounds,cut,MaxNumRecurssion);
    
    return make_pair(tmp.integral, tmp.error);
    
  }//LocalAdapStrat

   /*!
     Global Adaptative Strategy of integration to obtain error<tol*integral
     add a maximum number of iterations. 1000 is large enough and so may
     signal a numerical problem.
   */
  values_t GlobalAdapStrat(T tol = (T)1.0e-6, 
			   size_t MaxNumRecurssion = 1000, 
			   size_t MinNumRecurssion = 1) {
    bounds_t curBounds = initBounds_; // {a,b}
    //JEC    values_t curValues = IntEval(curBounds); //{int,err}
    //               T integral = curValues.first;
    //               T error    = curValues.second;
    ResultClass curValues = IntEval(curBounds); //{int,err, nodes}
    T integral = curValues.integral;
    T error    = curValues.error;

    //JEC    result_t curResult = make_pair(curBounds,curValues); //{{a,b}, {int,err}}
    result_t curResult = make_pair(curBounds,make_pair(integral,error)); //{{a,b}, {int,err}}
    
    colresults_t theResults;
    typename colresults_t::const_iterator iRes, iEnd;
    
    theResults.push_back(curResult);
     
    size_t n=0;

    while( ((error >= tol*fabs(integral)) && (n<MaxNumRecurssion)) || 
	   (n<MinNumRecurssion) ) {
      n++; 

      //      cout << "Recursion: " << n << "int: " << integral << " error/int: " << error/fabs(integral) << endl;

      curBounds = theResults.front().first; //{x0,x2}
      T xmiddle = (curBounds.first+curBounds.second)*0.5; //x1
      bounds_t lowerB = make_pair(curBounds.first,xmiddle); //{x0,x1}
      //JEC      values_t valLow = IntEval(lowerB);
      //JEC      result_t resLow = make_pair(lowerB,valLow);
      ResultClass valLow = IntEval(lowerB);
      result_t resLow = make_pair(lowerB,make_pair(valLow.integral,valLow.error));

      bounds_t upperB = make_pair(xmiddle,curBounds.second); //{x1,x2}
      //JEC      values_t valUp = IntEval(upperB);
      //JEC      result_t resUp = make_pair(upperB,valUp);
      ResultClass valUp = IntEval(upperB);
      result_t resUp = make_pair(upperB,make_pair(valUp.integral,valUp.error));

      //replace the first element (the one which has the highest error) 
      //of list of results by the two new results (resLow,resUp
      theResults.pop_front();
      theResults.push_back(resUp);
      theResults.push_back(resLow);

      //sort the list of results
      Comparator compare(this);
      theResults.sort(compare);

       
      integral = 0.;
      error    = 0.;
      iEnd=theResults.end();
      for(iRes=theResults.begin();iRes!=iEnd;++iRes){
	result_t iResCur=*iRes;
	integral += (iResCur.second).first;
	error    += (iResCur.second).second;
      }
       
    }//eod while
     
     //debug
     /*
       cout << "GlobalAdapStrat end: {" << n << ", " << integral << ", " 
       << error << "}" << endl;
       cout << "debug quadinteg: " << endl;
       cout << "{";
       vector<r_8> pts;
       for(iRes=theResults.begin();iRes!=iEnd;++iRes){
       result_t iResCur=*iRes;
       for(size_t i=0; i<npts_;i++){
       bounds_t bnd = iResCur.first;
       pts.push_back(bnd.first + absc_[i]*(bnd.second-bnd.first));
       }
       }
       std::sort(pts.begin(),pts.end());
       for(int i=0;i<pts.size();i++){
       cout << "{" << pts[i] <<"," << (*func_)(pts[i])<<"},"; 
       }
       cout << "};"<<endl;
*/

    return make_pair(integral,error);
  }//GlobalAdapStrat


protected:
    
  string    name_; //!< quadrature name 
  size_t    npts_; //!< number of sampling nodes
  vector<T> absc_; //!< sampling nodes abscissa
  vector<T> absw_; //!< quadrature weights for integral
  vector<T> errw_; //!< quadrature weights for error computation
  ClassFunc1D* func_; //<! the function to integrate
  bounds_t initBounds_; //<! the initial bounds
  
private:   
  /*!
    Explicit prohibit Copy Ctor for the time beeing
  */
  Quadrature(const Quadrature&);
  
  /*!
    Explicit prohibit  Assignment Operator for the time beeing
  */
  Quadrature& operator=(const Quadrature&);
};//end Quadrature
  
//-------------------------------------------------------------------------------------
template <typename T>
class ClenshawCurtisQuadrature : public Quadrature<T, ClenshawCurtisQuadrature<T> > {
public:
  /*!
    Default Ctor
  */
  ClenshawCurtisQuadrature():  
    Quadrature<T,ClenshawCurtisQuadrature<T> >("ClenshawCurtisQuadrature",39,"ClenshawCurtisRuleData-20.txt",false) {
  }//Ctor
    
  /*!
    Ctor used with a different number of points = 2n-1. Should implement ComputeAbsWeights
  */
  ClenshawCurtisQuadrature(size_t n, string fName, bool init):
    Quadrature<T,ClenshawCurtisQuadrature<T> >("ClenshawCurtisQuadrature",2*n-1,fName,init) {}
  

    size_t GetOrder() const {return 2*this->npts_-1;}

  /*!
    Implement the abscissa, weights and err weights in the range (0,1)
  */    
  void ComputeAbsWeights() throw(string) { 
    //code adapted from John Burkart 2009 
    //the abs, weights are for (-1,1) range so rescaling (0,1) is necessary
    //the error weights are adapted from Mathematica prescription
    //Todo: contact him to see how to use his code?


    int order = this->npts_;
    if (order < 3){
      string msg = this->name_ + " ComputeAbsWeights: Too few points ERROR";
      throw(msg);
    }

    this->absc_.resize(order);
    this->absw_.resize(order);
    this->errw_.resize(order);

    //Compute the abcsissa locations and their weights
    ComputeAbsWeights(order,this->absc_,this->absw_);

    //Compute the error weights based on the following properties:
    //the abscissa of computed with n points is a subset of the abscissa computed with 2n-1 points
    // so the difference between the two integral approximations give an error estimate.

    int subOrder = (order+1)/2;
    vector<T> xSub(subOrder);
    vector<T> wSub(subOrder);
    ComputeAbsWeights(subOrder,xSub,wSub);
      

    for(int i=0; i<order;i++){
      this->errw_[i] = (i%2 == 0) ? this->absw_[i] - wSub[i/2] : this->absw_[i];
    }

    //Explicit rescaling (-1,1) -> (0,1)
    this->RescaleAbsWeights();

       
  }//ComputeAbsWeights

  /*!
    ComputeAbsWeights more related to the original code
  */
  void ComputeAbsWeights(int order, vector<T>& x, vector<T>& w) throw(string) {
      
    double b;
    int i;
    int j;
    //  double pi = 3.141592653589793;
    double pi = M_PI; //JEC use cmath definition of PI
    double theta;

    //Todo this kind of error should be tracked before
    if ( order < 1 ) {
      cerr << "\n";
      cerr << "CLENSHAW_CURTIS_COMPUTE - Fatal error!\n";
      cerr << "  Illegal value of ORDER = " << order << "\n";
      exit (EXIT_FAILURE);
    }

 
    if ( order == 1 ) {
      x[0] = 0.0;
      w[0] = 2.0;
    } else {
      for ( i = 0; i < order; i++ ) {
	x[i] =  cos ( ( double ) ( order - 1 - i ) * pi
		      / ( double ) ( order - 1     ) );
      }
      x[0] = -1.0;
      if ( ( order % 2 ) == 1 ) {
	x[(order-1)/2] = 0.0;
      }
      x[order-1] = +1.0;
    
      for ( i = 0; i < order; i++ ) {
	theta = ( double ) ( i ) * pi / ( double ) ( order - 1 );  
	w[i] = 1.0;
      
	for ( j = 1; j <= ( order - 1 ) / 2; j++ ) {
	  if ( 2 * j == ( order - 1 ) ) {
	    b = 1.0;
	  } else {
	    b = 2.0;
	  }
	
	  w[i] = w[i] - b *  cos ( 2.0 * ( double ) ( j ) * theta )
	    / ( double ) ( 4 * j * j - 1 );
	}
      }
    
      w[0] = w[0] / ( double ) ( order - 1 );
      for ( i = 1; i < order - 1; i++ ) {
	w[i] = 2.0 * w[i] / ( double ) ( order - 1 );
      }
      w[order-1] = w[order-1] / ( double ) ( order - 1 );
    }

  }//ComputeAbsWeights

  /*!
    Destructor
   */
  ~ClenshawCurtisQuadrature() {}
}; //ClenshawCurtisQuadrature


//-------------------------------------------------------------------------------------
template <typename T>
class GaussKronrodQuadrature : public Quadrature<T, GaussKronrodQuadrature<T> > {
public:
  GaussKronrodQuadrature():  
    Quadrature<T,GaussKronrodQuadrature<T> >("GaussKronrodQuadrature",41,"GaussKronrodRuleData-20.txt",false) {
  }//Ctor
  
   /*!
     Ctor used with a different number of points = 2n+1. Should implement ComputeAbsWeights
   */
  GaussKronrodQuadrature(size_t n, string fName, bool init):
    Quadrature<T, GaussKronrodQuadrature<T> >("GaussKronrodQuadrature",2*n+1,fName,init) {}
  
    size_t GetOrder() const {return 2*this->npts_+1;}

  /*!
    Implement the abscissa, weights and err weights in the range (0,1)
  */    
    void ComputeAbsWeights()  throw(string) { 
    //code adapted from John Burkart 19/3/2009 
    //the abs, weights are for (-1,1) range so rescaling (0,1) is necessary
    //the error weights are adapted from Mathematica prescription
    //Todo: contact him to see how to use his code?


    int npts = this->npts_;
    this->absc_.resize(npts);
    this->absw_.resize(npts);
    this->errw_.resize(npts);

    int order= (npts-1)/2;
    if (order < 3){
      string msg = this->name_ + " ComputeAbsWeights: Too few points ERROR";
      throw(msg);
    }      
    double* x  = new double[order+1];
    double* w1 = new double[order+1];
    double* w2 = new double[order+1];
    double eps = 0.000001;

    kronrod(order, eps, x , w1, w2);
      
    for (int i=0;i<(order+1);i++){
//       cout << "["<<i<<"]"<<fixed << showpoint <<  setprecision(6) << x[i] << "\t" << w1[i] << "\t" << w2[i] << endl;

      this->absc_[i] = -x[i];
      this->absw_[i] = w1[i];
      this->errw_[i] = w1[i]-w2[i];

      int j=npts-1-i;
      this->absc_[j] = x[i];
      this->absw_[j] = this->absw_[i];
      this->errw_[j] = this->errw_[i];	
    }
      

    delete [] w2;
    delete [] w1;
    delete [] x;


    //Explicit rescaling (-1,1) -> (0,1)
    this->RescaleAbsWeights();
      
  
  }//ComputeAbsWeights


  //****************************************************************************80

  void abwe1 ( int n, int m, double eps, double coef2, bool even, double b[], 
	       double *x, double *w )
    
    //****************************************************************************80
    //
    //  Purpose:
    //
    //    ABWE1 calculates a Kronrod abscissa and weight.
    //
    //  Licensing:
    //
    //    This code is distributed under the GNU LGPL license.
    //
    //  Modified:
    //
    //    03 August 2010
    //
    //  Author:
    //
    //    Original FORTRAN77 version by Robert Piessens, Maria Branders.
    //    C++ version by John Burkardt.
    //
    //  Reference:
    //
    //    Robert Piessens, Maria Branders,
    //    A Note on the Optimal Addition of Abscissas to Quadrature Formulas
    //    of Gauss and Lobatto,
    //    Mathematics of Computation,
    //    Volume 28, Number 125, January 1974, pages 135-139.
    //
    //  Parameters:
    //
    //    Input, int N, the order of the Gauss rule.
    //
    //    Input, int M, the value of ( N + 1 ) / 2.
    //
    //    Input, double EPS, the requested absolute accuracy of the
    //    abscissas.
    //
    //    Input, double COEF2, a value needed to compute weights.
    //
    //    Input, bool EVEN, is TRUE if N is even.
    //
    //    Input, double B[M+1], the Chebyshev coefficients.
    //
    //    Input/output, double *X; on input, an estimate for
    //    the abscissa, and on output, the computed abscissa.
    //
    //    Output, double *W, the weight.
    //
  {
    double ai;
    double b0=0; //JEC avoid uninitialisation warning
    double b1;
    double b2;
    double d0;
    double d1;
    double d2;
    double delta;
    double dif;
    double f;
    double fd;
    int i;
    int iter;
    int k;
    int ka;
    double yy;

    if ( *x == 0.0 )
      {
	ka = 1;
      }
    else
      {
	ka = 0;
      }
    //
    //  Iterative process for the computation of a Kronrod abscissa.
    //
    for ( iter = 1; iter <= 50; iter++ )
      {
	b1 = 0.0;
	b2 = b[m];
	yy = 4.0 * (*x) * (*x) - 2.0;
	d1 = 0.0;

	if ( even )
	  {
	    ai = m + m + 1;
	    d2 = ai * b[m];
	    dif = 2.0;
	  }
	else
	  {
	    ai = m + 1;
	    d2 = 0.0;
	    dif = 1.0;
	  }

	for ( k = 1; k <= m; k++ )
	  {
	    ai = ai - dif;
	    i = m - k + 1;
	    b0 = b1;
	    b1 = b2;
	    d0 = d1;
	    d1 = d2;
	    b2 = yy * b1 - b0 + b[i-1];
	    if ( !even )
	      {
		i = i + 1;
	      }
	    d2 = yy * d1 - d0 + ai * b[i-1];
	  }

	if ( even )
	  {
	    f = ( *x ) * ( b2 - b1 );
	    fd = d2 + d1;
	  }
	else
	  {
	    f = 0.5 * ( b2 - b0 );
	    fd = 4.0 * ( *x ) * d2;
	  }
	//
	//  Newton correction.
	//
	delta = f / fd;
	*x = *x - delta;

	if ( ka == 1 )
	  {
	    break;
	  }

	if ( r8_abs ( delta ) <= eps )
	  {
	    ka = 1;
	  }
      }
    //
    //  Catch non-convergence.
    //
    if ( ka != 1 )
      {
	cout << "\n";
	cout << "ABWE1 - Fatal error!\n";
	cout << "  Iteration limit reached.\n";
	cout << "  EPS is " << eps << "\n";
	cout << "  Last DELTA was " << delta << "\n";
	exit ( 1 );
      }
    //
    //  Computation of the weight.
    //
    d0 = 1.0;
    d1 = *x;
    ai = 0.0;
    for ( k = 2; k <= n; k++ )
      {
	ai = ai + 1.0;
	d2 = ( ( ai + ai + 1.0 ) * ( *x ) * d1 - ai * d0 ) / ( ai + 1.0 );
	d0 = d1;
	d1 = d2;
      }

    *w = coef2 / ( fd * d2 );

    return;
  }
  //****************************************************************************80

  void abwe2 ( int n, int m, double eps, double coef2, bool even, double b[], 
	       double *x, double *w1, double *w2 )

    //****************************************************************************80
    //
    //  Purpose:
    //
    //    ABWE2 calculates a Gaussian abscissa and two weights.
    //
    //  Licensing:
    //
    //    This code is distributed under the GNU LGPL license.
    //
    //  Modified:
    //
    //    30 April 2013
    //
    //  Author:
    //
    //    Original FORTRAN77 version by Robert Piessens, Maria Branders.
    //    C++ version by John Burkardt.
    //
    //  Reference:
    //
    //    Robert Piessens, Maria Branders,
    //    A Note on the Optimal Addition of Abscissas to Quadrature Formulas
    //    of Gauss and Lobatto,
    //    Mathematics of Computation,
    //    Volume 28, Number 125, January 1974, pages 135-139.
    //
    //  Parameters:
    //
    //    Input, int N, the order of the Gauss rule.
    //
    //    Input, int M, the value of ( N + 1 ) / 2.
    //
    //    Input, double EPS, the requested absolute accuracy of the
    //    abscissas.
    //
    //    Input, double COEF2, a value needed to compute weights.
    //
    //    Input, bool EVEN, is TRUE if N is even.
    //
    //    Input, double B[M+1], the Chebyshev coefficients.
    //
    //    Input/output, double *X; on input, an estimate for
    //    the abscissa, and on output, the computed abscissa.
    //
    //    Output, double *W1, the Gauss-Kronrod weight.
    //
    //    Output, double *W2, the Gauss weight.
    //
  {
    double ai;
    double an;
    double delta;
    int i;
    int iter;
    int k;
    int ka;
    double p0;
    double p1;
    double p2;
    double pd0;
    double pd1;
    double pd2;
    double yy;

    if ( *x == 0.0 )
      {
	ka = 1;
      }
    else
      {
	ka = 0;
      }
    //
    //  Iterative process for the computation of a Gaussian abscissa.
    //
    for ( iter = 1; iter <= 50; iter++ )
      {
	p0 = 1.0;
	p1 = *x;
	pd0 = 0.0;
	pd1 = 1.0;
	//
	//  When N is 1, we need to initialize P2 and PD2 to avoid problems with DELTA.
	//
	if ( n <= 1 )
	  {
	    if ( r8_epsilon ( ) < r8_abs ( *x ) )
	      {
		p2 = ( 3.0 * ( *x ) * ( *x ) - 1.0 ) / 2.0;
		pd2 = 3.0 * ( *x );
	      }
	    else
	      {
		p2 = 3.0 * ( *x );
		pd2 = 3.0;
	      }
	  }

	ai = 0.0;
	for ( k = 2; k <= n; k++ )
	  {
	    ai = ai + 1.0;
	    p2 = ( ( ai + ai + 1.0 ) * (*x) * p1 - ai * p0 ) / ( ai + 1.0 );
	    pd2 = ( ( ai + ai + 1.0 ) * ( p1 + (*x) * pd1 ) - ai * pd0 ) 
	      / ( ai + 1.0 );
	    p0 = p1;
	    p1 = p2;
	    pd0 = pd1;
	    pd1 = pd2;
	  }
	//
	//  Newton correction.
	//
	delta = p2 / pd2;
	*x = *x - delta;

	if ( ka == 1 )
	  {
	    break;
	  }

	if ( r8_abs ( delta ) <= eps )
	  {
	    ka = 1;
	  }
      }
    //
    //  Catch non-convergence.
    //
    if ( ka != 1 )
      {
	cout << "\n";
	cout << "ABWE2 - Fatal error!\n";
	cout << "  Iteration limit reached.\n";
	cout << "  EPS is " << eps << "\n";
	cout << "  Last DELTA was " << delta << "\n";
	exit ( 1 );
      }
    //
    //  Computation of the weight.
    //
    an = n;

    *w2 = 2.0 / ( an * pd2 * p0 );

    p1 = 0.0;
    p2 = b[m];
    yy = 4.0 * (*x) * (*x) - 2.0;
    for ( k = 1; k <= m; k++ )
      {
	i = m - k + 1;
	p0 = p1;
	p1 = p2;
	p2 = yy * p1 - p0 + b[i-1];
      }

    if ( even )
      {
	*w1 = *w2 + coef2 / ( pd2 * (*x) * ( p2 - p1 ) );
      }
    else
      {
	*w1 = *w2 + 2.0 * coef2 / ( pd2 * ( p2 - p0 ) );
      }

    return;
  }
  //****************************************************************************80

  void kronrod ( int n, double eps, double x[], double w1[], double w2[] )

    //****************************************************************************80
    //
    //  Purpose:
    //
    //    KRONROD adds N+1 points to an N-point Gaussian rule.
    //
    //  Discussion:
    //
    //    This subroutine calculates the abscissas and weights of the 2N+1
    //    point Gauss Kronrod quadrature formula which is obtained from the 
    //    N point Gauss quadrature formula by the optimal addition of N+1 points.
    //
    //    The optimally added points are called Kronrod abscissas.  The 
    //    abscissas and weights for both the Gauss and Gauss Kronrod rules
    //    are calculated for integration over the interval [-1,+1].
    //
    //    Since the quadrature formula is symmetric with respect to the origin,
    //    only the nonnegative abscissas are calculated.
    //
    //    Note that the code published in Mathematics of Computation 
    //    omitted the definition of the variable which is here called COEF2.
    //
    //  Storage:
    //
    //    Given N, let M = ( N + 1 ) / 2.  
    //
    //    The Gauss-Kronrod rule will include 2*N+1 points.  However, by symmetry,
    //    only N + 1 of them need to be listed.
    //
    //    The arrays X, W1 and W2 contain the nonnegative abscissas in decreasing
    //    order, and the weights of each abscissa in the Gauss-Kronrod and
    //    Gauss rules respectively.  This means that about half the entries
    //    in W2 are zero.
    //
    //    For instance, if N = 3, the output is:
    //
    //    I      X               W1              W2
    //
    //    1    0.960491        0.104656         0.000000   
    //    2    0.774597        0.268488         0.555556    
    //    3    0.434244        0.401397         0.000000
    //    4    0.000000        0.450917         0.888889
    //
    //    and if N = 4, (notice that 0 is now a Kronrod abscissa)
    //    the output is
    //
    //    I      X               W1              W2
    //
    //    1    0.976560        0.062977        0.000000   
    //    2    0.861136        0.170054        0.347855    
    //    3    0.640286        0.266798        0.000000   
    //    4    0.339981        0.326949        0.652145    
    //    5    0.000000        0.346443        0.000000
    //
    //  Licensing:
    //
    //    This code is distributed under the GNU LGPL license.
    //
    //  Modified:
    //
    //    03 August 2010
    //
    //  Author:
    //
    //    Original FORTRAN77 version by Robert Piessens, Maria Branders.
    //    C++ version by John Burkardt.
    //
    //  Reference:
    //
    //    Robert Piessens, Maria Branders,
    //    A Note on the Optimal Addition of Abscissas to Quadrature Formulas
    //    of Gauss and Lobatto,
    //    Mathematics of Computation,
    //    Volume 28, Number 125, January 1974, pages 135-139.
    //
    //  Parameters:
    //
    //    Input, int N, the order of the Gauss rule.
    //
    //    Input, double EPS, the requested absolute accuracy of the
    //    abscissas.
    //
    //    Output, double X[N+1], the abscissas.
    //
    //    Output, double W1[N+1], the weights for the Gauss-Kronrod rule.
    //
    //    Output, double W2[N+1], the weights for 
    //    the Gauss rule.
    //
  {
    double ak;
    double an;
    double *b;
    double bb;
    double c;
    double coef;
    double coef2;
    double d;
    bool even;
    int i;
    int k;
    int l;
    int ll;
    int m;
    double s;
    double *tau;
    double x1;
    double xx;
    double y;

    b = new double[((n+1)/2)+1];
    tau = new double[(n+1)/2];
  
    m = ( n + 1 ) / 2;
    even = ( 2 * m == n );

    d = 2.0;
    an = 0.0;
    for ( k = 1; k <= n; k++ )
      {
	an = an + 1.0;
	d = d * an / ( an + 0.5 );
      }
    //
    //  Calculation of the Chebyshev coefficients of the orthogonal polynomial.
    //
    tau[0] = ( an + 2.0 ) / ( an + an + 3.0 );
    b[m-1] = tau[0] - 1.0;
    ak = an;

    for ( l = 1; l < m; l++ )
      {
	ak = ak + 2.0;
	tau[l] = ( ( ak - 1.0 ) * ak 
		   - an * ( an + 1.0 ) ) * ( ak + 2.0 ) * tau[l-1] 
	  / ( ak * ( ( ak + 3.0 ) * ( ak + 2.0 ) 
		     - an * ( an + 1.0 ) ) );
	b[m-l-1] = tau[l];

	for ( ll = 1; ll <= l; ll++ )
	  {
	    b[m-l-1] = b[m-l-1] + tau[ll-1] * b[m-l+ll-1];
	  }
      }

    b[m] = 1.0;
    //
    //  Calculation of approximate values for the abscissas.
    //
    //    bb = sin ( 1.570796 / ( an + an + 1.0 ) ); //JEC use M_PI_2
    bb = sin ( M_PI_2 / ( an + an + 1.0 ) );
    x1 = sqrt ( 1.0 - bb * bb );
    s = 2.0 * bb * x1;
    c = sqrt ( 1.0 - s * s );
    coef = 1.0 - ( 1.0 - 1.0 / an ) / ( 8.0 * an * an );
    xx = coef * x1;
    //
    //  Coefficient needed for weights.
    //
    //  COEF2 = 2^(2*n+1) * n// * n// / (2n+1)//
    //
    coef2 = 2.0 / ( double ) ( 2 * n + 1 );
    for ( i = 1; i <= n; i++ )
      {
	coef2 = coef2 * 4.0 * ( double ) ( i ) / ( double ) ( n + i );
      }
    //
    //  Calculation of the K-th abscissa (a Kronrod abscissa) and the
    //  corresponding weight.
    //
    for ( k = 1; k <= n; k = k + 2 )
      {
	abwe1 ( n, m, eps, coef2, even, b, &xx, w1+k-1 );
	w2[k-1] = 0.0;

	x[k-1] = xx;
	y = x1;
	x1 = y * c - bb * s;
	bb = y * s + bb * c;

	if ( k == n )
	  {
	    xx = 0.0;
	  }
	else
	  {
	    xx = coef * x1;
	  }
	//
	//  Calculation of the K+1 abscissa (a Gaussian abscissa) and the
	//  corresponding weights.
	//
	abwe2 ( n, m, eps, coef2, even, b, &xx, w1+k, w2+k );

	x[k] = xx;
	y = x1;
	x1 = y * c - bb * s;
	bb = y * s + bb * c;
	xx = coef * x1;
      }
    //
    //  If N is even, we have one more Kronrod abscissa to compute,
    //  namely the origin.
    //
    if ( even )
      {
	xx = 0.0;
	abwe1 ( n, m, eps, coef2, even, b, &xx, w1+n );
	w2[n] = 0.0;
	x[n] = xx;
      }

    delete [] b;
    delete [] tau;

    return;
  }
  //****************************************************************************80

  double r8_abs ( double x )

    //****************************************************************************80
    //
    //  Purpose:
    //
    //    R8_ABS returns the absolute value of an R8.
    //
    //  Licensing:
    //
    //    This code is distributed under the GNU LGPL license. 
    //
    //  Modified:
    //
    //    14 November 2006
    //
    //  Author:
    //
    //    John Burkardt
    //
    //  Parameters:
    //
    //    Input, double X, the quantity whose absolute value is desired.
    //
    //    Output, double R8_ABS, the absolute value of X.
    //
  {
//     double value;

//     if ( 0.0 <= x )
//       {
// 	value = x;
//       } 
//     else
//       {
// 	value = - x;
//       }
//     return value;
    return fabs(x); //JEC math.h
  }
  //****************************************************************************80

  double r8_epsilon ( )

    //****************************************************************************80
    //
    //  Purpose:
    //
    //    R8_EPSILON returns the R8 roundoff unit.
    //
    //  Discussion:
    //
    //    The roundoff unit is a number R which is a power of 2 with the
    //    property that, to the precision of the computer's arithmetic,
    //      1 < 1 + R
    //    but
    //      1 = ( 1 + R / 2 )
    //
    //  Licensing:
    //
    //    This code is distributed under the GNU LGPL license.
    //
    //  Modified:
    //
    //    01 September 2012
    //
    //  Author:
    //
    //    John Burkardt
    //
    //  Parameters:
    //
    //    Output, double R8_EPSILON, the R8 round-off unit.
    //
  {
//     static double value = 2.220446049250313E-016;

//     return value;
    return DBL_EPSILON; //JEC (float.h definition)
  }
  //****************************************************************************80

    /*!
      Destructor 
     */  
  ~GaussKronrodQuadrature() {}
}; //GaussKronrodQuadrature

//-------------------------------------------------------------------------------------
template <typename T>
  class GaussBerntsenEspelidQuadrature : public Quadrature<T, GaussBerntsenEspelidQuadrature<T> > {
public:
    /*!
      Default Ctor
    */
    GaussBerntsenEspelidQuadrature():  
      Quadrature<T,GaussBerntsenEspelidQuadrature<T> >("GaussBerntsenEspelidQuadrature",41,"GaussBerntsenEspelidRuleData-20.txt",false) {
    }//Ctor
    
    
    /*!
      Ctor used with a different number of points = 2n+1. Should implement ComputeAbsWeights
    */
    GaussBerntsenEspelidQuadrature(size_t n, string fName, bool init):
      Quadrature<T, GaussBerntsenEspelidQuadrature<T> >("GaussBerntsenEspelidQuadrature",2*n+1,fName,init) {}
  

    size_t GetOrder() const {return 2*this->npts_+1;}


    /*!
      Implement the abscissa, weights and err weights in the range (0,1)
    */    
    void ComputeAbsWeights() throw(string) { 
      //code adapted from John Burkart 2009 
      //the abs, weights are for (-1,1) range so rescaling (0,1) is necessary
      //the error weights are adapted from Mathematica prescription
      //Todo: contact him to see how to use his code?
      
      int npts = this->npts_;
      this->absc_.resize(npts);
      this->absw_.resize(npts);
      this->errw_.resize(npts);


      //Compute Abscissa and Weights according to Glaser-Liu-Rokhlin method
      legendre_compute_glr(npts,(this->absc_).data(),(this->absw_).data());

      //Explicit rescaling (-1,1) -> (0,1)
      this->RescaleAbsWeights();

      //Code not adapted form large order (JEC 22/10/16)
//       //Compute error weights according to Berntsen-Espelid formula
//       int n = (npts-1)/2;
//       double dn = (double)n;
//       double cte = (2.0/M_2_SQRTPI)*pow(tgamma(1.0+dn),2.0)/(pow(2.0,2.0*dn)*tgamma(1.5+2.0*dn));
//       if (n%2 == 1) cte *= -1.0;
//       for(int i=0;i<npts;i++){
// 	this->errw_[i] = cte;
// 	for (int j=0; j<npts; j++) {
// 	  if (j==i) continue;
// 	  this->errw_[i] /= this->absc_[i]-this->absc_[j];
// 	}
//       }
      
      
    }// ComputeAbsWeights

    
    void legendre_compute_glr ( int n, double x[], double w[] )

    //****************************************************************************80
    //
    //  Purpose:
    //
    //    LEGENDRE_COMPUTE_GLR: Legendre quadrature by the Glaser-Liu-Rokhlin method.
    //
    //  Licensing:
    //
    //    This code is distributed under the GNU LGPL license. 
    //
    //  Modified:
    //
    //    20 October 2009
    //
    //  Author:
    //
    //    Original C++ version by Nick Hale.
    //    This C++ version by John Burkardt.
    //
    //  Reference:
    //
    //    Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin, 
    //    A fast algorithm for the calculation of the roots of special functions, 
    //    SIAM Journal on Scientific Computing,
    //    Volume 29, Number 4, pages 1420-1438, 2007.
    //
    //  Parameters:
    //
    //    Input, int N, the order.
    //
    //    Output, double X[N], the abscissas.
    //
    //    Output, double W[N], the weights.
    //
    {
      int i;
      double p=0.; //JEC init to avoid compilo warning
      double pp=0.;//JEC
      double w_sum;
      //
      //  Get the value and derivative of the N-th Legendre polynomial at 0.
      //
      legendre_compute_glr0 ( n, &p, &pp );
      //
      //  If N is odd, then zero is a root.
      //  
      if ( n % 2 == 1 )
	{
	  x[(n-1)/2] = p;
	  w[(n-1)/2] = pp;
	}
      //
      //  If N is even, we have to call a function to find the first root.
      //
      else
	{
	  legendre_compute_glr2 ( p, n, &x[n/2], &w[n/2] );
	}
      //
      //  Get the complete set of roots and derivatives.
      //
      legendre_compute_glr1 ( n, x, w );
      //
      //  Compute the W.
      //
      for ( i = 0; i < n; i++ )
	{
	  w[i] = 2.0 / ( 1.0 - x[i] ) / ( 1.0 + x[i] ) / w[i] / w[i];
	}
      w_sum = 0.0;
      for ( i = 0; i < n; i++ )
	{
	  w_sum = w_sum + w[i];
	}
      for ( i = 0; i < n; i++ )
	{
	  w[i] = 2.0 * w[i] / w_sum;
	}
      return;
    }
    //****************************************************************************80

    void legendre_compute_glr0 ( int n, double *p, double *pp )

      //****************************************************************************80
      //
      //  Purpose:
      //
      //    LEGENDRE_COMPUTE_GLR0 gets a starting value for the fast algorithm.
      //
      //  Licensing:
      //
      //    This code is distributed under the GNU LGPL license. 
      //
      //  Modified:
      //
      //    19 October 2009
      //
      //  Author:
      //
      //    Original C++ version by Nick Hale.
      //    This C++ version by John Burkardt.
      //
      //  Reference:
      //
      //    Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin, 
      //    A fast algorithm for the calculation of the roots of special functions, 
      //    SIAM Journal on Scientific Computing,
      //    Volume 29, Number 4, pages 1420-1438, 2007.
      //
      //  Parameters:
      //
      //    Input, int N, the order of the Legendre polynomial.
      //
      //    Output, double *P, *PP, the value of the N-th Legendre polynomial
      //    and its derivative at 0.
      //
    {
      double dk;
      int k;
      double pm1;
      double pm2;
      double ppm1;
      double ppm2;

      pm2 = 0.0;
      pm1 = 1.0;
      ppm2 = 0.0;
      ppm1 = 0.0;

      for ( k = 0; k < n; k++)
	{
	  dk = ( double ) k;
	  *p = - dk * pm2 / ( dk + 1.0 );
	  *pp = ( ( 2.0 * dk + 1.0 ) * pm1 - dk * ppm2 ) / ( dk + 1.0 );
	  pm2 = pm1;
	  pm1 = *p;
	  ppm2 = ppm1;
	  ppm1 = *pp;
	}
      return;
    }
    //****************************************************************************80

    void legendre_compute_glr1 ( int n, double *x, double *w )

      //****************************************************************************80
      //
      //  Purpose:
      //
      //    LEGENDRE_COMPUTE_GLR1 gets the complete set of Legendre points and weights.
      //
      //  Discussion:
      //
      //    This routine requires that a starting estimate be provided for one
      //    root and its derivative.  This information will be stored in entry
      //    (N+1)/2 if N is odd, or N/2 if N is even, of X and W.
      //
      //  Licensing:
      //
      //    This code is distributed under the GNU LGPL license. 
      //
      //  Modified:
      //
      //    19 October 2009
      //
      //  Author:
      //
      //    Original C++ version by Nick Hale.
      //    This C++ version by John Burkardt.
      //
      //  Reference:
      //
      //    Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin, 
      //    A fast algorithm for the calculation of the roots of special functions, 
      //    SIAM Journal on Scientific Computing,
      //    Volume 29, Number 4, pages 1420-1438, 2007.
      //
      //  Parameters:
      //
      //    Input, int N, the order of the Legendre polynomial.
      //
      //    Input/output, double X[N].  On input, a starting value
      //    has been set in one entry.  On output, the roots of the Legendre 
      //    polynomial.
      //
      //    Input/output, double W[N].  On input, a starting value
      //    has been set in one entry.  On output, the derivatives of the Legendre 
      //    polynomial at the zeros.
      //
      //  Local Parameters:
      //
      //    Local, int M, the number of terms in the Taylor expansion.
      //
    {
      double dk;
      double dn;
      double h;
      int j;
      int k;
      int l;
      int m = 30;
      int n2;
//       static double pi = 3.141592653589793;
      double pi = M_PI; //JEC
      int s;
      double *u;
      double *up;
      double xp;

      if ( n % 2 == 1 )
	{
	  n2 = ( n - 1 ) / 2 - 1;
	  s = 1;
	}
      else
	{
	  n2 = n / 2 - 1;
	  s = 0;
	}

      u = new double[m+2];
      up = new double[m+1];

      dn = ( double ) n;

      for ( j = n2 + 1; j < n - 1; j++ )
	{
	  xp = x[j];

	  h = rk2_leg ( pi/2.0, -pi/2.0, xp, n ) - xp;

	  u[0] = 0.0;
	  u[1] = 0.0;
	  u[2] = w[j];

	  up[0] = 0.0;
	  up[1] = u[2];

	  for ( k = 0; k <= m - 2; k++ )
	    {
	      dk = ( double ) k;

	      u[k+3] = 
		( 
		 2.0 * xp * ( dk + 1.0 ) * u[k+2]
		 + ( dk * ( dk + 1.0 ) - dn * ( dn + 1.0 ) ) * u[k+1] / ( dk + 1.0 ) 
		 ) / ( 1.0 - xp ) / ( 1.0 + xp ) / ( dk + 2.0 );

	      up[k+2] = ( dk + 2.0 ) * u[k+3];
	    }

	  for ( l = 0; l < 5; l++ )
	    { 
	      h = h - ts_mult ( u, h, m ) / ts_mult ( up, h, m-1 );
	    }

	  x[j+1] = xp + h;
	  w[j+1] = ts_mult ( up, h, m - 1 );    
	}

      for ( k = 0; k <= n2 + s; k++ )
	{
	  x[k] = - x[n-1-k];
	  w[k] = w[n-1-k];
	}
      return;
    }
    //****************************************************************************80

    void legendre_compute_glr2 ( double pn0, int n, double *x1, double *d1 )

      //****************************************************************************80
      //
      //  Purpose:
      //
      //    LEGENDRE_COMPUTE_GLR2 finds the first real root.
      //
      //  Discussion:
      //
      //    This function is only called if N is even.
      //
      //  Licensing:
      //
      //    This code is distributed under the GNU LGPL license. 
      //
      //  Modified:
      //
      //    19 October 2009
      //
      //  Author:
      //
      //    Original C++ version by Nick Hale.
      //    This C++ version by John Burkardt.
      //
      //  Reference:
      //
      //    Andreas Glaser, Xiangtao Liu, Vladimir Rokhlin, 
      //    A fast algorithm for the calculation of the roots of special functions, 
      //    SIAM Journal on Scientific Computing,
      //    Volume 29, Number 4, pages 1420-1438, 2007.
      //
      //  Parameters:
      //
      //    Input, double PN0, the value of the N-th Legendre polynomial
      //    at 0.
      //
      //    Input, int N, the order of the Legendre polynomial.
      //
      //    Output, double *X1, the first real root.
      //
      //    Output, double *D1, the derivative at X1.
      //
      //  Local Parameters:
      //
      //    Local, int M, the number of terms in the Taylor expansion.
      //
    {
      double dk;
      double dn;
      int k;
      int l;
      int m = 30;
//       static double pi = 3.141592653589793;
      double pi = M_PI; //JEC 
      double t;
      double *u;
      double *up;

      t = 0.0;
      *x1 = rk2_leg ( t, -pi/2.0, 0.0, n );

      u = new double[m+2];
      up = new double[m+1];

      dn = ( double ) n;
      //
      //  U[0] and UP[0] are never used.
      //  U[M+1] is set, but not used, and UP[M] is set and not used.
      //  What gives?
      //
      u[0] = 0.0;
      u[1] = pn0;

      up[0] = 0.0;

      for ( k = 0; k <= m - 2; k = k + 2 )
	{
	  dk = ( double ) k;

	  u[k+2] = 0.0;
	  u[k+3] = ( dk * ( dk + 1.0 ) - dn * ( dn + 1.0 ) ) * u[k+1]
	    / (dk + 1.0) / (dk + 2.0 ); 

	  up[k+1] = 0.0;
	  up[k+2] = ( dk + 2.0 ) * u[k+3];
	}
  
      for ( l = 0; l < 5; l++ )
	{
	  *x1 = *x1 - ts_mult ( u, *x1, m ) / ts_mult ( up, *x1, m-1 );
	}
      *d1 = ts_mult ( up, *x1, m-1 );

      return;
    }
    //****************************************************************************80


    double rk2_leg ( double t1, double t2, double x, int n )

      //****************************************************************************80
      //
      //  Purpose:
      //
      //    RK2_LEG advances the value of X(T) using a Runge-Kutta method.
      //
      //  Licensing:
      //
      //    This code is distributed under the GNU LGPL license. 
      //
      //  Modified:
      //
      //    22 October 2009
      //
      //  Author:
      //
      //    Original C++ version by Nick Hale.
      //    This C++ version by John Burkardt.
      //
      //  Parameters:
      //
      //    Input, double T1, T2, the range of the integration interval.
      //
      //    Input, double X, the value of X at T1.
      //
      //    Input, int N, the number of steps to take.
      //
      //    Output, double RK2_LEG, the value of X at T2.
      //
    {
      double f;
      double h;
      int j;
      double k1;
      double k2;
      int m = 10;
      double snn1;
      double t;

      h = ( t2 - t1 ) / ( double ) m;
      snn1 = sqrt ( ( double ) ( n * ( n + 1 ) ) );
      t = t1;

      for ( j = 0; j < m; j++ )
	{
	  f = ( 1.0 - x ) * ( 1.0 + x );
	  k1 = - h * f / ( snn1 * sqrt ( f ) - 0.5 * x * sin ( 2.0 * t ) );
	  x = x + k1;

	  t = t + h;

	  f = ( 1.0 - x ) * ( 1.0 + x );
	  k2 = - h * f / ( snn1 * sqrt ( f ) - 0.5 * x * sin ( 2.0 * t ) );
	  x = x + 0.5 * ( k2 - k1 );
	}
      return x;
    }
    //****************************************************************************80


    double ts_mult ( double *u, double h, int n )

      //****************************************************************************80
      //
      //  Purpose:
      //
      //    TS_MULT evaluates a polynomial.
      //
      //  Licensing:
      //
      //    This code is distributed under the GNU LGPL license. 
      //
      //  Modified:
      //
      //    17 May 2013
      //
      //  Author:
      //
      //    Original C++ version by Nick Hale.
      //    This C++ version by John Burkardt.
      //
      //  Parameters:
      //
      //    Input, double U[N+1], the polynomial coefficients.
      //    U[0] is ignored.
      //
      //    Input, double H, the polynomial argument.
      //
      //    Input, int N, the number of terms to compute.
      //
      //    Output, double TS_MULT, the value of the polynomial.
      //
    {
      //JEC: notice that a Horner method is better

      double b = u[n];
      for(int i=n-1;i>=1;i--){
	b = u[i] + b*h;
      }
      return b;
      

      //       double hk;
      //       int k;
      //       double ts;
      
      //       ts = 0.0;
      //       hk = 1.0;
      //       for ( k = 1; k<= n; k++ )
      // 	{
      // 	  ts = ts + u[k] * hk;
      // 	  hk = hk * h;
      // 	}
      //       return ts;
    }
    //****************************************************************************80


      /*!
      Destructor
      */
  ~GaussBerntsenEspelidQuadrature() {}
}; //GaussBerntsenEspelidQuadrature
  



//-------------------------------------------------------------------------------------
// Gauss-Lobatto quadrature not yet completly available: the error weights are not known.
// JEC 25/10/13
//
// template <typename T>
// class GaussLobattoQuadrature : public Quadrature<T, GaussLobattoQuadrature<T> > {
// public:
//   GaussLobattoQuadrature():  
//     Quadrature<T,GaussLobattoQuadrature<T> >("GaussLobattoQuadrature",39,"GaussLobattoRuleData-20.txt",false) {
//   }//Ctor
  
//   /*
//     Ctor used with a different number of points = 2n-1. Should implement ComputeAbsWeights
//   */
//   GaussLobattoQuadrature(size_t n, string fName, bool init):
//     Quadrature<T,GaussLobattoQuadrature<T> >("GaussLobattoQuadrature",2*n-1,fName,init) {}
  
//   /*
//     Implement the abscissa, weights and err weights in the range (0,1)
//   */    
//   void ComputeAbsWeights(string fName) throw(string) { 
//     //Code by D. Martin 2005
    
    
//     int npts=this->npts_;
//     this->absc_.resize(npts);
//     this->absw_.resize(npts);
//     this->errw_.resize(npts);


//     int n= (npts +1)/2;
//     vector<double> x(n);
//     vector<double> w(n);
//     GaussLobattoRuleComputed(this->npts_,x,w);
    
//     for (int i=1; i<n;i++){//Apparently starts at 1 !!!
//       cout << "["<<i<<"]: " << x[i] << "\t" << w[i] << endl;
//     }
 


//   }//ComputeAbsWeights

//   /*
//     Destructor
//    */
//   ~GaussLobattoQuadrature() {}


// private:  
//   typedef double real_t; 
//   typedef unsigned long int number_t;
  
//   void GaussLobattoRuleComputed(const number_t n,
// 				vector<real_t>& points, 
// 				vector<real_t>& weights) {

//     real_t theTolerance_ = 0.0000001; //JEC guess
//     real_t pi = M_PI; // JEC

//     number_t nhalf(static_cast<number_t>((n+1)/2)), nj(static_cast<number_t>(nhalf-1));
//     real_t x_j(0.), x1(0.), Two_x, Two_k_1x;
    
//     const int n1 = n-1;
//     for ( number_t j = 1; j < nhalf; j++, nj--)
//       {
// 	real_t p_n, p_n1, p_n2;
// 	//Starting point of Newton algorithm for j-th root
// 	x_j = cos (pi*(j+.5)/(n+0.5));
// 	do {
// 	  Two_x = x_j+x_j;
// 	  Two_k_1x = x_j;
// 	  p_n = 1.; p_n1 = 0.; p_n2 = 0.;
	  
//           //compute P_{n-3}, P_{n-2}, P_{n-1} by recursion :
//           //k*P_k(x) = (2*k-1)*x P_{k-1}(x) - (k-1) P_{k-2}(x) , k <= n+1
// 	  for ( number_t k = 1; k < n ; k++)
// 	    {
// 	      p_n2 = p_n1; p_n1 = p_n;
// 	      p_n = ( Two_k_1x * p_n1 - (k-1.) * p_n2 ) / k;
// 	      Two_k_1x += Two_x;
// 	    }
	  
// 	  //        compute (1-x^2)*P'_{n-1} and (1-x^2)*P'_{n-2} for x_j :  */
// 	  //         (1-x^2) P'_k(x) = - k x P_k(x) + k P_{k-1}(x) */
// 	  real_t pp_n =      n1*(p_n1 - x_j*p_n);
// 	  real_t pp_n1 = (n1-1.)*(p_n2 - x_j*p_n1);
// 	  //       compute (1-x^2)^2*P''_{n-1} for x_j */
// 	  //  	 (1-x^2)*P''_k(x) = (2-k) x P'_k(x) + k { P'_{k-1}(x) - P_k(x) } */
	  
// 	  real_t pp2_n = ( (2.-n1)*x_j*pp_n + n1*pp_n1) / (1.-x_j*x_j) - n1*p_n;
// 	  x1 = x_j;
	  
// 	  //          Newton iteration update
// 	  x_j = x1 - pp_n / pp2_n ;
	  
// 	} while ( abs(x_j - x1) > theTolerance_);
        
// 	points[nj]  = x_j;
// 	weights[nj] = 2./( n*(n-1)*p_n*p_n );
//       }
//   }//GaussLobattoRuleComputed

  
// }; //GaussLobattoQuadrature


//-------------------------------------------------------------------------------------

template <typename T>
class LobattoKronrodQuadrature : public Quadrature<T, LobattoKronrodQuadrature<T> > {
public:
  LobattoKronrodQuadrature():  
    Quadrature<T,LobattoKronrodQuadrature<T> >("LobattoKronrodQuadrature",39,"LobattoKronrodRuleData-20.txt",false) {
  }//Ctor
    
  ~LobattoKronrodQuadrature() {}


  size_t GetOrder() const {
    return (this->npts_%2 == 0) ? 3*this->npts_-2 : 3*this->npts_-3;}

}; //LobattoKronrodQuadrature

  


}//namespace

#endif
