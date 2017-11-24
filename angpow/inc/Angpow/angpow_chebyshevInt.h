#ifndef ANGPOW_CHEBYSHEVINT_SEEN
#define ANGPOW_CHEBYSHEVINT_SEEN
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

#include <math.h> //cte numerique
#include <numeric>
#include <vector>
#include <algorithm>


#include <fftw3.h>        //Discrete Cosine Function

#include "angpow_func.h"
#include "angpow_fft.h"


namespace Angpow {

class ChebyshevInt {
 public:
  ChebyshevInt(int ordFunc1, int ordFunc2): 
    ordFunc1_(ordFunc1), ordFunc2_(ordFunc2),
    planFunc1_(0), planFunc2_(0), planInv_(0), planCC_(0)
    {
      Init();
    }
  virtual ~ChebyshevInt() {
    if (planFunc1_) delete planFunc1_; planFunc1_ = 0;
    if (planFunc2_) delete planFunc2_; planFunc2_ = 0;
    if (planInv_)   delete planInv_;   planInv_   = 0;
    if (planCC_)    delete planCC_;    planCC_    = 0;
  }
    
  void Init();


   /*! Perform Sampling & Chebyshev Transform save in vectors for future call to ComputeIntegral(vec,vec)
    */
  void ChebyshevTransform(ClassFunc1D* f1, ClassFunc1D* f2, std::vector<r_8>& vec, r_8 lowBnd, r_8 uppBnd){

     if(f1){
       ChebyshevSampling(vecDCTFunc1_,*f1,lowBnd,uppBnd);
       ChebyshevCoeffFFT(*planFunc1_, vecDCTFunc1_);
       std::fill(vecDCT1Inv_.begin(), vecDCT1Inv_.end(), 0.0);
       std::copy(vecDCTFunc1_.begin(),vecDCTFunc1_.end(),vecDCT1Inv_.begin());
       InverseChebyshevCoeffFFT();
       vec.resize(vecDCT1Inv_.size());
       std::copy(vecDCT1Inv_.begin(), vecDCT1Inv_.end(), vec.begin());

      } else if (f2) {

       ChebyshevSampling(vecDCTFunc2_,*f2,lowBnd,uppBnd);
       ChebyshevCoeffFFT(*planFunc2_, vecDCTFunc2_);
       std::fill(vecDCT1Inv_.begin(), vecDCT1Inv_.end(), 0.0);
       std::copy(vecDCTFunc2_.begin(),vecDCTFunc2_.end(),vecDCT1Inv_.begin());
       InverseChebyshevCoeffFFT();
       vec.resize(vecDCT1Inv_.size());
       std::copy(vecDCT1Inv_.begin(), vecDCT1Inv_.end(), vec.begin());

     } else {
       throw AngpowError("ChebyshevTransform: f1=f2=NULL");
     }
   }//ChebyshevTransform

   /*!
     follow ChebyshevTransform
    */
  r_8 ComputeIntegral(std::vector<r_8>& v1, std::vector<r_8>& v2, r_8 lowBnd, r_8 uppBnd);


  
  //for debug
  std::vector<r_8> GetVecDCT1() const {return  vecDCTFunc1_;}
  std::vector<r_8> GetVecDCT2() const {return  vecDCTFunc2_;}
  

 protected:

  /*! ChebyshevSampling
    Perform function sampling on the Chebyshev points defined in the range [a,b]
    \input f function to be sampled
    \input a lower bound of the range
    \input b upper bound of the range
    \output val  vector (user initialization) result of the sampling
  */

  inline void ChebyshevSampling(std::vector<r_8>& val, const ClassFunc1D& f, r_8 a, r_8 b){
    int n = val.size()-1;

    r_8 bma = 0.5*(b-a); r_8 bpa = 0.5*(b+a);
    r_8 cte = M_PI/((r_8)n);
    
    int k;
    r_8 x;
    //
    //Rq. JEC 7/11/16 in principle one could have defined  val[k]=cos(k*cte)*bma+bpa, then use
    //std::transform to get the final vector. It may be an alternative to openmp once STL will be
    //parallelized
    //
    //#pragma omp parallel for private(x)
    for(k=0;k<=n;k++){
      x = cos(k*cte)*bma+bpa;
      val[k] = f(x);
    }
  }

  
  /*! ChebyshevCoeffFFT
    Compute the Chebyshev transform using FFT
    \input plan the FFTPlanning object that handle the data and the type of transform
    \output val user-initialized vector with the coefficients
  */
   void ChebyshevCoeffFFT(const FFTPlanning& plan, std::vector<r_8>& val){

     int n = val.size()-1;

     plan.Execute();
    //
    //  Chebyshev Coeff. the noramization of FFTW is propto  1/(val.size()-1) = 1/n
    //
     r_8 norm = 1./((r_8)n); 
     std::transform(val.begin(), val.end(), val.begin(), std::bind1st(std::multiplies<r_8>(),norm));
    
     val[0] *= 0.5;
     val[n] *= 0.5;
  }


  
  void InverseChebyshevCoeffFFT(){

    vecDCT1Inv_.front() *= 2.0;
    vecDCT1Inv_.back() *= 2.0;
    
    planInv_->Execute();
    
    std::transform(vecDCT1Inv_.begin(), vecDCT1Inv_.end(), 
		   vecDCT1Inv_.begin(), std::bind1st(std::multiplies<r_8>(),0.5)); 
  }


  /*! ClenshawCurtisWeightsFast
    Determine the weights of the Clenshaw-Curtis quadrature using DCT-I algorithm.
    For the normalization one should take into account that
    FFTW transform 
    Y_k = 2 Sum''_j=0^(N-1) X_j Cos[Pi k j/(N-1)]
    while Clenshaw-Curtis weights are
    W_k = 4/(N-1) a_k  Sum''_{j=0, j even}^(N-1) 1/(1-j^2) Cos[Pi k j/(N-1)]
    
    (nb. Sum'' means that the first and last elements of the sum are devided by 2)
    
    \input plan the FFTPlanning object that handle the data and the type of transform
    \ouput w the weights defined for [-1, 1] quadrature
  */
  void ClenshawCurtisWeightsFast() {
    
    //dim of w is n
    fill(wCC_.begin(), wCC_.end(), (r_8)0.0);
    
    int n = wCC_.size();
    for(int k=0;k<n; k +=2){
      wCC_[k] = 1./(1.-(r_8)(k*k));
    }
    
    planCC_->Execute();
    
    r_8 norm = 2.0/(r_8)(n-1); //2 * FFTW DCT-1 
    std::transform(wCC_.begin(), wCC_.end(), wCC_.begin(), std::bind1st(std::multiplies<r_8>(),norm));
    wCC_[0] /= (r_8)2; wCC_[n-1] /= (r_8)2;
  }


 private:   
  int ordFunc1_; //!< order used for function 1 (see nOrdFunc1_)
  int ordFunc2_; //!< order used for function 2 (see nOrdFunc2_)

  std::vector<r_8> vecDCTFunc1_; //!< vector used by FFTW for function 1
  FFTPlanning*  planFunc1_; //!< FFTW plan for function 1, it uses  vecDCTFunc1_

   std::vector<r_8> vecDCTFunc2_;  //!< vector used by FFTW for function 2
   FFTPlanning* planFunc2_;  //!< FFTW plan for function 2, it uses  vecDCTFunc2_

   std::vector<r_8> vecDCT1Inv_; //!< vector used by FFTW for the inversion of the product
  FFTPlanning* planInv_;  //!< FFTW plan associated to vecDCT1Inv_

  std::vector<r_8> wCC_; //!< vector of the Clenshaw-Curtis weights 
  FFTPlanning*  planCC_; //! FFTW plan associated to wCC_

  int nOrdFunc1_; //!< size of the vector vecDCTFunc1_
  int nOrdFunc2_; //!< size of the vector vecDCTFunc2_
  int nOrdProd_; //!< size of the vector vecDCT1Inv_ and wCC_

};//ChebyshevInt


}//end namespace
#endif //ANGPOW_CHEBYSHEVINT_SEEN
