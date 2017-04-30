#include "Angpow/angpow_clbase.h"
#include "Angpow/angpow_tools.h" //generate the ell-list & CSpline
#include "Angpow/angpow_exceptions.h"

#include <iostream> //debug
#include <numeric>  //iota     
namespace Angpow {

  /*!
    Ctor
    \input Lmax the ells are in the range [0, Lmax-1]
    \input l_linstep produces ells with unit step up to this threshold
    \input l_logstep then uses logarithmic stepping
   */
Clbase::Clbase(int Lmax, int l_linstep, r_8 l_logstep): Lmax_(Lmax){

  getLlist(0,Lmax-1, l_linstep, l_logstep, ells_); //warning ell is in [0,Lmax-1] included
  int nels = ells_.size();
  cls_.resize(nels);
  for(int i=0; i<nels;i++) cls_[i].first = ells_[i];
  
  ellsAll_.resize(Lmax);
  std::iota(std::begin(ellsAll_),std::end(ellsAll_),0); //0, 1,..., Lmax-1
  
  maximal_ = false;
  if(ells_ == ellsAll_) maximal_ = true;
  
}//Ctor

  /*!
    Interpolation of the Cls if the pre-computed ones are not produced
    with all ells filling the range [0, Lmax-1]. Use spline interpolation
    with automatic computation of the derivatives at both ends.
   */
  
void Clbase::Interpolate() {
  
  if(maximal_) return; //nothing to do

  //prepare Spline input
  int nels = cls_.size();
  std::vector<r_8> x(nels);
  std::vector<r_8> y(nels);
  for(int i=0; i<nels; i++){
    x[i]=cls_[i].first;  y[i]=cls_[i].second;
  }
  
  CSpline spline(nels, x.data(), y.data(), 0, 0, CSpline::AutoDeriv, false);
  
  //update cls...
  ells_ = ellsAll_;
  maximal_ = true;
  nels = ells_.size();
  if(nels != Lmax_)
    throw AngpowError("Clbase::InterpolCls() strange nels != Lmax");
  cls_.resize(nels);
  for(int i=0; i<nels; i++){
    cls_[i].first = ells_[i];
    cls_[i].second = spline((r_8)ells_[i]);
  }
}

}//namespace
