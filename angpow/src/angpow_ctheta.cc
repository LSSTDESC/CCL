#include "Angpow/angpow_ctheta.h"
#include "Angpow/angpow_exceptions.h"
#include<string>
#include <cmath>
#include <iostream> //debug
#include <fstream> //debug
#include <iomanip> //debug
using namespace std;

namespace Angpow {

  /*!
    Ctor from Cl object
   */
  CTheta::CTheta(const Clbase& c,double apod) {
    Clbase cl(c);
    cl.Interpolate(); //in case not interpolated yet

    //check first ell is 0
    if (cl[0].first != 0 ) {
      string msg = "CTheta::CTheta cannot be computed since first multipole is not l=0 0 but l="+cl[0].first;
      throw AngpowError(msg);
    }

    _cl.resize(cl.Size());
    int lmax=_cl.size()-1;


    _p.resize(cl.Size()); //thse will contain lgd polynomails
    _p[0]=1; 

    
    l_apod=lmax*apod;
    double ls2=l_apod*l_apod;

    //keep apodized log cls
    for (size_t l=0;l<_cl.size();l++){
      auto cell=cl[l];
      double lcl=log(2*l+1.)+log(abs(cell.second))-l*(l+1)/ls2;
      _cl[l]=(cell.second>0? exp(lcl) : -exp(lcl));
    }
    
  }

  r_8 CTheta::operator()(const double& t){
    double x=std::cos(t);
    _p[1]=x;
    double c=_cl[0]+_cl[1]*x;
    
    for (size_t l=2;l<_p.size();l++){
      _p[l]=((2*l-1)*x*_p[l-1]-(l-1)*_p[l-2])/l;
      c+=(_cl[l]*_p[l]);
    }
    return c/(4*M_PI);


  }


  ////////////////////////////
  //helper
  void CTheta::WriteApodCls(const std::string & fn) const{

    ofstream f(fn.c_str());
    f << "l\tClsapod"<<endl;
    f << scientific;
    for (size_t l=0;l<_cl.size();l++) f << l << "\t" << _cl[l] << endl;
    f.close();

  }


}//namespace
