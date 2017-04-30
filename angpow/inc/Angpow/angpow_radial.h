#ifndef ANGPOW_RADIAL_SEEN
#define ANGPOW_RADIAL_SEEN
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

#include <math.h>
#include "angpow_radial_base.h"
#include "angpow_tools.h" 


namespace Angpow {


  //! Dirac-like window
class DiracSelect: public RadSelectBase {
public:
  DiracSelect(r_8 zmean, r_8 zmin, r_8 zmax): 
    RadSelectBase(zmin, zmax), zmean_(zmean) {}
  virtual r_8 operator()(r_8 ) const {
    return 1.; //not used ?
  }
  r_8 GetZval() const { return zmean_;}
private:
  r_8 zmean_;
};//DiracFunc 


  //! Normalized Uniform function (Tophat sharp edges) between zmin, zmax
class RadUniformSelect : public RadSelectBase {
 public:
  RadUniformSelect(r_8 zmin, r_8 zmax): 
    RadSelectBase(zmin, zmax) { dZInv_ = 1./(zmax-zmin); }

  virtual r_8 operator()(r_8 x) const {
    return (x<=zmax_ && x>=zmin_) ? dZInv_ : 0.0;
  }
 private:
  r_8 dZInv_;
};//RadUniformSelect


  /*! Unnormalized tophat function. width stand for half-width
    at half maximum or so. The smooth factor is typically 0.1 
    then the fall down is within for instance at the highest
    edge: mean+width+/-smooth
  */
class RadTopHatSmoothSelect : public RadSelectBase {
 public:
    RadTopHatSmoothSelect(r_8 zmin, r_8 zmax, 
			  r_8 mean, r_8 width, r_8 smooth): 
      RadSelectBase(zmin, zmax), mean_(mean), width_(width), smooth_(smooth) {}

      virtual r_8 operator()(r_8 x) const {
      r_8 val=0; 
      if(x<=zmax_ && x>=zmin_){
	val = 0.5*(1.0-tanh((fabs(x-mean_)-width_)/(smooth_*width_)));
      }
      return val;
  }
 private:
    r_8 mean_;
    r_8 width_;
    r_8 smooth_;
};//RadUniformSelect




  //! Gaussian selection window 
class RadGaussSelect : public RadSelectBase {
 public:
  RadGaussSelect(r_8 zmean, r_8 zsigma, r_8 zmin=0, r_8 zmax=1e30): 
    RadSelectBase(zmin, zmax), 
    zmean_(zmean), 
    zsigma_(zsigma) { }

  r_8 GetZMean() const { return zmean_;}
  r_8 GetZSigma() const { return zsigma_;}

  virtual r_8 operator()(r_8 x) const {
    if (x<=zmax_ && x>=zmin_) {
      r_8 arg = (x-zmean_)/zsigma_;
      arg *= arg;
      return exp(-0.5*arg)/(sqrt(2*M_PI)*zsigma_);
    } else {
      return 0.;
    }
  }

 private:
  r_8 zmean_;
  r_8 zsigma_;
};//RadGaussSelect

  //! Test Gauss(z) * dN/dz  with dN/dz = (z/0.55)^2 * Exp[-(z/0.55)^1.5] there is a typo in arXiv:1307.1459v4
class RadModifiedGaussSelect : public RadGaussSelect {
public:
  RadModifiedGaussSelect(r_8 zmean, r_8 zsigma, r_8 zmin=0, r_8 zmax=1e30):
    RadGaussSelect(zmean,zsigma,zmin,zmax) {}
  virtual r_8 operator()(r_8 x) const {
    r_8 val = RadGaussSelect::operator()(x);
    r_8 tmp = x/0.55;
    return val*tmp*tmp*exp(-pow(tmp,1.5));
  }
};//RadModifiedGaussSelect


   //! Selection window with array input 
class RadArraySelect : public RadSelectBase {
 public:
 RadArraySelect(int nz, r_8* z, r_8* n): 
  RadSelectBase(z[0], z[nz-1]), nz_(nz), z_(z), n_(n) {
      std::vector<r_8> vz(nz);
      std::vector<r_8> vn(nz);
      for(int i=0; i<nz; i++){
	vz[i]=z[i];
	vn[i]=n[i];
      }
      nfunc_.DefinePoints(vz,vn);
    }

  int GetNZ() const { return nz_;}
  r_8* GetZ() const { return z_;}
  r_8* GetN() const { return n_;}

  virtual r_8 operator()(r_8 z) const {
    return nfunc_.YInterp(z);
  }

 private:
  int nz_;
  r_8*  z_;
  r_8*  n_;
  SLinInterp1D  nfunc_; //!< linear interpolation w(z)
};//RadArraySelect



}//end namespace
#endif //ANGPOW_RADIAL_SEEN


