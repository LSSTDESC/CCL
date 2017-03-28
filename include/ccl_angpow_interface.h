#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// CCL includes
extern "C" {
#include "ccl.h"
}

// Angpow includes
/* #include <iostream> */
/* #include <fstream>  */
/* #include <string>  */
/* #include <vector> */
/* #include <numeric>  */
/* #include <math.h> */


/* #include "angpow_numbers.h" */
/* #include "angpow_func.h" */
/* #include "angpow_cosmo.h" */
#include "angpow_tools.h"
#include "angpow_parameters.h"
#include "angpow_pk2cl.h"
#include "angpow_powspec_base.h"
#include "angpow_cosmo_base.h"
#include "angpow_radial.h"
#include "angpow_radial_base.h"
#include "angpow_clbase.h"
#include "angpow_ctheta.h"


// CCL inputs :
// comoving radial distance
// ccl_comoving_radial_distance
// P(k)

// growth ? RSD ? P(k,z) ?
// ccl_growth_factor
// ...
// redshift windows

// Angpow classes to feed
//CosmoCoordBase * coscoord;
//RadSelecBase * Z1win;
//RadSelecBase * Z2win;
//PowerSpecBase * pk;
//Clbase * clout;

// Write new classes PkCCL, CosmoCCL... that inherit from PkBase, CosmoBase

namespace Angpow {

class PowerSpecCCL : public PowerSpecBase {
 public:
  //! Constructor
 PowerSpecCCL(ccl_cosmology * cosmo, double kmin=1e-5, double kmax=10, int nk=1000, bool use_rsd=true) : ccl_cosmo_(cosmo) {
    if(use_rsd) {
      has_rsd_ = true;
      use_rsd_ = true;
    }
    int status =0;
    double * ks = ccl_log_spacing(kmin,kmax,N_K);
    double Pks[N_K];
    double aref = 1.; // z=0
    for(int i=0; i<nk; i++) {
      Pks[i] = ccl_linear_matter_power(cosmo, aref, ks[i], &status);
    }
    
    std::vector<double> vx;
    std::vector<double> vy;
    
    for(int i=0;i<N_K;i++){
      vx.push_back(ks[i]);
      vy.push_back(Pks[i]);
    }
    double vxmin = *(std::min_element(vx.begin(), vx.end()));
    double vxmax = *(std::max_element(vx.begin(), vx.end()));
    Pk_ = new SLinInterp1D(vx,vy,vxmin,vxmax,0);
  }
  //! Destructor
  virtual ~PowerSpecCCL() {}
  
  //! Used to delete explicitly the local pointers
  virtual void ExplicitDestroy() { 
    if(ccl_cosmo_) delete ccl_cosmo_;
    if(Pk_) delete Pk_;
  }

  /*! Explicit to get a clone of the primary object via shallow copy
    using the Copy Ctor
   */
  virtual PowerSpecCCL* clone() const {
    return new PowerSpecCCL(static_cast<const PowerSpecCCL&>(*this));
  }
  
  /*! called by angpow_kinteg.cc to fix the value of some function
    at fixed z value (and l too if necessary)
   */
  void Init(double z) {
    if(use_rsd_) has_rsd_=true;
    int status=0; double bias=1.0;
    double tmp= bias*ccl_growth_factor(ccl_cosmo_,1.0/(1+z), &status);
    growth2_ = tmp*tmp;
    // WARNING: here we want to store dlnD/dln(+1z) = - dlnD/dlna
    fz_= - ccl_growth_rate(ccl_cosmo_,1.0/(1+z), &status);
  }

  //Main operator
  virtual r_8 operator()(double k, double z) {
     return growth2_*(Pk_->operator()(k));
  }


 private:

  SLinInterp1D* Pk_;          //!< access to  Pk(k)
  ccl_cosmology* ccl_cosmo_;   //!< access to CCL cosmology
  double growth2_;               //!< D(zi)^2
  //double fz_; //! growth rate f(z)
  bool use_rsd_;

  //forbid for the time beeing the assignment operator
  PowerSpecCCL& operator=(const PowerSpecCCL& copy);
  
  //Minimal copy to allow Main operator(int, r_8, r_8) to work
  PowerSpecCCL(const PowerSpecCCL& copy) :
  Pk_(copy.Pk_), ccl_cosmo_(copy.ccl_cosmo_), growth2_(copy.growth2_) {}  // , fz_(copy.fz_)
};




class CosmoCoordCCL : public CosmoCoordBase {
public:
  //! Ctor
CosmoCoordCCL(ccl_cosmology * cosmo, double zmin=0., double zmax=9., size_t npts=1000)
  : ccl_cosmo_(cosmo), zmin_(zmin), zmax_(zmax), npts_(npts) 
  {
    int status = 0;
    std::vector<double> vlos(npts_);
    std::vector<double> vz(npts_);
    for(size_t i=0; i<npts_; i++) {
      vz[i]=i*(zmax-zmin)/(double)npts_;
      vlos[i]=ccl_comoving_radial_distance(cosmo, 1.0/(1+vz[i]), &status);
    }
    rofzfunc_.DefinePoints(vz,vlos); //r(z0)
    zofrfunc_.DefinePoints(vlos,vz); //z(r0) = r^{-1}(r0)
  }//Ctor

  //! Dtor
  virtual ~CosmoCoordCCL() {}


  //! r(z): radial comoving distance Mpc
  inline double getLOS(double z) const { return  r(z); } 
  inline virtual double r(double z) const { return rofzfunc_.YInterp(z); } 
  //! z(r): the inverse of radial comoving distance (Mpc)
  inline virtual double z(double r) const { return zofrfunc_.YInterp(r); }
  inline double getInvLOS(double r) const { return z(r); }
  
  //!Hubble Cte
  inline double h() const { return ccl_cosmo_->params.h; }
  //!Hubble function
  inline double Ez(double z) const { int status=0; return ccl_h_over_h0(ccl_cosmo_,1.0/(1+z),&status); }
  //inline double EzMpcm1(double z) const {int status=0; double HL=cosmofunc_.HubbleLengthMpc(); return Ez(z)/HL; }

  //! z=0 matter density 
  inline double OmegaMatter() const { return ccl_cosmo_->params.Omega_m; }
  //! z=0 cosmological constant density 
  inline double OmegaLambda() const { return ccl_cosmo_->params.Omega_l; }

  
    
 protected:
  ccl_cosmology* ccl_cosmo_;  //!< access to CCL cosmology
  //CosmoFuncImp  cosmofunc_;  //!< Cosmological functions
  SLinInterp1D  rofzfunc_; //!< linear interpolation r(z0)
  SLinInterp1D  zofrfunc_; //!< linear interpolation z(r0) = r^{-1}(r0)
  double zmin_;           //!< minimal z
  double zmax_;           //!< maximal z
  size_t npts_;        //!< number of points to define the interpolation in [zmin, zmax]
};// CosmoCoordCCL




 
}//end namespace
