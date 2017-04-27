#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// CCL includes
extern "C" {
#include "ccl.h"
}

// Angpow includes
#include "Angpow/angpow_tools.h"
#include "Angpow/angpow_parameters.h"
#include "Angpow/angpow_pk2cl.h"
#include "Angpow/angpow_powspec_base.h"
#include "Angpow/angpow_cosmo_base.h"
#include "Angpow/angpow_radial.h"
#include "Angpow/angpow_radial_base.h"
#include "Angpow/angpow_clbase.h"
#include "Angpow/angpow_ctheta.h"
#include "Angpow/angpow_exceptions.h"  //exceptions
#include "Angpow/angpow_integrand_base.h"



//! Here CCL is passed to Angpow base classes
namespace Angpow {

//! Selection window W(z) with spline input from CCL 
class RadSplineSelect : public RadSelectBase {
 public:
 RadSplineSelect(SplPar* spl): 
  RadSelectBase(spl->x0, spl->xf), spl_(spl) {}


  virtual r_8 operator()(r_8 z) const {
    return spline_eval(z,spl_);
  }

 private:
  SplPar* spl_;
};//RadSplineSelect


//! This class import the cosmology from CCL to make the conversion z <-> r(z)
//! and then to compute the cut-off in the integrals
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
  SLinInterp1D  rofzfunc_; //!< linear interpolation r(z0)
  SLinInterp1D  zofrfunc_; //!< linear interpolation z(r0) = r^{-1}(r0)
  double zmin_;           //!< minimal z
  double zmax_;           //!< maximal z
  size_t npts_;        //!< number of points to define the interpolation in [zmin, zmax]
};// CosmoCoordCCL





//! Base class of half the integrand function
//! Basically Angpow does three integrals of the form
//!    C_ell = int dz1 int dz2 int dk f1(ell,k,z1)*f2(ell,k,z2)
//! This class set the f1 and f2 functions from CCL_ClTracer class
//! for galaxy counts :
//!    f(ell,k,z) = k*sqrt(P(k,z))*[ b(z)*j_ell(r(z)*k) + f(z)*j"_ell(r(z)*k) ]
class IntegrandCCL : public IntegrandBase {
 public:
 IntegrandCCL(CCL_ClTracer* clt, ccl_cosmology* cosmo, int ell=0, r_8 z=0.):
  clt_(clt), cosmo_(cosmo), ell_(ell), z_(z) {
    Init(ell,z);
  }
  //! Initialize the function f(ell,k,z) at a given ell and z
  //! to be integrated over k
  void Init(int ell, r_8 z){
    int status=0;
    ell_=ell; z_=z;
    R_ = ccl_comoving_radial_distance(cosmo_, 1.0/(1+z), &status);
    jlR_ = new JBess1(ell_,R_);
    if(clt_->has_rsd) {
      jlp1R_ = new JBess1(ell_+1,R_);
      // WARNING: here we want to store dlnD/dln(+1z) = - dlnD/dlna
      fz_= - ccl_growth_rate(cosmo_,1.0/(1+z), &status);
    }
    bz_ = spline_eval(z,clt_->spl_bz);
  }
  virtual ~IntegrandCCL() {}
  //! Return f(ell,k,z) for a given k
  //! (ell and z must be initialized before) 
  virtual r_8 operator()(r_8 k) const {
    int status=0;
    r_8 Pk = ccl_linear_matter_power(cosmo_, k , 1./(1+z_), &status);
    r_8 x = k*R_;
    r_8 jlRk = (*jlR_)(k);
    r_8 delta = bz_*jlRk; // density term with bias
    if(clt_->has_rsd){ // RSD term
      r_8 jlRksecond = 0.;
      if(x<1e-40) { // compute second derivative j"_ell(r(z)*k)
    	if(ell_==0) {
    	  jlRksecond = -1./3. + x*x/10.;
    	} else if(ell_==2) {
    	  jlRksecond = 2./15. - 2*x*x/35.;
    	} else {
    	  jlRksecond = 0.;
    	}
      } else {
    	jlRksecond = 2.*(*jlp1R_)(k)/x + (ell_*(ell_-1.)/(x*x) - 1.)*jlRk;
      }
      delta += fz_*jlRksecond;
    }
    return(k*sqrt(fabs(Pk))*delta);
  }
  //! Clone function for OpenMP integration
  virtual IntegrandCCL* clone() const {
    return new IntegrandCCL(static_cast<const IntegrandCCL&>(*this));
  }
  virtual void ExplicitDestroy() {
    if(jlR_) delete jlR_;
    if(jlp1R_) delete jlp1R_;
  }
private:
  CCL_ClTracer* clt_;  //no ownership
  ccl_cosmology* cosmo_;  //no ownership
  int ell_;  // multipole ell
  r_8 z_;   // redshift z
  r_8 R_;   // radial comoving distance r(z)
  r_8 fz_;  // growth rate f(z)
  r_8 bz_;  // bias b(z)
  JBess1* jlR_;  // j_ell(k*R)
  JBess1* jlp1R_;   // j_(ell+1)(k*R)

  //Minimal copy to allow Main operator(int, r_8, r_8) to work
  //JEC 22/4/17 use cloning of PowerSpectrum
 IntegrandCCL(const IntegrandCCL& copy) : clt_(copy.clt_),
       cosmo_(copy.cosmo_), ell_(0), z_(0), jlR_(0), jlp1R_(0){} 

};//IntegrandCCL

 
}//end namespace






//Compute angular power spectrum given two ClTracers from ell=0 to lmax
//ccl_cosmo -> CCL cosmology (for P(k) and distances)
//lmax -> maximum angular multipole
//clt1 -> tracer #1
//clt2 -> tracer #2
//status -> status
SplPar * ccl_angular_cls_angpow(ccl_cosmology *ccl_cosmo, int lmax, CCL_ClTracer *clt_gc1, CCL_ClTracer *clt_gc2, int * status)
{
  if(clt_gc1->has_magnification || clt_gc2->has_magnification)
    printf("Magnification term not implemented in Angpow yet: will be ignored");
  if(clt_gc1->tracer_type==CL_TRACER_WL || clt_gc2->tracer_type==CL_TRACER_WL)
    printf("Weak lensing functions not implemented in Agnpow yet: will fail");
  
  // Initialize the Angpow parameters
  Angpow::Parameters para = Angpow::Param::Instance().GetParam();
  para.chebyshev_order_1 = 9;
  para.chebyshev_order_2 = 9;
  para.cl_kmax = 10;
  para.linearStep = 40;
  para.logStep = 1.15;

  // Initialize the radial selection windows W(z)
  Angpow::RadSplineSelect Z1win(clt_gc1->spl_nz);
  Angpow::RadSplineSelect Z2win(clt_gc2->spl_nz);

  // The cosmological distance tool to make the conversion z <-> r(z)
  Angpow::CosmoCoordCCL cosmo(ccl_cosmo, 1./A_SPLINE_MAX-1, 1./A_SPLINE_MIN-1, A_SPLINE_NA); //, para.cosmo_precision);

  // Initilaie the two integrand functions f(ell,k,z)
  Angpow::IntegrandCCL int1(clt_gc1, ccl_cosmo);
  Angpow::IntegrandCCL int2(clt_gc2, ccl_cosmo);

  // Initialize the Cl with parameters to select the ell set which is interpolated after the processing
  Angpow::Clbase clout(lmax,para.linearStep, para.logStep);

  // Main class to compute Cl with Angpow
  Angpow::Pk2Cl pk2cl; //Default: the user parameters are used in the Constructor 
  pk2cl.PrintParam();
  pk2cl.Compute(int1, int2, cosmo, &Z1win, &Z2win, lmax, clout);

  // Pass the Clbase class values (ell and C_ell) to the output spline
  int n_l = clout.Size();
  std::vector<double> ls(n_l);
  std::vector<double> cls(n_l);
  for(int index_l=0; index_l<n_l; index_l++) {
    ls[index_l]=clout[index_l].first; cls[index_l]=clout[index_l].second; 
  }
  SplPar * spl_cl = spline_init(clout.Size(), &ls[0], &cls[0], 0., 0. );

  return spl_cl;
}
