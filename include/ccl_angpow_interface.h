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



//Compute angular power spectrum between two bins
//lmax -> maximum angular multipole
//clt1 -> tracer #1
//clt2 -> tracer #2
//double * ccl_angular_cls_angpow(int lmax, CCL_ClTracer *clt1, CCL_ClTracer *clt2, int * status);



namespace Angpow {

   //! Selection window with array input 
class RadSplineSelect : public RadSelectBase {
 public:
 RadSplineSelect(SplPar* spl): 
  RadSelectBase(spl->x0, spl->xf), spl_(spl) {}


  virtual r_8 operator()(r_8 z) const {
    return spline_eval(z,spl_);
  }

 private:
  SplPar* spl_;
};//RadArraySelect



/* class PowerSpecCCL : public PowerSpecBase { */
/*  public: */
/*   //! Constructor */
/*  PowerSpecCCL(ccl_cosmology * cosmo, double kmin=1e-5, double kmax=10, int nk=1000, bool use_rsd=true) */
/*    : ccl_cosmo_(cosmo), use_rsd_(use_rsd), has_rsd_(true) { */
/*     int status =0; */
/*     double * ks = ccl_log_spacing(kmin,kmax,N_K); */
/*     double Pks[N_K]; */
/*     double aref = 1.; // z=0 */
/*     for(int i=0; i<nk; i++) { */
/*       Pks[i] = ccl_linear_matter_power(cosmo, ks[i], aref, &status); */
/*     } */
    
/*     std::vector<double> vx; */
/*     std::vector<double> vy; */
    
/*     for(int i=0;i<N_K;i++){ */
/*       vx.push_back(ks[i]); */
/*       vy.push_back(Pks[i]); */
/*     } */
/*     double vxmin = *(std::min_element(vx.begin(), vx.end())); */
/*     double vxmax = *(std::max_element(vx.begin(), vx.end())); */
/*     Pk_ = new SLinInterp1D(vx,vy,vxmin,vxmax,0); */
/*   } */
/*   //! Destructor */
/*   virtual ~PowerSpecCCL() {} */
  
/*   //! Used to delete explicitly the local pointers */
/*   virtual void ExplicitDestroy() {  */
/*     if(ccl_cosmo_) delete ccl_cosmo_; */
/*     if(Pk_) delete Pk_; */
/*   } */

/*   /\*! Explicit to get a clone of the primary object via shallow copy */
/*     using the Copy Ctor */
/*    *\/ */
/*   virtual PowerSpecCCL* clone() const { */
/*     return new PowerSpecCCL(static_cast<const PowerSpecCCL&>(*this)); */
/*   } */
  
/*   /\*! called by angpow_kinteg.cc to fix the value of some function */
/*     at fixed z value (and l too if necessary) */
/*    *\/ */
/*   void Init(double z) { */
/*     int status=0;  */
/*     double tmp= ccl_growth_factor(ccl_cosmo_,1.0/(1+z), &status); */
/*     growth2_ = tmp*tmp; */
/*     // WARNING: here we want to store dlnD/dln(+1z) = - dlnD/dlna */
/*     fz_= - ccl_growth_rate(ccl_cosmo_,1.0/(1+z), &status); */
/*     // TO SET with CCL */
/*     bias_ = 1.; */
/*   } */

/*   //Main operator */
/*   virtual r_8 operator()(double k, double z) { */
/*     r_8 pk = (Pk_->operator()(k)); */
/*     if(pk<0) pk=0.; // P(k) values can be negative because of poor interpolation below kmin */
/*     return growth2_*pk; */
/*   } */

/*   virtual bool get_has_rsd() { return has_rsd_; } */
/*   virtual r_8 get_fz() { return fz_; } */
/*   virtual r_8 get_bias() { return bias_; } */

/*  private: */

/*   SLinInterp1D* Pk_;          //!< access to  Pk(k) */
/*   ccl_cosmology* ccl_cosmo_;   //!< access to CCL cosmology */
/*   double growth2_;               //!< D(zi)^2 */
/*   //double fz_; //! growth rate f(z) */
/*   bool use_rsd_; */
/*   bool has_rsd_; */
/*   r_8 fz_; */
/*   r_8 bias_; */

/*   //forbid for the time beeing the assignment operator */
/*   PowerSpecCCL& operator=(const PowerSpecCCL& copy); */
  
/*   //Minimal copy to allow Main operator(int, r_8, r_8) to work */
/*  PowerSpecCCL(const PowerSpecCCL& copy) :  */
/*   Pk_(copy.Pk_), ccl_cosmo_(copy.ccl_cosmo_), growth2_(copy.growth2_), has_rsd_(copy.has_rsd_), fz_(copy.fz_) {}  */
/* }; */




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





/*   //! Base class of half the integrand function */
/* class IntegrandCCL : public IntegrandBase { */
/*  public: */
/*  IntegrandCCL(PowerSpecCCL& PSpec, CosmoCoordCCL& cosmo, int ell=0, r_8 z=0.): */
/*   PSpec_(PSpec), cosmo_(cosmo), ell_(ell), z_(z) { */
/*     R_ = cosmo_.r(z_); */
/*     jlR_ = new JBess1(ell_,R_); */
/*     jlp1R_ = new JBess1(ell_+1,R_); */
/*   } */
/*   void Init(int ell, r_8 z){ */
/*     ell_=ell; z_=z; */
/*     PSpec_.Init(z_); */
/*     R_ = cosmo_.r(z_); */
/*     jlR_ = new JBess1(ell_,R_); */
/*     jlp1R_ = new JBess1(ell_+1,R_); */
/*   } */
/*   virtual ~IntegrandCCL() {} */
/*   virtual r_8 operator()(r_8 k) const { */
/*     r_8 x = k*R_; */
/*     r_8 jlRk = (*jlR_)(k); */
/*     r_8 delta = PSpec_.get_bias()*jlRk; */
/*     r_8 jlRksecond = 0.; */
/*     if(PSpec_.get_has_rsd()){ */
/*       if(x<1e-40) { */
/*     	if(ell_==0) { */
/*     	  jlRksecond = -1./3. + x*x/10.; */
/*     	} else if(ell_==2) { */
/*     	  jlRksecond = 2./15. - 2*x*x/35.; */
/*     	} else { */
/*     	  jlRksecond = 0.; */
/*     	} */
/*       } else { */
/*     	jlRksecond = 2.*(*jlp1R_)(k)/x + (ell_*(ell_-1.)/(x*x) - 1.)*jlRk; */
/*       } */
/*       delta += PSpec_.get_fz()*jlRksecond; */
/*     } */
/*     return(k*sqrt(fabs(PSpec_(k,z_)))*delta); */
/*   } */
/*   virtual IntegrandCCL* clone() const { */
/*     return new IntegrandCCL(static_cast<const IntegrandCCL&>(*this)); */
/*   } */
/*   virtual void ExplicitDestroy() { */
/*     if(jlR_) delete jlR_; */
/*     if(jlp1R_) delete jlp1R_; */
/*   } */
/* private: */
/*   PowerSpecCCL& PSpec_;  //no ownership */
/*   CosmoCoordCCL& cosmo_;  //no ownership */
/*   int ell_; */
/*   r_8 z_; */
/*   r_8 R_; */
/*   JBess1* jlR_;  // j_ell(k*R) */
/*   JBess1* jlp1R_;   // j_(ell+1)(k*R) */
/* };//IntegrandCCL */


 

  //! Base class of half the integrand function
class IntegrandCCL : public IntegrandBase {
 public:
 IntegrandCCL(CCL_ClTracer* clt, ccl_cosmology* cosmo, int ell=0, r_8 z=0.):
  clt_(clt), cosmo_(cosmo), ell_(ell), z_(z) {
    Init(ell,z);
  }
  void Init(int ell, r_8 z){
    int status=0;
    ell_=ell; z_=z;
    R_ = ccl_comoving_radial_distance(cosmo_, 1.0/(1+z), &status);
    jlR_ = new JBess1(ell_,R_);
    jlp1R_ = new JBess1(ell_+1,R_);
    if(clt_->has_rsd) {
      // WARNING: here we want to store dlnD/dln(+1z) = - dlnD/dlna
      fz_= - ccl_growth_rate(cosmo_,1.0/(1+z), &status);
    }
    bz_ = spline_eval(z,clt_->spl_bz);
  }
  virtual ~IntegrandCCL() {}
  virtual r_8 operator()(r_8 k) const {
    int status=0;
    r_8 Pk = ccl_linear_matter_power(cosmo_, k , 1./(1+z_), &status);
    r_8 x = k*R_;
    r_8 jlRk = (*jlR_)(k);
    r_8 delta = bz_*jlRk;
    r_8 jlRksecond = 0.;
    if(clt_->has_rsd){
      if(x<1e-40) {
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
  int ell_; 
  r_8 z_;   // redshift z
  r_8 R_;   // radial comoving distance r(z)
  r_8 fz_;  // growth rate f(z)
  r_8 bz_;  // bias b(z)
  JBess1* jlR_;  // j_ell(k*R)
  JBess1* jlp1R_;   // j_(ell+1)(k*R)
};//IntegrandCCL




 
}//end namespace






//Compute angular power spectrum between two bins
//lmax -> maximum angular multipole
//ls -> list of multipole value (output)
//clt1 -> tracer #1
//clt2 -> tracer #2
SplPar * ccl_angular_cls_angpow(ccl_cosmology *ccl_cosmo, int lmax, CCL_ClTracer *clt_gc1, CCL_ClTracer *clt_gc2, int * status)
{
  // Initialize the Angpow parameters
  Angpow::Parameters para = Angpow::Param::Instance().GetParam();
  para.wtype1 = Angpow::Parameters::Dirac; para.wtype2 = Angpow::Parameters::Dirac;
  para.mean1 = 1.0; para.mean2 = 1.0;
  //para.cosmo_zmax = 9.0;
  //para.cosmo_npts = 1000;
  para.chebyshev_order_1 = 9;
  para.chebyshev_order_2 = 9;
  para.cl_kmax = 10;

  // Initialize the radial selection windows
  Angpow::RadSplineSelect Z1win(clt_gc1->spl_nz);
  Angpow::RadSplineSelect Z2win(clt_gc2->spl_nz);

  //The cosmological distance tool 
  Angpow::CosmoCoordCCL cosmo(ccl_cosmo, 1./A_SPLINE_MAX-1, 1./A_SPLINE_MIN-1, A_SPLINE_NA); //, para.cosmo_precision);

  // Integrand functions
  Angpow::IntegrandCCL int1(clt_gc1, ccl_cosmo);
  Angpow::IntegrandCCL int2(clt_gc2, ccl_cosmo);

  //Initialize the Cl with parameters to select the ell set which is interpolated after the processing
  int Lmax = 500; //para.Lmax; //ell in [0, Lmax-1]
  Angpow::Clbase clout(Lmax,para.linearStep, para.logStep);

  //Main class
  Angpow::Pk2Cl pk2cl; //Default: the user parameters are used in the Constructor 
  pk2cl.PrintParam();
  pk2cl.Compute(int1, int2, cosmo, &Z1win, &Z2win, Lmax, clout);

  int n_l = clout.Size();
  std::vector<double> ls(n_l);
  std::vector<double> cls(n_l);
  for(int index_l=0; index_l<n_l; index_l++) {
    ls[index_l]=clout[index_l].first; cls[index_l]=clout[index_l].second; 
  }
  SplPar * spl_cl = spline_init(clout.Size(), &ls[0], &cls[0], 0., 0. );

  /* {//save the Cls */
  /*   std::fstream ofs; */
  /*   std::string outName = para.output_dir + para.common_file_tag + "cl.txt"; */
  /*   ofs.open(outName, std::fstream::out); */
  /*   for(int index_l=0;index_l<clout.Size();index_l++){ */
  /*     ofs << std::setprecision(20) << clout[index_l].first << " " << clout[index_l].second << std::endl; */
  /*   } */
  /*   ofs.close(); */
  /* } */
  
  /* {//save ctheta */

  /*   Angpow::CTheta ct(clout,para.apod); */

  /*   std::fstream ofs; */
  /*   std::string outName = para.output_dir + para.common_file_tag + "ctheta.txt"; */
  /*   //define theta values */
  /*   const int Npts=100; */
  /*   const double theta_max=para.theta_max*M_PI/180; */
  /*   double step=theta_max/(Npts-1); */
      
  /*   ofs.open(outName, std::fstream::out); */
  /*   for (size_t i=0;i<Npts;i++){ */
  /*     double t=i*step; */
  /*     ofs << std::setprecision(20) << t << " " << ct(t) << std::endl; */
  /*   } */
  /*   ofs.close(); */
    
  /*   outName = para.output_dir + para.common_file_tag + "apod_cl.txt"; */
  /*   ct.WriteApodCls(outName); */
  /* } */
  return spl_cl;
}
