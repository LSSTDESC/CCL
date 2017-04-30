#ifndef ANGPOW_COSMO_SEEN
#define ANGPOW_COSMO_SEEN
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
#include "angpow_cosmo_base.h"
#include "angpow_tools.h" 
#include "angpow_quadinteg.h"
#include "angpow_parameters.h"


//#include <iostream> //debug singleton

//---------------
//Concrete implementation of Base Comological Coordinate
//---------------


namespace Angpow {

////////////////
//Implementation of some usefull Comological Functions
// Adpatation from SOPHYA::LUC module (www.sophya.org)
// Ansari, Magneville, Campagne
///////////////

class CosmoIntegrator : public  GaussKronrodQuadrature<r_8> {
public:
  typedef GaussKronrodQuadrature<r_8> type_t;

  static CosmoIntegrator& Instance() {
    static  CosmoIntegrator myInstance;
    return myInstance;
  }
  
  // delete copy and move constructors and assign operators
  CosmoIntegrator(CosmoIntegrator const&) = delete;             // Copy construct
  CosmoIntegrator(CosmoIntegrator&&) = delete;                  // Move construct
  CosmoIntegrator& operator=(CosmoIntegrator const&) = delete;  // Copy assign
  CosmoIntegrator& operator=(CosmoIntegrator &&) = delete;      // Move assign

protected:
  //Use a "a priori" fixed quadrature
  CosmoIntegrator() : GaussKronrodQuadrature<r_8>(20,Param::Instance().GetParam().quadrature_rule_ios_dir+"/CosmoRuleData.txt",true){}

};


class CosmoFuncImp {
public:


  //Default Ctor
  CosmoFuncImp(r_8 prec = 0.001);
  
  virtual ~CosmoFuncImp() {}

  //! return the reduced Hubble constant value normalized to 100 km/s/Mpc (h = H0/100) 
  inline double h() const { return(h_); }
  //! z=0 matter density 
  inline double OmegaMatter() const { return(omegamat_); }
//   //! z=0 radiation density 
//   inline double OmegaRadiation() const { return(omegarad_); }
//   //! z=0 photon density 
//   inline double OmegaPhoton() const { return(omegaphot_); }
//   //! z=0 baryon density 
//   inline double OmegaBaryon() const { return(omegabaryon_); }
//   //! z=0 non-baryonic matter density (cold dark matter) 
//   inline double OmegaCDM() const { return(omegamat_-omegabaryon_); }
//    //! z=0 total density 
//   inline double OmegaTotal() const { return(omegarad_+omegamat_+omegaL_+omegaX_); }
  //! z=0 cosmological constant density 
  inline double OmegaLambda() const { return(omegaL_); }
//   //! z=0 curvature density 
//   inline double OmegaCurv() const { return(omegaCurv_); }
//   //! z=0 dark energy density 
//   inline double OmegaDE() const { return(omegaX_); }
//   //! z=0 dark energy equation of state density
//   inline double wDE()  const { return(wX_); }
//   //! z=0 dark energy equation of state density w(a) = wDE + waDE*(1-a)
//   inline double waDE() const { return(waX_); }
//   // CMB temperature and photon and neutrino densities (in Kg/m^3) (today z=0)
  static inline r_8 PhotonNuDensityKgm3() { return rho_gamma_nu_Par; }
  static inline r_8 PhotonDensityKgm3()   { return rho_gamma_Par; }

   //! return the critical density (in units of kg/cm^3)
  inline r_8 CriticalDensityKgm3() const { return( h_*h_*RhoCrit_h1_Cst); }
  
  inline r_8 HubbleLengthMpc() const {
    return (SpeedOfLight_Cst/(h_*100.*1.e3));
  }

  void SetEmissionRedShift(double ze,  bool fginc=true);

  //! Return the line of sight distance for redshift \b z, in c/H0 unit
  inline r_8 LineOfSightComovDistance(r_8 z) {
    SetEmissionRedShift(z);
    return integGz_;
  }
  //! Return the line of sight distance for redshift \b z, in Mpc 
  inline r_8 LineOfSightComovDistanceMpc(r_8 z) { 
    return LineOfSightComovDistance(z)*HubbleLengthMpc(); 
  }

  //! Value of E(z) following Peebles/Hogg notation for redshift \b z 
  r_8 Ez(r_8 z) const;
  
protected:
  
  //! class operator
  class Gz: public ClassFunc1D {
  public:
    Gz(CosmoFuncImp* p) {p_=p;}
     virtual r_8 operator()(r_8 z) const {
       return 1./(p_->Ez(z));
     }
  private:
    CosmoFuncImp* p_;
  };//Gz


  //! Numerical integration of G(z) dz = 1/E(z) dz , [z1,z2]
  void NumIntegrateGz(double z1, double z2, double& resG);

private:
  //! forbid Copy & Assignment for the time beeing
  CosmoFuncImp& operator=(const CosmoFuncImp&);


protected:
  r_8 prec_;           //!< precision of computation

  r_8 h_;              //!< H0 in units of 100 km/s/Mpc : 0.5 -> 50 km/s/MPc
  r_8 omegamat_;       //!< Total matter density in universe at z=0 (including baryons)
  r_8 omegarad_;       //!< Total radiation density in universe at z=0 
  r_8 omegabaryon_;    //!< Baryon density (included in omegamat_); 
  r_8 omegaphot_;      //!< Photon energy density, (included in omegarad_) 

  r_8 omegaL_;         //!< cosmological constant
  r_8 omegaX_;         //!< The dark energy density 
  r_8 wX_;             //!< Dark energy density equation of state
  r_8 waX_;            //!< Dark energy density equation of state, scale factor dependant
  bool hasL_;             //!< True-> Non zero cosmological constant
  bool hasX_;             //!< True-> Non zero dark energy density

  r_8 omegaCurv_;      //!< curvature density (1-( omega0_ + omegaL_) )
  int kcurvature_;     //!< 0 Flat , +1 closed, -1 Open

  r_8 ze_;             //!< Emission redshift
  r_8 integGz_;        //!< Integral[G(z) dz]_[ze_ ... 0]

  // ---- Useful constants ----
  static constexpr r_8 MpctoMeters_Cst  = 3.0856e22;       //!< Mpc to meter conversion factor
  static constexpr r_8 LightYear_Cst    = 9.46073042e15;          //!< LightYear to meter conversion factor
  static constexpr r_8 SpeedOfLight_Cst = 2.99792458e8;       //!< Speed of light  m/sec

  // ---- Universe parameters 
  static constexpr r_8 rho_gamma_Par    = 4.6417e-31 ;          //!< Photon density today  (kg/m^3)
  static constexpr r_8 rho_gamma_nu_Par = 7.8042e-31;       //!< Photon+neutrino density today  (kg/m^3)
  static constexpr r_8 RhoCrit_h1_Cst   = 1.879e-26;         //!< Critical density for h=1 Kg/m^3


};//CosmoFuncImp



  class CosmoCoord : public CosmoCoordBase {
public:
  //! Ctor
  CosmoCoord(r_8 zmin=0., r_8 zmax=10., size_t npts=1000, r_8 prec=0.001)
    : cosmofunc_(prec),zmin_(zmin), zmax_(zmax), npts_(npts)
  {
    std::vector<r_8> vlos(npts_);
    std::vector<r_8> vz(npts_);
    for(size_t i=0; i<npts_; i++) {
      vz[i]=i*10./(r_8)npts_;
      vlos[i]=cosmofunc_.LineOfSightComovDistanceMpc(vz[i]);
    }
    rofzfunc_.DefinePoints(vz,vlos); //r(z0)
    zofrfunc_.DefinePoints(vlos,vz); //z(r0) = r^{-1}(r0)
  }//Ctor

  //! Dtor
  virtual ~CosmoCoord() {}


  //! r(z): radial comoving distance Mpc
  inline r_8 getLOS(r_8 z) const { return  r(z); } 
  inline virtual r_8 r(r_8 z) const { return rofzfunc_.YInterp(z); } 
  //! z(r): the inverse of radial comoving distance (Mpc)
  inline virtual r_8 z(r_8 r) const { return zofrfunc_.YInterp(r); }
  inline r_8 getInvLOS(r_8 r) const { return z(r); }
  


  //r_8 operator()(r_8 z) const { return getLOS(z); } //JEC 22/6/16 to be used for std::transform

  //!Hubble Cte
  inline double h() const { return cosmofunc_.h(); }
  //!Hubble function
  inline r_8 Ez(r_8 z) const { return cosmofunc_.Ez(z); }
  inline r_8 EzMpcm1(r_8 z) const {r_8 HL=cosmofunc_.HubbleLengthMpc(); return Ez(z)/HL; }

  //! z=0 matter density 
  inline double OmegaMatter() const { return cosmofunc_.OmegaMatter(); }
  //! z=0 cosmological constant density 
  inline double OmegaLambda() const { return cosmofunc_.OmegaLambda(); }

  
    
 protected:
  CosmoFuncImp  cosmofunc_;  //!< Cosmological functions
  SLinInterp1D  rofzfunc_; //!< linear interpolation r(z0)
  SLinInterp1D  zofrfunc_; //!< linear interpolation z(r0) = r^{-1}(r0)
  r_8 zmin_;           //!< minimal z
  r_8 zmax_;           //!< maximal z
  size_t npts_;        //!< number of points to define the interpolation in [zmin, zmax]
};// CosmoCoord





}//end namespace
#endif //ANGPOW_COSMO_SEEN
