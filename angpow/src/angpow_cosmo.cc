#include "Angpow/angpow_cosmo.h"
#include "Angpow/angpow_parameters.h"
#include <sstream>

namespace Angpow {


CosmoFuncImp::CosmoFuncImp(r_8 prec): prec_(prec) {

  Parameters para = Param::Instance().GetParam();

  //LambdaCDM input parameters
  h_ = para.h;
  omegamat_    = para.omega_matter;
  omegabaryon_ = para.omega_baryon;

  omegarad_  = PhotonNuDensityKgm3() / CriticalDensityKgm3();
  omegaphot_ = PhotonDensityKgm3() / CriticalDensityKgm3();

  omegaX_ = para.omega_X;
  hasX_   = para.hasX;
  wX_     = para.wX;
  waX_    = para.waX;

  //deduced parameters

  omegaL_ = 1. - (omegamat_ + omegarad_ + omegaX_);
  if (omegaL_ < 0.) {
    std::cout<< " CosmoFuncImp::Ctor Error -> " 
	     << " OmegaL=" << omegaL_ << " < 0" << endl;
    throw AngpowError("CosmoFuncImp::Ctor OmegaL<0");
  }
  hasL_ = true;

  omegaCurv_ = (1. - (omegamat_ + omegarad_ + omegaL_ + omegaX_));
  if (fabs(omegaCurv_) < 1.e-39) {
    omegaCurv_ = 0.;
    kcurvature_ = 0;
  }
  else { 
    if (omegaCurv_ < 0) kcurvature_ = 1;  
    else kcurvature_ = -1;
  }


  // We initilalize ze_, ...  to today
  ze_ = 0.;
  integGz_ = 0;

}//Ctor


r_8 CosmoFuncImp::Ez(r_8 z) const {
   double zz = 1+z;
   double ez2 = zz*zz* ( omegarad_*zz*zz + omegamat_*zz + omegaCurv_);
   if (hasL_) ez2 += omegaL_;
   else if (hasX_) {
     if ( waX_ != 0.) ez2 += omegaX_*pow(zz,3.*(1.+wX_+waX_))*exp(-3.*waX_*z/zz);
     else             ez2 += omegaX_*pow(zz,3.*(1.+wX_));
   }
   return sqrt(ez2);
}//Ez



/*!
  Defines the Emission redshift (ze) and computes the corresponding coordinates 
  Time (Emission time = LookBackTime) , RadCoord , Chi , ...
  perform most of the computation, which may take a while. 
  \param ze : the target / emission redshift 
  \param fginc : if true, perform incremental calculation, starting from the last computed ze

  \warning This method should be called again whenever you change any of the cosmological parameters.
*/
/* --Methode-- */
void CosmoFuncImp::SetEmissionRedShift(double ze, bool fginc)
{
  if (ze < 0.) {
    std::cout << " CosmoFuncImp::SetEmissionRedShift/Error ze = " << ze 
	      << " less than 0 ! " << std::endl;
    throw AngpowError("CosmoFuncImp::SetEmissionRedShift(ze < 0.)");
  }

  if (ze < 1.e-39) {
    ze_ = 0.;
    integGz_ = 0;
    return;
  }

  double zelast = ze_;
  ze_ = ze;  
  // Reference: Principles of Physical Cosmology - P.J.E. Peebles  
  //            Princeton University Press - 1993
  //              ( See Chapter 13)
  // We have to integrate Integral(dz / E(z)) from 0 to ze  (cf 13.29)
  //      E(z) = Sqrt(Omega0*(1+z)^3 + OmegaCurv*(1+z)^2 + OmegaL) (cf 13.3)
  // G(z) = 1/E(z) : integration element for the calculation of D_A  

  if (fginc) {  // Calcul incremental
    double inGz = 0.;
    if (zelast < ze_) {
      NumIntegrateGz(zelast, ze_, inGz);
      integGz_ += inGz;
    } else {
      NumIntegrateGz(ze_, zelast, inGz);
      integGz_ -= inGz;
    }
  } else {
    NumIntegrateGz(0., ze_, integGz_);
  }

}//SetEmissionRedShift

/*!
  Numerical integration of G(z) dz = 1/E(z) dz and numerical
  \input z1 Lower redshift bound
  \input z2 Upper redshift bound
  \input resG Resulting integration of G(z)
*/
void CosmoFuncImp::NumIntegrateGz(double z1, double z2, double& resG){
   if ((z1 < 0.) || (z2 < 0.) || (z2 < z1)) {
     std::cout << " CosmoFuncImp::NumIntegGz()/Error invalid values for z1,z2: " 
	       << z1 << "," << z2 << std::endl;
     throw AngpowError("CosmoFuncImp::NumIntegGzGTz() invalid z1/z2 values");
   }
   
   Gz f(this);

   CosmoIntegrator::Instance().SetFuncBounded(f,z1,z2);
   Quadrature<r_8,CosmoIntegrator::type_t>::values_t integ_val = CosmoIntegrator::Instance().GlobalAdapStrat(prec_);
   resG = integ_val.first;

}//NumIntegrateGz



}//namespace
