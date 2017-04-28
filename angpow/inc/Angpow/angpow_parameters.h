#ifndef ANGPOW_PARAMETERS_SEEN
#define ANGPOW_PARAMETERS_SEEN
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
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <iostream>

#include "angpow_numbers.h"
#include "angpow_utils.h"
#include "angpow_exceptions.h"
namespace Angpow {

struct Parameters {
  /*
    Cl parameters : see angpow_clbase.h
    ell in [0, Lmax-1]
    ell sampling lineat up to linearStep then logarithmic. If logStep=0 ell sampling linear
   */
  int Lmax;
  int linearStep;
  r_8 logStep;
  /*
    Selection Windows : see. angpow_radial.h
    Select_t: type of selection in redshift
      . Dirac: 1 redshift
      . Gauss: gaussian selection
      . GaussGal: gaussian selection x dN/dz-galaxy function 
      . Uniform: Cte with a  sharp cut in z [zmin, zmax]
      . TopHat: Cte with smooth edges 
    mean, width: the mean value of the z-selection and width = 1 sigma (Gauss) or  1/2 half width for TopHat & Uniform
    smooth: smoothing of the edges of TopHat 
    n_sigma_cut: 
      . z in [mean - n_sigma_cut * width, mean + n_sigma_cut * width] for Gaussian like
      . z in [mean - (1+ n_sigma_cut*smooth)*width, mean + (1 + n_sigma_cut*smooth)*width]
   */
  enum Select_t { Dirac, Gauss, GaussGal, Uniform, TopHat };
  Select_t wtype1, wtype2;
  r_8 mean1, width1, mean2, width2;
  r_8 smooth_edges;
  r_8 n_sigma_cut;


  /*
    Control integration algorithm : see angpow_pk2cl.h
      . kmax : value in Mpc^-1 of the maximum value of k to be considered in the k-integration
      . radial_order_1: the quadrature order over the redshift axis for the first selection Window
      . chebyshev_order_1 : Order of the Chebyshev Transform for k*Sqrt[Pws]*Bessel(r(z1)*k) (first selection Window)
      . idem for radial_order_2 & chebyshev_order_2 for the second selection Window
      . n_bessel_roots_per_interval: the k-integration is performed by sum over intervales defined using
        j_l(x) roots. This parameter control how many roots to be gathered in a single intervalle where the functions
	are expended over Chebyshev polynomial series.
      . total_weight_cut: to be taken into account, the product of quadrature weights and selection 
                          function satisfies |wij|>total_weight_cut. 0 means no cut
      . if has_deltaR_cut == true; then to be taken into account |r(z_i) - r(r_j)|<deltaR_cut (Mpc); 
           if has_deltaR_cut == false ù> no cut on |r(z_i) - r(r_j)| applied
   */
  r_8 cl_kmax;
  int radial_order_1, radial_order_2;
  int chebyshev_order_1, chebyshev_order_2;
  int n_bessel_roots_per_interval;
  r_8 total_weight_cut;
  bool has_deltaR_cut;
  r_8 deltaR_cut;


  /*
    Cosmological parameters: see angpow_cosmo.h (Simple Cosmological Universe)
     . h: the reduced Hubble constant value normalized to 100 km/s/Mpc (h = H0/100) 
     . omega_matter: z=0 matter density 
     . omega_baryon: z=0 baryon density
     . omega_X     : z=0 general dark energy density
     . wX, waX     = z=0 dark energy equation of state density w(a) = wX + waX*(1-a)
     . nb: values fixed by default
          omega Radiation = h^2 rho_gamma/rho_critic = h^2 4.6417e-31/1.879e-26;
	  omega Lambda    = 1 - (omegamat_ + omegarad_ + omegaX_)
          omega Curvature = (1. - (omegamat_ + omegarad_ + omegaL_ + omegaX_));
   */
  r_8 h;
  r_8 omega_matter;
  r_8 omega_baryon;
  r_8 omega_X;
  bool hasX;
  r_8 wX;
  r_8 waX;
  r_8 bias;
  bool include_rsd;
  /*
    Cosmological distances interpolation parameters: see angpow_cosmo.h 
      . z in [zmin, zmax]
      . npts: number of z to used by the interpolation
      . precision: precision of the integration  1/E(z) dz
   */
  r_8 cosmo_zmin, cosmo_zmax;
  int cosmo_npts;
  r_8 cosmo_precision;
  
  
  /*
    Bessel parameters (see angpow_bessel.cc MakeBesselJImpXmin method)
      find x_min such that j_l(x_min) = jl_xmin_cut for l in [0, Lmax_for_xmin-1] 
  */
  int Lmax_for_xmin;
  r_8 jl_xmin_cut;
  
  /*
    IOs
     . output_dir: directory fo output files (ex. Cls)
     . common_file_tag: tag common to all output files (ex. angpow_cls.txt)
     . quadrature_rule_ios_dir: directory where to find and/or save the quadrature files
  */
  std::string output_dir;
  std::string common_file_tag;

  std::string quadrature_rule_ios_dir;

  /*
    Power Spectrum file (zref = 0) see angpow_powspec.h
      P(k) in Mpc^3
     . power_spectrum_input_dir: directory location
     . power_spectrum_input_file: file name
     . pw_kmin: minimal k used in the file (k in Mpc^-1)
     . pw_kmax: maximal k "
   */
  std::string power_spectrum_input_dir;
  std::string power_spectrum_input_file;
  r_8 pw_kmin;
  r_8 pw_kmax;
  
  /* ctheta control: 
     -thetamax : max angle in degree where ctheta will be computed
     -apod: frac of lmax for gaussian apodization ie. sigma_l=l/ls
     where ls=apod*lmax
  */
  r_8 theta_max;
  r_8 apod;

};//parameters


class Param {
public:

  static Param& Instance() {
    static Param myInstance;
    return myInstance;
  }

  void SetToDefault();


  void ReadParam(const std::string& fName);
  std::ostream& WriteParam(std::ostream& os=std::cout);
  Parameters& GetParam() {return par_;}
  void Reset() {SetToDefault();}

  //singleton idiom
  // delete copy and move constructors and assign operators
  Param(Param const&) = delete;             // Copy construct
  Param(Param&&) = delete;                  // Move construct
  Param& operator=(Param const&) = delete;  // Copy assign
  Param& operator=(Param &&) = delete;      // Move assign

  typedef  std::map<const std::string, const std::string> Dico_t;

  
protected:
  Param() {
    SetToDefault();
  }


  void GetDico(const std::string& fName, Dico_t& dico);

  //Find key into dico and transform associated value into val of type T
  template<class T>
  int findParam(const Dico_t& dico, const std::string& key, T& val, 
		const std::string& sdefault = "default");
 
  //Find key into dico and transform the associated value into val of type T
  template<class T>
  int findParam(const Dico_t& dico, const std::string& key, std::vector<T>& val, 
		const std::string& sdefault = "default");


  inline Parameters::Select_t SetSelectType(std::string& str) {
    std::transform(str.begin(), str.end(), str.begin(), ToLower);
    Parameters::Select_t wtype = Parameters::Dirac;
    if(str == "dirac"){
      wtype = Parameters::Dirac;
    } else if (str == "gauss") {
      wtype = Parameters::Gauss;
    } else if (str == "gaussgal") {
      wtype = Parameters::GaussGal;
    } else if (str == "tophat") {
      wtype = Parameters::TopHat;
    } else {
      throw AngpowError("Param::SetSelectType unknown window type");
    }
    return wtype;
  }
  
  inline std::string GetSelectType(Parameters::Select_t type){
    std::string swin;
    switch(type){
    case Parameters::Dirac:
      swin = "Dirac";
      break;
    case Parameters::Gauss:
      swin = "Gauss";
      break;
    case Parameters::GaussGal:
      swin = "GaussGal";
      break;
    case Parameters::TopHat:
      swin = "TopHat";
      break;
    default:
      swin = "Unknown";
      break;
    }//sw
    return swin;
  }


  Parameters par_;
};//Param


}//end namespace
#endif //ANGPOW_PARAMETERS_SEEN
