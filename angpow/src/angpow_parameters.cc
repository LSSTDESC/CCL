#include "Angpow/angpow_parameters.h"
#include "Angpow/angpow_utils.h"
#include "Angpow/angpow_exceptions.h"

#include <iostream>
#include <fstream> 
#include <iomanip>  

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <algorithm>
#include <iterator>

namespace Angpow {

void Param::SetToDefault() {

  // Cl parameters
  par_.Lmax = 501;
  par_.linearStep = 40;
  par_.logStep    = 1.15;
  
  // Selection Selects


  //Cross Dirac-Dirac z=1 (cl_kmax can be set to 10 Mpc^-1)
  par_.wtype1 = Parameters::Dirac;
  par_.mean1  = 1.0;
  par_.width1 = 0.0;
  par_.wtype2 = par_.wtype1;
  par_.mean2  = par_.mean1;
  par_.width2 = par_.width1;
  
  par_.smooth_edges = 0.1; //in case of TopHat 
  
  par_.n_sigma_cut = 5.;

  // Control integration algorithm
  par_.cl_kmax = 1;

  //  par_.radial_order_1    = 350;
  par_.radial_order_1    = 70;
  par_.chebyshev_order_1 = 9;
  par_.radial_order_2    =  par_.radial_order_1;
  par_.chebyshev_order_2 =  par_.chebyshev_order_1;
  par_.n_bessel_roots_per_interval = 100;
  par_.total_weight_cut = 1.e-6;
  par_.has_deltaR_cut = false;
  par_.deltaR_cut = 10000.;


  //Cosmological parameters
  par_.h = 0.679;
  par_.omega_matter = 0.3065;
  par_.omega_baryon = 0.0483;
  par_.omega_X      = 0.;
  par_.hasX = false;
  par_.wX   = -1.;
  par_.waX  = 0.;
  par_.bias     = 1.;
  par_.include_rsd = false;

  //Cosmological distances interpolation parameters
  par_.cosmo_zmin = 0.;
  par_.cosmo_zmax = 10.;
  par_.cosmo_npts = 1000;
  par_.cosmo_precision = 0.001;


  //Bessel parameters
  par_.Lmax_for_xmin = 2000;
  par_.jl_xmin_cut   = 5e-10;
  
  
  //IOs
  par_.output_dir = "./";
  par_.common_file_tag = "angpow_";
  par_.quadrature_rule_ios_dir = "./data/";


  //PowerSpectrum
  par_.power_spectrum_input_dir = "./data/";
  par_.power_spectrum_input_file = "classgal_pk_z0.dat";
  par_.pw_kmin = 1.e-5;
  par_.pw_kmax = 100.;
  
  //Ctheta
  par_.theta_max=10.;
  par_.apod=0.4;


}//SetToDefault


template<class T>
int Param::findParam(const Dico_t& dico, const std::string& key, T& val, const std::string& sdefault) {
  Dico_t::const_iterator it = dico.find(key);

  if(it != dico.end()){
    std::string tmp = it->second;
    if( tmp == sdefault) {
      return 1;
    } else {
      val = ToNumber<T>(tmp);
      return 0;
    }
  }else{
    return -1;
  }
  return 0;
}


template<class T>
int Param::findParam(const Dico_t& dico, const std::string& key, std::vector<T>& val, const std::string& sdefault){

  Dico_t::const_iterator it = dico.find(key);
  val.clear();

  if(it != dico.end()){
    std::string tmp = it->second;
    if( tmp == sdefault ) {
      return 1;
    } else {
      std::vector<std::string> tokens;
      split(tmp,",",tokens);
      val.resize(tokens.size());
      std::transform(tokens.begin(),tokens.end(),val.begin(), ToNumber<T>);
      return 0;
    }
  }else{
    return -1;
  }
  return 0;
}


template<>
int Param::findParam(const Dico_t& dico, const std::string& key, std::string& val, const std::string& sdefault) {

  Dico_t::const_iterator it = dico.find(key);

  if(it != dico.end()){
    val = it->second;

    if( val == sdefault) {
      return 1;
    } else {
      return 0;
    }
  }else{
    return -1;
  }
  return 0;

}

template<>
int Param::findParam(const Dico_t& dico, const std::string& key, std::vector<std::string>& val, const std::string& sdefault) {

  Dico_t::const_iterator it = dico.find(key);
  val.clear();

  if(it != dico.end()){
    std::string tmp = it->second;
    if( tmp == sdefault ) {
      return 1;
    } else {
      split(tmp,",",val);
      return 0;
    }
  }else{
    return -1;
  }
  return 0;

}


void Param::ReadParam(const std::string& fName){
  Dico_t  dico;
  
  GetDico(fName, dico);

  //Debug  
//   for(Dico_t::iterator it=dico.begin(); it!=dico.end();++it){
//     std::cout << it->first << " => " << it->second << '\n';
//   }
//   std::cout << "----------------------- " << std::endl;

  int itmp;
  r_8 dtmp;
  std::string stmp;
  std::vector<int> ivec;
  std::vector<r_8> dvec;
  std::vector<std::string> svec;

  //   Cl parameters : see angpow_clbase.h
  if(findParam(dico,"Lmax",itmp)==0) par_.Lmax = itmp;
  if(findParam(dico,"linearStep",itmp)==0) par_.linearStep = itmp;
  if(findParam(dico,"logStep",dtmp)==0) par_.logStep = dtmp;

  //Ctheta: see angpwo_ctheta
  if(findParam(dico,"theta_max",dtmp)==0) par_.theta_max = dtmp;
  if(findParam(dico,"apod",dtmp)==0) par_.apod = dtmp;

  // Selection Selects
  std::vector<std::string> wtype;
  if(findParam(dico,"wtype",wtype)==0) {
    par_.wtype1 = SetSelectType(wtype[0]);
    if(wtype.size() == 2){
      par_.wtype2 = SetSelectType(wtype[1]);
    }else{
      par_.wtype2 = par_.wtype1;
    }
  }

  if(findParam(dico,"mean",dvec)==0){
    par_.mean1 = dvec[0];
    if(dvec.size() == 2){
       par_.mean2 = dvec[1];
    }else{
      par_.mean2 = par_.mean1;
    }
  }

  
  if(findParam(dico,"width",dvec)==0){
    par_.width1 = dvec[0];
    if(dvec.size() == 2){
      par_.width2 = dvec[1];
    }else{
      par_.width2 = par_.width1;
    }
  }

  if(findParam(dico,"smooth_edges",dtmp)==0) par_.smooth_edges = dtmp;

  if(findParam(dico,"n_sigma_cut",dtmp)==0) par_.n_sigma_cut = dtmp;

  //    Control integration algorithm 
  if(findParam(dico,"cl_kmax",dtmp)==0) par_.cl_kmax = dtmp;
  
  if(findParam(dico,"radial_order",dvec)==0){
    par_.radial_order_1 = dvec[0];
    if(dvec.size() == 2){
      par_.radial_order_2 = dvec[1];
    }else{
      par_.radial_order_2 = par_.radial_order_1;
    }
  }

  if(findParam(dico,"chebyshev_order",dvec)==0){
    par_.chebyshev_order_1 = dvec[0];
    if(dvec.size() == 2){
      par_.chebyshev_order_2 = dvec[1];
    }else{
      par_.chebyshev_order_2 = par_.chebyshev_order_1;
    }
  }

  if(findParam(dico,"n_bessel_roots_per_interval",itmp)==0) par_.n_bessel_roots_per_interval = itmp;

  if(findParam(dico,"total_weight_cut",dtmp)==0) par_.total_weight_cut = dtmp;
  if(findParam(dico,"has_deltaR_cut",itmp)==0) par_.has_deltaR_cut = (itmp == 1) ? true: false;
  if(findParam(dico,"deltaR_cut",dtmp)==0) par_.deltaR_cut = dtmp;


  //Cosmological parameters
  if(findParam(dico,"h",dtmp)==0) par_.h = dtmp;
  if(findParam(dico,"omega_matter",dtmp)==0) par_.omega_matter = dtmp;
  if(findParam(dico,"omega_baryon",dtmp)==0) par_.omega_baryon = dtmp;
  if(findParam(dico,"hasX",itmp)==0) par_.hasX = (itmp == 1) ? true: false;
  if(findParam(dico,"omega_X",dtmp)==0) par_.omega_X = dtmp;
  if(findParam(dico,"wX",dtmp)==0) par_.wX = dtmp;
  if(findParam(dico,"waX",dtmp)==0) par_.waX = dtmp;
  if(findParam(dico,"bias",dtmp)==0) par_.bias = dtmp;
  if(findParam(dico,"include_rsd",itmp)==0) par_.include_rsd = (itmp == 1) ? true: false;

  //Cosmological distances interpolation parameters
  if(findParam(dico,"cosmo_zmin",dtmp)==0) par_.cosmo_zmin = dtmp;
  if(findParam(dico,"cosmo_zmax",dtmp)==0) par_.cosmo_zmax = dtmp;
  if(findParam(dico,"cosmo_npts",itmp)==0) par_.cosmo_npts = itmp;
  if(findParam(dico,"cosmo_precision",dtmp)==0) par_.cosmo_precision = dtmp;
  
  //  Bessel parameters
  if(findParam(dico,"Lmax_for_xmin",itmp)==0) par_.Lmax_for_xmin = itmp;
  if(findParam(dico,"jl_xmin_cut",dtmp)==0) par_.jl_xmin_cut  = dtmp;
  
  //    IOs
  if(findParam(dico,"output_dir",stmp)==0) par_.output_dir = stmp;
  if(findParam(dico,"common_file_tag",stmp)==0) par_.common_file_tag = stmp;
  if(findParam(dico,"quadrature_rule_ios_dir",stmp)==0) par_.quadrature_rule_ios_dir = stmp;

  //    Power Spectrum file
  if(findParam(dico,"power_spectrum_input_dir",stmp)==0) par_.power_spectrum_input_dir = stmp;
  if(findParam(dico,"power_spectrum_input_file",stmp)==0) par_.power_spectrum_input_file = stmp;
  if(findParam(dico,"pw_kmin",dtmp)==0) par_.pw_kmin = dtmp;
  if(findParam(dico,"pw_kmax",dtmp)==0) par_.pw_kmax = dtmp;
  

}//ReadParam


std::ostream& Param::WriteParam(std::ostream& os){

  os << "Parameters..." << std::endl;
  os << "Lmax = " << par_.Lmax;
  os << " linearStep = " << par_.linearStep;
  os << " logStep = " <<  par_.logStep << "\n";
  os << "wtype 1 = " << GetSelectType(par_.wtype1);
  os << " mean = " <<   par_.mean1;
  os << " width = " <<  par_.width1 << "\n";
  os << "wtype 2 = " << GetSelectType(par_.wtype2);
  os << " mean = " <<   par_.mean2;
  os << " width = " <<  par_.width2 << "\n";
  os << "n_sigma_cut = " << par_.n_sigma_cut << "\n";
  os << "cl_kmax = " << par_.cl_kmax << "\n";
  os << "radial_order = " <<  par_.radial_order_1 << ", "<< par_.radial_order_2 << "\n";
  os << "chebyshev_order = " <<  par_.chebyshev_order_1 << ", "<<  par_.chebyshev_order_2 << "\n";
  os << "n_bessel_roots_per_interval = " << par_.n_bessel_roots_per_interval << "\n";
  os << "total_weight_cut = " << par_.total_weight_cut << "\n";
  if(par_.has_deltaR_cut){
    os << "deltaR_cut = " << par_.deltaR_cut;
  } else {
    os << "No deltaR cut used";
  }
  os << "\n";
  os << "h = " << par_.h << ", " 
     << "omega_matter = " << par_.omega_matter << ", "
     << "omega_baryon = " <<  par_.omega_baryon;
  if(par_.hasX){
    os << ", omega_X = " <<  par_.omega_X 
       << ", wX = " <<  par_.wX 
       << ", waX = " <<  par_.waX;
  } else {
    os << "No omeag_X used";
  }
  os << "\n";
  os << "cosmo_zmin = " << par_.cosmo_zmin << ", "
     << "cosma_zmax = " << par_.cosmo_zmax << ", "
     << "cosmo_npts = " <<  par_.cosmo_npts << ", "
     << "cosmo_precision = " << par_.cosmo_precision << "\n"; 
  os << "Lmax_for_xmin = " << par_.Lmax_for_xmin << ", " 
     << "jl_xmin_cut = " << par_.jl_xmin_cut << "\n";
  os << "output_dir = <"<<  par_.output_dir << ">\n";
  os << "common_file_tag = <" <<  par_.common_file_tag << ">\n";
  os << "quadrature_rule_ios_dir = <" << par_.quadrature_rule_ios_dir << ">\n";
  os << "power_spectrum_input_dir = <" << par_.power_spectrum_input_dir << ">\n";
  os << "power_spectrum_input_file = <" << par_.power_spectrum_input_file << ">\n";
  os << "pw_kmin = " <<  par_.pw_kmin << ", " << "pw_kmaw = " <<  par_.pw_kmax << "\n";
  os << "theta_max = " <<  par_.theta_max << ", " << "apod = " <<  par_.apod << "\n";

  return os;

}//WriteParam



void Param::GetDico(const std::string& fName, Dico_t& dico){
  std::ifstream ifs (fName.c_str(), std::ifstream::in);
  if (!ifs) {
    throw AngpowError("Param::GetDico: Open error");
  }
  
  //clear 
  dico.clear();

  std::string line;

  while(std::getline(ifs, line)){
    /* test first character of the line
       . keep only upper or lower letter
    */
    if(isupper(line[0]) || islower(line[0])) {

      /*
	look if the character "=" is in the line otherwise skipt it
       */
      if(line.find("=") == std::string::npos) continue;
      
      /*
	look if the line contains a comment "#"
       */
      std::vector<std::string> tokens;
      split(line,"#",tokens);
      
      std::string data = tokens[0];
      
      /*
	extract Key, Value(s) (remove extra space)
       */
      tokens.clear();
      split(data,"=",tokens);
      std::string key = tokens[0];
      key.erase(std::remove_if(key.begin(), key.end(), IsSpace), key.end());
      std::string value = "default";
      if(tokens.size() == 2){
	value = tokens[1];
	value.erase(std::remove_if(value.begin(), value.end(), IsSpace), value.end());
      }
      
      //store in the dico
      dico.insert(std::make_pair(key,value));
    }

  }//while

  if(dico.empty()){
    throw AngpowError("Param::GetDico: Nothing read???");
  }

}//GetDico






}//namespace
