#include "ccl_angpow_interface.h"

// #include "ccl_lsst_specs.h"

#define OC 0.25
#define OB 0.05
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define NORMPS 0.80
#define ZD 0.5
#define NZ 128
#define Z0_GC 0.50
#define SZ_GC 0.05
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 512

namespace Angpow {

PowerSpecCCL::PowerSpecCCL(ccl_cosmology * cosmo, double a, double kmin, double kmax) {
  double h100 = cosmo->params.h;
  std::cout<<h100<<std::endl;
}

}//namespace

// Do the computation with :
//  Pk2Cl()
//  Compute()

int main(int argc,char **argv){
  //status flag
  // int status =0;
  // Initialize cosmological parameters
  ccl_configuration config=default_config;
  config.transfer_function_method=ccl_bbks;
  ccl_parameters params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,NORMPS,NS,-1,NULL,NULL);

  // Initialize cosmology object given cosmo params
  ccl_cosmology *cosmo=ccl_cosmology_create(params,config);
  // Compute radial distances (see include/ccl_background.h for more routines)
  //printf("Comoving distance to z = %.3lf is chi = %.3lf Mpc\n",
  //	 ZD,ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status));
//printf("Luminosity distance to z = %.3lf is chi = %.3lf Mpc\n",
  //	 ZD,ccl_luminosity_distance(cosmo,1./(1+ZD), &status));
  double aref = 0.5;
  double kmin = 1e-4;
  double kmax = 1;
  Angpow::PowerSpecCCL Pk(cosmo, aref, kmin, kmax);
}
