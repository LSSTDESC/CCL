/** @file */
#ifdef __cplusplus
extern "C" {
#endif

#pragma once

#include "ccl_core.h"


  /**
   * Computes mass density at given radius for a NFW halo profile
   * @param cosmo: cosmology object containing parameters
   * @param c: halo concentration consistent with halo size definition
   * @param halomass: halo mass
   * @param massdef_delta_m: overdensity relative to matter density for halo size definition
   * @param a: scale factor normalised to a=1 today
   * @param r: radius at which to calculate output
   * @param status: Status flag: 0 if there are no errors, non-zero otherwise
   * @return rho_r: mass density at given radius for a NFW halo profile, in units of Msun/Mpc^{3}
   */
  double ccl_halo_profile_nfw(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double r, int *status);

  /**
   * Computes surface mass density at given projected radius for a NFW halo profile
   * @param cosmo: cosmology object containing parameters
   * @param c: halo concentration consistent with halo size definition
   * @param halomass: halo mass
   * @param massdef_delta_m: overdensity relative to matter density for halo size definition
   * @param a: scale factor normalised to a=1 today
   * @param rp: projected radius at which to calculate output
   * @param status: Status flag: 0 if there are no errors, non-zero otherwise
   * @return sigma: nsurface mass density integrated along line of sight at given projected radius for a NFW halo profile, in units of Msun/Mpc^{2} 
   */
  double ccl_projected_halo_profile_nfw(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double rp, int *status);



#ifdef __cplusplus
}
#endif
