/** @file */

#ifndef __CCL_HALOPROFILE_H_INCLUDED__
#define __CCL_HALOPROFILE_H_INCLUDED__

CCL_BEGIN_DECLS


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
   * Computes mass density at given radius for a Einasto halo profile
   * @param cosmo: cosmology object containing parameters
   * @param c: halo concentration consistent with halo size definition
   * @param halomass: halo mass
   * @param massdef_delta_m: overdensity relative to matter density for halo size definition
   * @param a: scale factor normalised to a=1 today
   * @param r: radius at which to calculate output
   * @param status: Status flag: 0 if there are no errors, non-zero otherwise
   * @return rho_r: mass density at given radius for an Einasto halo profile, in units of Msun/Mpc^{3}
   */
  double ccl_halo_profile_einasto(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double r, int *status);


  /**
   * Computes mass density at given radius for a Hernquist halo profile
   * @param cosmo: cosmology object containing parameters
   * @param c: halo concentration consistent with halo size definition
   * @param halomass: halo mass
   * @param massdef_delta_m: overdensity relative to matter density for halo size definition
   * @param a: scale factor normalised to a=1 today
   * @param r: radius at which to calculate output
   * @param status: Status flag: 0 if there are no errors, non-zero otherwise
   * @return rho_r: mass density at given radius for an Einasto halo profile, in units of Msun/Mpc^{3}
   */
  double ccl_halo_profile_hernquist(ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double r, int *status);

  CCL_END_DECLS

  #endif
