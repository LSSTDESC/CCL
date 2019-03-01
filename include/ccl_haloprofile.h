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
   * @param r: radii at which to calculate output
   * @param nr: number of radii at which to calculate output
   * @param rho_r: array to store mass density at given radius for a NFW halo profile, in units of Msun/Mpc^{3}
   * @param status: Status flag: 0 if there are no errors, non-zero otherwise
   * @return void
   */
  void ccl_halo_profile_nfw (ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double* r, int nr, double* rho_r, int *status);


  /**
   * Computes projected mass density at given radius for a NFW halo profile
   * @param cosmo: cosmology object containing parameters
   * @param c: halo concentration consistent with halo size definition
   * @param halomass: halo mass
   * @param massdef_delta_m: overdensity relative to matter density for halo size definition
   * @param a: scale factor normalised to a=1 today
   * @param rp: radii at which to calculate output
   * @param nr: number of radii at which to calculate output
   * @param sigma_r: array to store projected mass density at given radius for a NFW halo profile, in units of Msun/Mpc^{2}
   * @param status: Status flag: 0 if there are no errors, non-zero otherwise
   * @return void
   */
  void ccl_projected_halo_profile_nfw (ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double* rp, int nr, double* sigma_r, int *status);


  /**
   * Computes mass density at given radius for a Einasto halo profile
   * @param cosmo: cosmology object containing parameters
   * @param c: halo concentration consistent with halo size definition
   * @param halomass: halo mass
   * @param massdef_delta_m: overdensity relative to matter density for halo size definition
   * @param a: scale factor normalised to a=1 today
   * @param r: radii at which to calculate output
   * @param nr: number of radii at which to calculate output
   * @param rho_r: array to store mass density at given radius for a Einasto halo profile, in units of Msun/Mpc^{3}
   * @param status: Status flag: 0 if there are no errors, non-zero otherwise
   * @return void
   */
  void ccl_halo_profile_einasto (ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double* r, int nr, double* rho_r, int *status);


  /**
   * Computes mass density at given radius for a Hernquist halo profile
   * @param cosmo: cosmology object containing parameters
   * @param c: halo concentration consistent with halo size definition
   * @param halomass: halo mass
   * @param massdef_delta_m: overdensity relative to matter density for halo size definition
   * @param a: scale factor normalised to a=1 today
   * @param r: radii at which to calculate output
   * @param nr: number of radii at which to calculate output
   * @param rho_r: array to store mass density at given radius for a Hernquist halo profile, in units of Msun/Mpc^{3}
   * @param status: Status flag: 0 if there are no errors, non-zero otherwise
   * @return void
   */
  void ccl_halo_profile_hernquist (ccl_cosmology *cosmo, double c, double halomass, double massdef_delta_m, double a, double* r, int nr, double* rho_r, int *status);

  CCL_END_DECLS

  #endif
