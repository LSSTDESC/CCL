/** @file */
#ifdef __cplusplus
extern "C" {
#endif

#pragma once

#define HM_MMIN 1e7 // Minimum mass for the halo-model integration
#define HM_MMAX 1e17 // Maximum mass for the halo-model integration
#define HM_EPSABS 0 // Absolute error for the halo-model integration
#define HM_EPSREL 1E-4 // Relative error for the halo-model integration
#define HM_LIMIT 1000 // Maximum sub intervals for the halo-model integration
#define HM_INT_METHOD GSL_INTEG_GAUSS41 // Integration scheme for halo-model integration
  
#include "ccl_core.h"  

  // concentration-mass relation
  typedef enum ccl_conc_label {
    Bhattacharya2011 = 1,
    Duffy2008_virial = 2,
    constant = 3,
  } ccl_conc_label;

  // halo window profiles
  typedef enum ccl_win_label {
    NFW = 1,
  } ccl_win_label;

  /**
   * Computes the halo model density-density power spectrum two-halo term.
   * @param cosmo: cosmology object containing parameters
   * @param k: wavenumber in units of Mpc^{-1}
   * @param a: scale factor normalised to a=1 today
   * @param status: Status flag: 0 if there are no errors, non-zero otherwise
   * @return 2halo_matter_power: halo-model two-halo matter power spectrum, P(k), units of Mpc^{3}
   */
  double ccl_twohalo_matter_power(ccl_cosmology *cosmo, double k, double a, int *status);
  
  /**
   * Computes the halo model density-density power spectrum one-halo term.
   * @param cosmo: cosmology object containing parameters
   * @param k: wavenumber in units of Mpc^{-1}
   * @param a: scale factor normalised to a=1 today
   * @param status: Status flag: 0 if there are no errors, non-zero otherwise
   * @return 1halo_matter_power: halo-model one-halo matter power spectrum, P(k), units of Mpc^{3}
   */
  double ccl_onehalo_matter_power(ccl_cosmology *cosmo, double k, double a, int *status);

  /**
   * Computes the halo model density-density power spectrum as the sum of two- and one-halo terms.
   * @param cosmo: cosmology object containing parameters
   * @param k: wavenumber in units of Mpc^{-1}
   * @param a: scale factor normalised to a=1 today
   * @param status: Status flag: 0 if there are no errors, non-zero otherwise
   * @return halomodel_matter_power: halo-model power spectrum, P(k), units of Mpc^{3}
   */
  double ccl_halomodel_matter_power(ccl_cosmology *cosmo, double k, double a, int *status);

  /**
   * Computes the concentration of a halo of mass M. This is the ratio of virial raidus to scale radius for
   * an NFW halo
   * @param cosmo: cosmology object containing parameters
   * @param halomass: halo mass in units of Msun
   * @param a: scale factor normalised to a=1 today
   * @param Delta_v: overdensity criteria (with respect to matter density) used for halo mass
   * @param status: Status flag: 0 if there are no errors, non-zero otherwise
   * @return halo_concentration: the halo concentration
   */
  double halo_concentration(ccl_cosmology *cosmo, double halomass, double a, double Delta_v, ccl_conc_label label, int *status);
  
#ifdef __cplusplus
}
#endif
