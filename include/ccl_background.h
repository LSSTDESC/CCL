#pragma once
#include "ccl_core.h"

// Normalized expansion rate at scale factor a
double ccl_h_over_h0(ccl_cosmology * cosmo, double a);
// Normalized expansion rate at scale factors as given in list a[0..na-1]
void ccl_h_over_h0s(ccl_cosmology * cosmo, int na, double a[na], double output[na]);

// Comoving radial distance in Mpc from today to scale factor a
double ccl_comoving_radial_distance(ccl_cosmology * cosmo, double a);
// Comoving radial distances in Mpc to scale factors as given in list a[0..na-1]
void ccl_comoving_radial_distances(ccl_cosmology * cosmo, int na, double a[na], double output[na]);

// Comoving luminosity distance in Mpc from today to scale factor a
double ccl_luminosity_distance(ccl_cosmology * cosmo, double a);
// Comoving luminosity distances in Mpc to scale factors as given in list a[0..na-1]
void ccl_luminosity_distances(ccl_cosmology * cosmo, int na, double a[na], double output[na]);

// Growth factor at scale factor a normalized to 1 at z=0
double ccl_growth_factor(ccl_cosmology * cosmo, double a);
// Growth factors at a list of scale factor given in a[0..na-1] normalized to 1 at z=0
void ccl_growth_factors(ccl_cosmology * cosmo, int na, double a[na], double output[na]);

// Growth factor at scale factor a normalized to a in matter domination
double ccl_growth_factor_unnorm(ccl_cosmology * cosmo, double a);
// Growth factors at a list of scale factor given in a[0..na-1] normalized to a in matter domination
void ccl_growth_factors_unnorm(ccl_cosmology * cosmo, int na, double a[na], double output[na]);

// Logarithmic rate of d ln g/d lna a at scale factor a 
double ccl_growth_rate(ccl_cosmology * cosmo, double a);
// Logarithmic rates of d ln g/d lna a at alist of  scale factor a [0..na-1]
void ccl_growth_rates(ccl_cosmology * cosmo, int na, double a[na], double output[na]);

// Scale factor for a given comoving distance
double ccl_scale_factor_of_chi(ccl_cosmology * cosmo, double chi);
// Scale factors for a given list of comoving distances
void ccl_scale_factor_of_chis(ccl_cosmology * cosmo, int nchi, double chi[nchi], double output[nchi]);
double ccl_omega_m_z(ccl_cosmology * cosmo, double a);
