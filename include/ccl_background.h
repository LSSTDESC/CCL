#pragma once
#include "ccl_core.h"

//Omega_x labels
typedef enum ccl_omega_x_label {
  ccl_omega_m_label=0,
  ccl_omega_l_label=1,
  ccl_omega_g_label=2,
  ccl_omega_k_label=3
} ccl_omega_x_label;

// Normalized expansion rate at scale factor a
double ccl_h_over_h0(ccl_cosmology * cosmo, double a, int * status);
// Normalized expansion rate at scale factors as given in list a[0..na-1]
void ccl_h_over_h0s(ccl_cosmology * cosmo, int na, double a[], double output[], int * status);

// Comoving radial distance in Mpc from today to scale factor a
double ccl_comoving_radial_distance(ccl_cosmology * cosmo, double a, int* status);
// Comoving radial distances in Mpc to scale factors as given in list a[0..na-1]
void ccl_comoving_radial_distances(ccl_cosmology * cosmo, int na, double a[], double output[], int* status);

//Transforms between radial and angular distances
double ccl_sinn(ccl_cosmology *cosmo,double chi, int *status);

// Comoving angular distance in Mpc from today to scale factor a
double ccl_comoving_angular_distance(ccl_cosmology * cosmo, double a, int* status);
// Comoving angular distances in Mpc to scale factors as given in list a[0..na-1]
void ccl_comoving_angular_distances(ccl_cosmology * cosmo, int na, double a[], double output[], int* status);

// Comoving luminosity distance in Mpc from today to scale factor a
double ccl_luminosity_distance(ccl_cosmology * cosmo, double a, int * status);
// Comoving luminosity distances in Mpc to scale factors as given in list a[0..na-1]
void ccl_luminosity_distances(ccl_cosmology * cosmo, int na, double a[], double output[], int * status);

// Growth factor at scale factor a normalized to 1 at z=0
double ccl_growth_factor(ccl_cosmology * cosmo, double a, int * status);
// Growth factors at a list of scale factor given in a[0..na-1] normalized to 1 at z=0
void ccl_growth_factors(ccl_cosmology * cosmo, int na, double a[], double output[], int * status);

// Growth factor at scale factor a normalized to a in matter domination
double ccl_growth_factor_unnorm(ccl_cosmology * cosmo, double a, int * status);
// Growth factors at a list of scale factor given in a[0..na-1] normalized to a in matter domination
void ccl_growth_factors_unnorm(ccl_cosmology * cosmo, int na, double a[], double output[], int * status);

// Logarithmic rate of d ln g/d lna a at scale factor a 
double ccl_growth_rate(ccl_cosmology * cosmo, double a, int* status);
// Logarithmic rates of d ln g/d lna a at alist of  scale factor a [0..na-1]
void ccl_growth_rates(ccl_cosmology * cosmo, int na, double a[], double output[], int * status);

// Scale factor for a given comoving distance
double ccl_scale_factor_of_chi(ccl_cosmology * cosmo, double chi, int * status);
// Scale factors for a given list of comoving distances
void ccl_scale_factor_of_chis(ccl_cosmology * cosmo, int nchi, double chi[], double output[], int* status);

// Omega functions of a
double ccl_omega_x(ccl_cosmology * cosmo, double a, ccl_omega_x_label label, int* status);
