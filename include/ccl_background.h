/** @file */
#ifndef __CCL_BACKGROUND_H_INCLUDED__
#define __CCL_BACKGROUND_H_INCLUDED__

CCL_BEGIN_DECLS

//species_x labels
typedef enum ccl_species_x_label {
  ccl_species_crit_label=0,
  ccl_species_m_label=1,
  ccl_species_l_label=2,
  ccl_species_g_label=3,
  ccl_species_k_label=4,
  ccl_species_ur_label=5,
  ccl_species_nu_label=6,
} ccl_species_x_label;

/**
 * Normalized expansion rate at scale factor a.
 * Returns H(a)/H0 in a given cosmology.
 * @param cosmo Cosmological parameters
 * @param a scale factor, normalized to 1 for today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return h_over_h0, the value of H(a)/H0.
 */
double ccl_h_over_h0(ccl_cosmology * cosmo, double a, int * status);

/**
 * Normalized expansion rate at scale factors as given in list a[0..na-1]
 * Returns H(a)/H0 for an array of scale factors a of length na.
 * @param cosmo Cosmological parameters
 * @param na Number of scale factors in a
 * @param a array of scale factors
 * @param output array of length na to store the results of the calculation. The entry at index i stores H(a[i])/H0
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_h_over_h0s(ccl_cosmology * cosmo, int na, double a[], double output[], int * status);

/**
 * Comoving radial distance in Mpc from today to scale factor a
 * @param cosmo Cosmological parameters
 * @param a scale factor, normalized to 1 for today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return comoving_radial_distance, Comoving radial distance in Mpc
 */
double ccl_comoving_radial_distance(ccl_cosmology * cosmo, double a, int* status);

/**
 * Comoving radial distances in Mpc to scale factors as given in list a[0..na-1]
 * @param cosmo Cosmological parameters
 * @param na Number of scale factors in a
 * @param a array of scale factors
 * @param output array of length na to store the results of the calculation. The entry at index i stores the
 * distance for a[i].
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_comoving_radial_distances(ccl_cosmology * cosmo, int na, double a[], double output[], int* status);

/**
 * Transforms between radial and transverse comoving distances
 * Calculate the comoving radial distance of two objects with comoving radial distance chi via:
 *          { sin(x)  , if k==1
 *  sinn(x)={  x      , if k==0
 *          { sinh(x) , if k==-1
 * @param cosmo Cosmological parameters
 * @param chi Comoving radial distance of two objects
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return chi_t, the transverse comoving distance
 */
double ccl_sinn(ccl_cosmology *cosmo,double chi, int *status);

/**
 * Comoving angular distance in Mpc from today to scale factor a
 * NOTE this quantity is otherwise known as the transverse comoving distance, and is NOT angular diameter
 * distance or angular separation
 * @param cosmo Cosmological parameters
 * @param a scale factor, normalized to 1 for today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return comoving_angular_distance, the angular distance in Mpc
 */
double ccl_comoving_angular_distance(ccl_cosmology * cosmo, double a, int* status);

/**
 * Comoving angular distances in Mpc to scale factors as given in array a[0..na-1]
 * NOTE this quantity is otherwise known as the transverse comoving distance, and is NOT angular diameter
 * distance or angular separation
 * @param cosmo Cosmological parameters
 * @param na Number of scale factors in a
 * @param a array of scale factors
 * @param output array of length na to store the results of the calculation. The entry at index i stores the
 * distance for a[i].
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_comoving_angular_distances(ccl_cosmology * cosmo, int na, double a[], double output[], int* status);

/**
 * Comoving luminosity distance in Mpc from today to scale factor a
 * @param cosmo Cosmological parameters
 * @param a scale factor, normalized to 1 for today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return luminosity_distance, the angular distance in Mpc
 */
double ccl_luminosity_distance(ccl_cosmology * cosmo, double a, int * status);

/**
 * Comoving luminosity distances in Mpc to scale factors as given in array a[0..na-1]
 * @param cosmo Cosmological parameters
 * @param na Number of scale factors in a
 * @param a array of scale factors
 * @param output array of length na to store the results of the calculation. The entry at index i stores the
 * distance for a[i].
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_luminosity_distances(ccl_cosmology * cosmo, int na, double a[], double output[], int * status);

/**
 * Distance modulus for object at scale factor a. Note the factor of 6 arises from the conversion from Mpc to pc.
 * @param cosmo Cosmological parameters
 * @param a scale factor, normalized to 1 today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return distance modulus
 */
double ccl_distance_modulus(ccl_cosmology * cosmo, double a, int * status);

/**
 * Distance moduli for objects at scale factors as given in list a[0..na-1]. Note the factor of 6 arises from the conversion from Mpc to pc.
 * @param cosmo Cosmological parameters
* @param na Number of scale factors in a
 * @param a array of scale factors
 * @param output array of length na to store the results of the calculation. The entry at index i stores the
 * distance for a[i].
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
*/
void ccl_distance_moduli(ccl_cosmology * cosmo, int na, double a[], double output[], int * status);


/**
 * Growth factor at scale factor a, where g(z=0) is normalized to 1
 * @param cosmo Cosmological parameters
 * @param a scale factor, normalized to 1 for today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return growth_factor, growth factor at a
 */
double ccl_growth_factor(ccl_cosmology * cosmo, double a, int * status);

/**
 * Growth factors at an array of scale factor given in a[0..na-1], where g(z=0) is normalized to 1
 * @param cosmo Cosmological parameters
 * @param na Number of scale factors in a
 * @param a array of scale factors
 * @param output array of length na to store the results of the calculation. The entry at index i stores the
 * distance for a[i].
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_growth_factors(ccl_cosmology * cosmo, int na, double a[], double output[], int * status);

/**
 * Growth factor at scale factor a, where g(a) is normalized to a in matter domination
 * @param cosmo Cosmological parameters
 * @param a scale factor, normalized to 1 for today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return growth_factor_unnorm, Unnormalized growth factor, normalized to the scale factor at early times.
 */
double ccl_growth_factor_unnorm(ccl_cosmology * cosmo, double a, int * status);

/**
 * Growth factors at a list of scale factor given in a[0..na-1], where g(a) is normalized to a in matter domination
 * @param cosmo Cosmological parameters
 * @param na Number of scale factors in a
 * @param a array of scale factors
 * @param output array of length na to store the results of the calculation. The entry at index i stores the
 * distance for a[i].
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_growth_factors_unnorm(ccl_cosmology * cosmo, int na, double a[], double output[], int * status);

/**
 * Logarithmic rate of d ln(g)/d ln(a)  at scale factor a
 * @param cosmo Cosmological parameters
 * @param a scale factor, normalized to 1 for today
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return f, the growth rate at a
 */
double ccl_growth_rate(ccl_cosmology * cosmo, double a, int* status);

/**
 * Logarithmic rates of d ln(g)/d ln(a) at an array of scale factors a[0..na-1]
 * @param cosmo Cosmological parameters
 * @param na Number of scale factors in a
 * @param a array of scale factors
 * @param output array of length na to store the results of the calculation. The entry at index i stores the
 * distance for a[i].
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_growth_rates(ccl_cosmology * cosmo, int na, double a[], double output[], int * status);

/**
 * Scale factor for a given comoving distance (in Mpc)
 * @param cosmo Cosmological parameters
 * @param chi Comoving distance in Mpc
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return a, scale factor of distance xi
 */
double ccl_scale_factor_of_chi(ccl_cosmology * cosmo, double chi, int * status);

/**
 * Scale factors for a given array of comoving distances chi[0..nchi-1]
 * @param cosmo Cosmological parameters
 * @param nchi Number of chis in chi
 * @param chi array of comoving distances
 * @param output array of length na to store the results of the calculation. The entry at index i stores the
 * scale factor for chi[i].
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_scale_factor_of_chis(ccl_cosmology * cosmo, int nchi, double chi[], double output[], int* status);

/**
 * Physical density (rho) as a function of scale factor.  Critical density is defined as rho_critical = 3 H^2(a)/ (8 pi G). Density of a given species is then rho_x = Omega_x(a) rho_critical(a). For example, rho_matter(a) = Omega_m  a^{-3} / (H^2/H0^2)  3H^2 / (8 pi G) =  Omega_m  a^{-3}  3H0^2 / (8 pi G) =  Omega_m a^{-3}  rho_critical_present. Units of M_sun/(Mpc)^3.
 * @param cosmo Cosmological parameters
 * @param a scale factor, normalized to 1 for today
 * @param label species type. Available: 'critical'(0), 'matter'(1), 'dark_energy'(2), 'radiation'(3), 'curvature'(4), 'massless neutrinos'(5), 'massive neutrinos'(6).
 * @param int is_comoving. 0 for physical densities, and nonzero for comoving densities (via a^3 factor).
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.
 * @return rho_x, physical density at scale factor a.
 */
double ccl_rho_x(ccl_cosmology * cosmo, double a, ccl_species_x_label label, int is_comoving, int* status);

/**
 * Density fraction of a given species at a redshift different than z=0.
 * @param cosmo Cosmological parameters
 * @param a scale factor, normalized to 1 for today
 * @param label species type. Available: 'matter'(1), 'dark_energy'(2), 'radiation'(3), 'curvature'(4), 'massless neutrinos'(5), 'massive neutrinos'(6).
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return omega_x, Density fraction of a given species at scale factor a.
 */
double ccl_omega_x(ccl_cosmology * cosmo, double a, ccl_species_x_label label, int* status);

/**
 * Compute comoving distances and spline to be stored
 * in the cosmology structure.
 * @param cosmo Cosmological parameters
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_cosmology_compute_distances(ccl_cosmology * cosmo,int *status);

/**
 * Compute the growth function and a spline to be stored
 * in the cosmology structure.
 * @param cosmo Cosmological parameters
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.c
 * @return void
 */
void ccl_cosmology_compute_growth(ccl_cosmology * cosmo, int * status);

CCL_END_DECLS

#endif
