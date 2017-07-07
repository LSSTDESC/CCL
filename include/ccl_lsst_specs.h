/** @file */

#pragma once
#include "ccl_core.h"
#include "math.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"

/** 
 * A user-defined P(z) function.
 * This is a user-defined P(z) function, 
 * with a void* field to contain the parameters to that function.
 */
typedef struct {
  double (* your_pz_func)(double, double, void *, int*); /*< Function returns the liklihood of measuring a z_ph
							  * (first double) given a z_spec (second double), with a pointer to additonal arguments and a status flag.*/
  void *  your_pz_params; /*< Additional parameters to be passed into your_pz_func */
} user_pz_info;

/**
 * Compute b(a), the bias of the clustering sample of a cosmology at a given scale factor
 * This is input from LSS group.
 * @param cosmo Cosmological parameters
 * @param a scale factor, normalized to a=1 today.
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * For specific cases see documentation for ccl_error.
 * @return b, the bias at a in cosmo
 */
double ccl_specs_bias_clustering(ccl_cosmology * cosmo, double a, int * status);

/** 
 * Return dNdz in a particular tomographic bin, 
    convolved with a photo-z model (defined by the user), and normalized.
 * @param z redshift 
 * @param dNdz_type the choice of dN/dz from Chang+
 * @param bin_zmin the minimum redshift of the tomorgraphic bin
 * @param bin_zmax the maximum redshift of the tomographic bin
 * @param user_info the user P(z) info struct
 * @param tomoout the output dN/dz
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * @return void 
 */
void ccl_specs_dNdz_tomog(double z, int dNdz_type, double bin_zmin, double bin_zmax, user_pz_info * user_info,  double *tomoout, int *status);

/** 
 * This function creates a structure amalgamating the user-input information on the photo-z model, P(z) plus some parameters.
 * @param user_params User-defined parameters for the P(z) function
 * @param user_pz_func P(z) function
 * @return a structure with the user-provided P(z) and parameters
 */
user_pz_info* ccl_specs_create_photoz_info(void * user_params, double(*user_pz_func)(double, double,void*,int*));

/** Free memory holding the structure containing user-input photoz information.
 * @param my_photoz_info that holds user-defined P(z) and parameters
 * @return void
 */
void ccl_specs_free_photoz_info(user_pz_info *my_photoz_info);

/** 
 * Return sigma(z), the photo-z dispersion, for the clustering sample
   This is if you want to assume Gaussian uncertainties.
 *  @param z redshift
 *  @return sigma(z) for the clustering sample
 */
double ccl_specs_sigmaz_clustering(double z);

/** 
 * Return sigma(z), the photo-z dispersion, for the lensing sample
   This is if you want to assume Gaussian uncertainties.
 *  @param z redshift
 *  @return sigma(z) for the lensing sample
 */
double ccl_specs_sigmaz_sources(double z);

// Specifying the dNdz
// lensing (Chang et al 2013)
#define DNDZ_WL_CONS 1  //k=0.5
#define DNDZ_WL_FID 2  //k=1
#define DNDZ_WL_OPT 3 //k=2
// Clustering
#define DNDZ_NC 4

//LSST redshift range for lensing sources
#define Z_MIN_SOURCES 0.1
#define Z_MAX_SOURCES 3.0
