#pragma once
#include "ccl_core.h"
#include "math.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"

typedef struct {
        double (* your_pz_func)(double, double, void *, int*); /*< Function returns the liklihood of measuring a z_ph
 * (first double) given a z_spec (second double), with a pointer to additonal arguments and a status flag.*/
        void *  your_pz_params; /*< Additional parameters to be passed into your_pz_func */
} user_pz_info;
/*<
 * User defined photoz function and information
 */

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
 * dNdz in a particular tomographic bin,
   convolved with a photo-z model (defined by the user), and normalized.
   returns a status integer 0 if called with an allowable type of dNdz, non-zero otherwise
   (this is different from the regular status handling procedure because we don't pass a cosmology to this function)
 * @param z redshift to compute
 * @param dNdz_type
 * @param bin_zmin
 * @param bin_zmax
 * @param user_info
 * @param tomoout
 * @param status
 */
void ccl_specs_dNdz_tomog(double z, int dNdz_type, double bin_zmin, double bin_zmax, user_pz_info * user_info,  double *tomoout, int *status);
user_pz_info* ccl_specs_create_photoz_info(void * user_params, double(*user_pz_func)(double, double,void*,int*));
void ccl_specs_free_photoz_info(user_pz_info *my_photoz_info);
double ccl_specs_sigmaz_clustering(double z);
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

