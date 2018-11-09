/** @file */

#ifndef __CCL_LSST_SPECS_INCLUDED__
#define __CCL_LSST_SPECS_INCLUDED__

#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>

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

CCL_BEGIN_DECLS
/** 
 * P(z) function.
 * This is a P(z) function (which can be user defined) 
 * with a void* field to contain the parameters to that function.
 */
typedef struct {
        double (* your_pz_func)(double, double, void *, int*); /*< Function returns the likelihood of measuring a z_ph
        * (first double) given a z_spec (second double), with a pointer to additonal arguments and a status flag.*/
        void *  your_pz_params; /*< Additional parameters to be passed into your_pz_func */
} pz_info;

/** 
 * dNdz function.
 * This is a dNdz function (which can be user defined)
 * with a void* field to contain the parameters to that function.
 */
typedef struct {
        double (* your_dN_func)(double, void *, int*); /*< Function returns the differential number density of galaxies wrt redshifts, 
        * with a pointer to additonal arguments and a status flag.*/
        void *  your_dN_params; /*< Additional parameters to be passed into your_dN_func */
} dNdz_info;

/**
 * dNdz smail parmas structure.
 * This is a convenience parameters structure
 * to hold the three parameters of the Smail et al. analytic dNdz
 */
 typedef struct{
		double alpha;
		double beta;
		double z0;
	} smail_params;

/** 
 * Return dNdz in a particular tomographic bin, 
    convolved with a photo-z model (defined by the user), and normalized.
 * @param z redshift 
 * @param dNdz_type the choice of dN/dz from Chang+
 * @param bin_zmin the minimum redshift of the tomorgraphic bin
 * @param bin_zmax the maximum redshift of the tomographic bin
 * @param photo_info the P(z) info struct
 * @param tomoout the output dN/dz
 * @param status Status flag. 0 if there are no errors, nonzero otherwise.
 * @return void 
 */
void ccl_dNdz_tomog(double z, double bin_zmin, double bin_zmax, pz_info * photo_info,  dNdz_info * dN_info, double *tomoout, int *status);

/** 
 * This function creates a structure amalgamating the information on an analytic true dNdz, plus some parameters.
 * @param params parameters for the analytic dNdz form
 * @param dNdz_func dNdz function
 * @return a structure with the dNdz and parameters
 */
 
dNdz_info* ccl_create_dNdz_info(void * params, double(*dNdz_func)(double,void*,int*));

/** 
 * This function creates a structure containing the true dNdz for the built-in Smail-type analytic form:
 * dNdz ~ z^alpha exp(- (z/z0)^beta)
 * @param alpha 
 * @param z0 
 * @param beta
 * @return a structure with the built-in Smail-type dNdz and parameters
 */
 dNdz_info* ccl_create_Smail_dNdz_info(double alpha, double beta, double z0);


/** Free memory holding the structure containing dNdz information.
 * @param dN_info that holds user-defined dNdz and parameters
 * @return void
 */
void ccl_free_dNdz_info(dNdz_info * dN_info);

/** 
 * This function creates a structure amalgamating the information on the photo-z model, P(z) plus some parameters.
 * @param params parameters for the P(z) function
 * @param pz_func P(z) function
 * @return a structure with the P(z) and parameters
 */
 
pz_info* ccl_create_photoz_info(void * params, double(*pz_func)(double, double,void*,int*));

/** 
 * This function creates a structure containing the photo-z model for the built-in Gaussian photo-z pdf.
 * @param sigma_z0 The photo-z uncertainty at z=0. The photo-z uncertainty is assumed to scale like (1 + z).
 * @return a structure with the built-in Gaussian P(z) and parameters
 */
pz_info* ccl_create_gaussian_photoz_info(double sigma_z0);


/** Free memory holding the structure containing user-input photoz information.
 * @param my_photoz_info that holds user-defined P(z) and parameters
 * @return void
 */
void ccl_free_photoz_info(pz_info *my_photoz_info);

CCL_END_DECLS

#endif
