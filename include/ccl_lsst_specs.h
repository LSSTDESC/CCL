#pragma once
#include "ccl_core.h"
#include "math.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"

typedef struct {
        double (* your_pz_func)(double, double, void *, int*); //first double corresponds to photo-z, second to spec-z
        void *  your_pz_params;
} user_pz_info;

double ccl_specs_bias_clustering(ccl_cosmology * cosmo, double a, int * status); 
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

