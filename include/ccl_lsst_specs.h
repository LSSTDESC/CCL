#pragma once
#include "ccl_core.h"
#include "math.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"

typedef struct {
        double (* your_pz_func)(double, double, void *);
        void *  your_pz_params;
} user_pz_info;

double ccl_specs_bias_clustering(ccl_cosmology * cosmo, double a); 
int ccl_specs_dNdz_tomog(double z, int dNdz_type, double bin_zmin, double bin_zmax, user_pz_info * user_info,  double *tomoout);

// Specifying the dNdz
// lensing (Chang et al 2013)
#define DNDZ_WL_CONS 1  //k=0.5
#define DNDZ_WL_FID 2  //k=1
#define DNDZ_WL_OPT 3 //k=2
// Clustering
#define DNDZ_NC 4

