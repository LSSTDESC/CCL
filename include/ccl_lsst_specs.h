#pragma once
#include "ccl_core.h"
double ccl_specs_bias_clustering(ccl_cosmology * cosmo, double a); 
int ccl_specs_dNdz_tomog(double z, int dNdz_type, double bin_zmin, double bin_zmax,double *tomoout);

// Specifying the dNdz
// lensing (Chang et al 2013)
#define DNDZ_WL_CONS 1  //k=0.5
#define DNDZ_WL_FID 2  //k=1
#define DNDZ_WL_OPT 3 //k=2
// Clustering
#define DNDZ_NC 4

