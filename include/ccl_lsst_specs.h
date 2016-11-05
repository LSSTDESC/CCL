#pragma once
#include "ccl_core.h"
double dNdz_clustering(double z, void* params);
double sigmaz_clustering(double z);
double sigmaz_sources(double z);
double bias_clustering(ccl_cosmology * cosmo, double a); 
double dNdz_sources_basic(double z, void* params);
double dNdz_sources_unnormed(double z, void *params);
double photoz(double z_ph, void * params);
static double norm_integrand(double z, void* params);
double dNdz_tomog(double z, void * params, double bin_zmin, double bin_zmax, double (*dNdz)(double,void *), double (*sigmazin)(double));

// Structures for the parameters of integrands related to the WL source redshift distribution
struct dNdz_sources_params{
  int type_; // Sets which Chang et al. 2013 dNdz you are using; pick 1 for k=5, 2 for k=1, and 3 for k=2.
};
struct pz_params{
  double z_true; // Gives the true redshift at which to evaluate 
  double (*sigmaz)(double); //Calls the photo-z scatter model
};
struct norm_params{
  double bin_zmin_;
  double bin_zmax_;
  int type_;
  double (*sigmaz)(double); //Calls the photo-z scatter model
  double (*unnormedfunc)(double,void *); 
};
