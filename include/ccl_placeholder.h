#pragma once
#include "ccl_core.h"
double ccl_bbks_power(ccl_parameters * params, double k);
double ccl_sigmaR(gsl_spline * P, double R, int * status);
double ccl_sigma8(gsl_spline * P, double h, int * status);
double dNdz_clustering(double z);
double dNdz_sources_k1(double z, void* params);
double dNdz_sources_k2(double z, void* params);
double dNdz_sources_k0pt5(double z, void* params);
double sigmaz_clustering(double z);
double sigmaz_sources(double z);
double bias_clustering(ccl_cosmology * cosmo, double a); 
double dNdz_sources_tomog(double z, double zmin, double zmax, void * pass_params,  double (*dndz_func)(double, void*), double (*photoz_func)(double,void*, double (double, void*) ));
double photoz_dNdz(double z,void * pass_params, double (*dndz_func)(double, void*));
double norm_dNdz(double zmin, double zmax, void* pass_params,  double (*dndz_func)(double, void*));
