#include "ccl.h"
#include "ccl_correlation.h"
#include "ccl_utils.h"
#include "ctest.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Define cosmological parameter values
#define OC 0.25
#define OB 0.05
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define NORMPS 0.80
#define ZD 0.5
#define NZ 128
#define Z0_GC 0.50
#define SZ_GC 0.05
#define NL 512
#define PS 0.1 
#define NREL 3.046
#define NMAS 0
#define MNU 0.0
#define ELL_MAX_CL 20000


int main(int argc,char **argv)
{
  // Use the default configuration, plus the cosmological parameters that were 
  // defined above
  int status=0;
  ccl_configuration config = default_config;
  ccl_parameters params = ccl_parameters_create(OC, OB, OK, NREL, NMAS, MNU, 
                                                W0, WA, HH, NORMPS, NS,
						0, NULL, NULL, &status);
  ccl_cosmology *cosmo = ccl_cosmology_create(params,config);

  // Create example number density and bias for tracer
  double z_arr_gc[NZ], nz_arr_gc[NZ], bz_arr[NZ];
  for(int i=0; i < NZ; i++) {
    z_arr_gc[i] = Z0_GC - 5.*SZ_GC + 10.*SZ_GC*(i + 0.5)/NZ;
    nz_arr_gc[i] = exp(-0.5*pow((z_arr_gc[i] - Z0_GC) / SZ_GC, 2));
    bz_arr[i] = 1. + z_arr_gc[i];
  }

  // Define a galaxy clustering tracer and calculate C_ell's
  CCL_ClTracer *ct_gc = ccl_cl_tracer_number_counts_simple_new(cosmo, 
							       NZ, z_arr_gc, nz_arr_gc,
							       NZ, z_arr_gc, bz_arr,
							       &status);
  int il;
  double *clarr = malloc(ELL_MAX_CL*sizeof(double));
  double *larr = malloc(ELL_MAX_CL*sizeof(double));
  for(il=0; il < ELL_MAX_CL; il++){
    // Calculate auto-correlation of this tracer
    larr[il] = il;
    clarr[il] = ccl_angular_cl(cosmo, il, ct_gc, ct_gc, &status);
  }
  
  // Define cosine tapering, to reduce ringing. The first two numbers are 
  // [lmin, lmax] for the low-ell taper, and the last two are [lmin, lmax] for 
  // the high-ell taper.
  double taper_cl_limits[4] = {1, 2, 10000, 15000};
  
  double *clustering_corr, *theta;
  int ntheta = 15;
  theta = ccl_log_spacing(0.01, 5., ntheta); // New array with log spacing
  clustering_corr = malloc(ntheta*sizeof(double));
  
  // Calculate correlation function from angular power spectrum
  ccl_correlation(cosmo, ELL_MAX_CL, larr, clarr, 
                  ntheta, theta, clustering_corr, 
                  CCL_CORR_GG, 0, taper_cl_limits, 
                  CCL_CORR_FFTLOG, &status);
  
  // Print results
  for(int it=0; it < ntheta; it++)
    printf("%le %le\n", theta[it], clustering_corr[it]);
  
  // Free tracers and other allocated memory
  ccl_cl_tracer_free(ct_gc);
  ccl_cosmology_free(cosmo);
  free(clustering_corr);
  free(larr);
  free(clarr);

  return 0;
}
