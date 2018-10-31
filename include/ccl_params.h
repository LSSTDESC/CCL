/** @file */
#include "ccl_constants.h"

#ifndef __CCL_PARAMS_H_INCLUDED__
#define __CCL_PARAMS_H_INCLUDED__

CCL_BEGIN_DECLS
/**
 * Struct that contains all the parameters needed to create certain splines.
 * This includes splines for the scale factor, masses, and power spectra.
 */
typedef struct ccl_spline_params {
   //Scale factor splines
  int  A_SPLINE_NA;
  double A_SPLINE_MIN;
  double A_SPLINE_MINLOG_PK;
  double A_SPLINE_MIN_PK;
  double  A_SPLINE_MAX;
  double A_SPLINE_MINLOG;
  int A_SPLINE_NLOG;

  //Mass splines
  double LOGM_SPLINE_DELTA;
  int LOGM_SPLINE_NM;
  double LOGM_SPLINE_MIN;
  double LOGM_SPLINE_MAX;

  //PS a and k spline
  int A_SPLINE_NA_PK;
  int A_SPLINE_NLOG_PK;

  //k-splines and integrals
  double K_MAX_SPLINE;
  double K_MAX;
  double K_MIN;
  int N_K;
  int N_K_3DCOR;

  //Correlation function parameters
  double ELL_MIN_CORR;
  double ELL_MAX_CORR;
  int N_ELL_CORR;
} ccl_spline_params;

extern ccl_spline_params * ccl_splines;

int ccl_get_pk_spline_na(void);
int ccl_get_pk_spline_nk(void);
void ccl_get_pk_spline_a_array(int ndout,double* doutput,int *status);
void ccl_get_pk_spline_lk_array(int ndout,double* doutput,int *status);

/**
 * Struct that contains parameters that control the accuracy of various GSL
 * routines.
 */
typedef struct ccl_gsl_params {
  // General parameters. If not otherwise specified, those will be copied to the
  // more specialised cases.
  double EPSREL;
  size_t N_ITERATION;

  // Integration
  int INTEGRATION_GAUSS_KRONROD_POINTS;
  double INTEGRATION_EPSREL;
  // Limber integration
  int INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS;
  double INTEGRATION_LIMBER_EPSREL;
  // Distance integrals
  double INTEGRATION_DISTANCE_EPSREL;
  // dndz integrals
  double INTEGRATION_DNDZ_EPSREL;
  // sigma_R integral
  double INTEGRATION_SIGMAR_EPSREL;
  // Neutrino integral
  double INTEGRATION_NU_EPSREL;
  double INTEGRATION_NU_EPSABS;

  // Root finding
  double ROOT_EPSREL;
  int ROOT_N_ITERATION;

  // ODE
  double ODE_GROWTH_EPSREL;

} ccl_gsl_params;

extern ccl_gsl_params * ccl_gsl;

CCL_END_DECLS

#endif
