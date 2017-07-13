#pragma once


#include "gsl/gsl_const_mksa.h"

//Spline types
#define A_SPLINE_TYPE gsl_interp_akima
#define K_SPLINE_TYPE gsl_interp_akima
#define L_SPLINE_TYPE gsl_interp_akima
#define M_SPLINE_TYPE gsl_interp_akima
#define D_SPLINE_TYPE gsl_interp_akima
#define PNL_SPLINE_TYPE gsl_interp2d_bicubic
#define PLIN_SPLINE_TYPE gsl_interp2d_bicubic
#define CORR_SPLINE_TYPE gsl_interp_akima

/** @file */

/**
 *  k pivot. These are in units of Mpc (no factor of h)
*/
#define K_PIVOT 0.05

/** 
 * Rho critical in units of M_sun/h / (Mpc/h)^3
 */
#define RHO_CRITICAL 2.7744948E11

/**
 * Lightspeed / H0 in units of Mpc/h
 */
#define CLIGHT_HMPC 2997.92458 //H0^-1 in Mpc/h

/**
 * Newton's gravitational constant in units of m^3/Kg/s^2 (from PDG 2013)
 */
//#define GNEWT 6.6738e-11    //(from PDG 2013) in m^3/Kg/s^2
#define GNEWT 6.67428e-11 // CLASS VALUE

/**
 * Solar mass in units ofkg (from PDG 2013)
 */
#define SOLAR_MASS 1.9885e30

/**
 * Mpc to meters (from PDG 2013)
 */
#define MPC_TO_METER 3.08567758149e22

/**
 * pc to meters (from PDG 2013)
 */
#define PC_TO_METER 3.08567758149e16

/**
 * Boltzmann constant in units of J/K
*/
#define KBOLTZ  GSL_CONST_MKSA_BOLTZMANN

/**
 * Stefan-Boltzmann constant in units of kg/s^3 / K^4
 */
#define STBOLTZ GSL_CONST_MKSA_STEFAN_BOLTZMANN_CONSTANT

/**
 * Planck's constant in units kg m^2 / s
 */
#define HPLANCK  GSL_CONST_MKSA_PLANCKS_CONSTANT_H 

/**
 * The speed of light in m/s
 */
#define CLIGHT   GSL_CONST_MKSA_SPEED_OF_LIGHT

/**
 * Electron volt to Joules convestion
 */
#define EV_IN_J  GSL_CONST_MKSA_ELECTRON_VOLT

/**
 * T_ncdm, as taken from CLASS, explanatory.ini
 */
#define TNCDM 0.71611

//Precision parameters
/**
 * Relative precision in distance calculations
 */
#define EPSREL_DIST 1E-6

/**
 * Relative precision in growth calculations
 */
#define EPSREL_GROWTH 1E-6

/**
 * Relative precision in dNdz calculations
 */
#define EPSREL_DNDZ 1E-6

/**
 * Absolute precision in growth calculations
 */
#define EPS_SCALEFAC_GROWTH 1E-6

//Correlation function related parameters
#define EPSREL_CORR_FUNC 1E-3
#define GSL_INTEGRATION_LIMIT 1000

//LSST specific numbers
#define Z_MIN_SOURCES 0.1
#define Z_MAX_SOURCES 3.0
