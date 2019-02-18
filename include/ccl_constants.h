#ifndef __CCL_CONSTANTS_H_INCLUDED__
#define __CCL_CONSTANTS_H_INCLUDED__

#include <gsl/gsl_const_mksa.h>

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
 * Lightspeed / H0 in units of Mpc/h (from CODATA 2014)
 */
#define CLIGHT_HMPC 2997.92458 //H0^-1 in Mpc/h

/**
 * Newton's gravitational constant in units of m^3/Kg/s^2
 */
//#define GNEWT 6.6738e-11  /(from PDG 2013) in m^3/Kg/s^2
//#define GNEWT 6.67428e-11 // CLASS VALUE
#define GNEWT 6.67408e-11 // from CODATA 2014

/**
 * Solar mass in units of kg (from GSL)
 */
//#define SOLAR_MASS GSL_CONST_MKSA_SOLAR_MASS
//#define SOLAR_MASS 1.9885e30 //(from PDG 2015) in Kg
#define SOLAR_MASS 1.9884754153381438e+30 //from IAU 2015
/**
 * Mpc to meters (from PDG 2013)
 */
#define MPC_TO_METER 3.08567758149e22

/**
 * pc to meters (from PDG 2013)
 */
#define PC_TO_METER 3.08567758149e16

/**
 * Rho critical in units of M_sun/h / (Mpc/h)^3
 */
#define RHO_CRITICAL ((3*100*100)/(8*M_PI*GNEWT)) * (1000*1000*MPC_TO_METER/SOLAR_MASS)

/**
 * Boltzmann constant in units of J/K
*/
//#define KBOLTZ  GSL_CONST_MKSA_BOLTZMANN
#define KBOLTZ 1.38064852e-23 //from CODATA 2014

/**
 * Stefan-Boltzmann constant in units of kg/s^3 / K^4
 */
//#define STBOLTZ GSL_CONST_MKSA_STEFAN_BOLTZMANN_CONSTANT
#define STBOLTZ 5.670367e-8 //from CODATA 2014
/**
 * Planck's constant in units kg m^2 / s
 */
//#define HPLANCK  GSL_CONST_MKSA_PLANCKS_CONSTANT_H
#define HPLANCK 6.626070040e-34 //from CODATA 2014

/**
 * The speed of light in m/s
 */
//#define CLIGHT   GSL_CONST_MKSA_SPEED_OF_LIGHT
#define CLIGHT 299792458.0 //from CODATA 2014

/**
 * Electron volt to Joules convestion
 */
//#define EV_IN_J  GSL_CONST_MKSA_ELECTRON_VOLT
#define EV_IN_J 1.6021766208e-19  //from CODATA 2014

/**
 * Temperature of the CMB in K
 */
#define TCMB 2.725
//#define TCMB 2.7255 // CLASS value

/**
 * T_ncdm, as taken from CLASS, explanatory.ini
 */
#define TNCDM 0.71611

/**
 * neutrino mass splitting differences
 * See Lesgourgues and Pastor, 2012 for these values.
 * Adv. High Energy Phys. 2012 (2012) 608515,
 * arXiv:1212.6154, page 13
*/
#define DELTAM12_sq 7.62E-5
#define DELTAM13_sq_pos 2.55E-3
#define DELTAM13_sq_neg -2.43E-3

#endif
