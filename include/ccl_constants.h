#pragma once

//Spline types
#define A_SPLINE_TYPE gsl_interp_akima
#define K_SPLINE_TYPE gsl_interp_akima
#define M_SPLINE_TYPE gsl_interp_akima
#define D_SPLINE_TYPE gsl_interp_akima
#define PNL_SPLINE_TYPE gsl_interp2d_bicubic
#define PLIN_SPLINE_TYPE gsl_interp2d_bicubic

// These are in units of Mpc (no factor of h)
#define K_PIVOT 0.05

//Rho critical in units of M_sun/h / (Mpc/h)^3
#define RHO_CRITICAL 2.7744948E11

//Lightspeed / H0 in units of Mpc/h
#define CLIGHT_HMPC 2997.92458 //H0^-1 in Mpc/h

//Newton's gravitational constant
#define GNEWT 6.6738e-11    //(from PDG 2013) in m^3/Kg/s^2

//Solar mass
#define SOLAR_MASS 1.9885e30 //in kg (from PDG 2013)

//Distance conversions
#define MPC_TO_METER 3.08567758149e22  //(from PDG 2013) Mpc to m 
#define PC_TO_METER 3.08567758149e16   //(from PDG 2013) pc to m

//Precision parameters
#define EPSREL_DIST 1E-6
#define EPSREL_GROWTH 1E-6
#define EPSREL_DNDZ 1E-6
#define EPS_SCALEFAC_GROWTH 1E-6

//LSST specific numbers
#define Z_MIN_SOURCES 0.1
#define Z_MAX_SOURCES 3.0
