#pragma once

// Parameters for grids and related things
// one day to be determined by a long careful process.
// At the moment we are just thinking that:
//    the only high-z thing we need is CMB lensing
//    the contribution to this from W*P is very small above z=10 ish
// NB: Need to calculate chi_star separated from the process
// of filling these splines
#define A_SPLINE_DELTA 0.001
#define A_SPLINE_NA    1000
#define A_SPLINE_MIN   0.1
#define A_SPLINE_MAX   1.0

#define LOGM_SPLINE_DELTA 0.025
#define LOGM_SPLINE_NM    440
#define LOGM_SPLINE_MIN   6
#define LOGM_SPLINE_MAX   17

//for 2D SPLINE, e.g. P_NL, use coarser binning
#define N_A 20

#define A_SPLINE_TYPE gsl_interp_akima
#define K_SPLINE_TYPE gsl_interp_akima
#define M_SPLINE_TYPE gsl_interp_akima
#define PNL_SPLINE_TYPE gsl_interp2d_bicubic

// These are in units of Mpc (no factor of h)
#define K_PIVOT 0.05
#define K_MAX_SPLINE 500.
#define K_MAX 1e3
#define K_MIN 1e-5
#define K_MAX_INT (1e3/1.1) //minimum integration range
#define K_MIN_INT (1e-5*1.1) //maximum integration range
#define N_K 1000

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
