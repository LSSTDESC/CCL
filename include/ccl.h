/** @file */

#ifndef __CCL_H_INCLUDED__
#define __CCL_H_INCLUDED__

/*! \mainpage Core Cosmology Library
 * The Core Cosmology Library (CCL) provides routines to compute basic cosmological observables with validated numerical accuracy. In the current version, predictions are provided for distances and background quantities, an
 gular auto- and cross-spectra of cosmic shear and clustering, correlation functions and the halo mass function. Fiducial specifications for the expected LSST galaxy distributions and clustering bias are also included, together with a facility to compute redshift distributions for a user-defined photometric redshift model. CCL is written in C with a Python interface.
 *
 *\section Installation
 *For installation instructions, see the README file.
 *
 *\section Contribute
 *If you plan to contribute new features to CCL, please see our CONTRIBUTE file for instructions.
 *
 *\section License
 *CCL is released under the license provided in the LICENSE file.
 *
 *\section Repository
 *For the latest version of CCL, please visit the github repository: https://github.com/LSSTDESC/CCL
 */

#include "ccl_defs.h"
#include "ccl_utils.h"
#include "ccl_fftlog.h"
#include "ccl_f1d.h"
#include "ccl_f2d.h"
#include "ccl_f3d.h"
#include "ccl_config.h"
#include "ccl_core.h"
#include "ccl_error.h"
#include "ccl_power.h"
#include "ccl_tracers.h"
#include "ccl_cls.h"
#include "ccl_background.h"
#include "ccl_correlation.h"
#include "ccl_massfunc.h"
#include "ccl_neutrinos.h"
#include "ccl_bbks.h"
#include "ccl_eh.h"
#include "ccl_halofit.h"
#include "ccl_musigma.h"
#include "ccl_mass_conversion.h"

CCL_BEGIN_DECLS
/* add function and variable declarations here */
CCL_END_DECLS

#endif
