*******************
Navigating the Code
*******************

The CCL package is laid out as follows.

- ``.travis/``: helper scripts for running tests on Travis-CI
- ``benchmarks/``: source code and data for the benchmark tests, see :ref:`benchmarks`
  for more details.
- ``cmake/``: ``CMake`` modules for building the CCL ``C`` layer
- ``doc/``: latex source for the CCL note and the CCL paper
- ``examples/``: (possible outdated) examples of how to use CCL
- ``include/``: CCL ``C`` layer header files
- ``pyccl/``: the CCL ``Python`` package
- ``readthedocs/``: the CCL ``Read the Docs`` source
- ``src/``: the CCL ``C`` layer source code
- ``setup.py``: the ``Python`` install script
- ``CMakeLists.txt``: the ``CMake`` installation configuration
- ``.travis.yml``: the Travis-CI configuration
- ``CHANGELOG.md``: the CCL log of changes to the code


Locations of Core Cosmological Quantities
-----------------------------------------

Here we describe briefly the locations of the computations of core cosmological
quantities. Usually each of these ``C`` files has a corresponding ``Python`` file,
``SWIG`` interface file, and ``C`` header file. However this is not always true
for various reasons.

- ``ccl_background.c``: computations of growth functions, the Hubble function
  and distances
- ``ccl_core.c``: functions to handle cosmology structures and parameters
- ``ccl_error.c``: functions to help handle ``C``-layer errors
- ``ccl_bbks.c``: the BBKS transfer function
- ``ccl_f1d.c``: 1-d interpolations in ``C``
- ``ccl_f2d.c``: 2-d interpolations in ``C``
- ``ccl_power.c``: code to spline and initialize the power spectra and
  transfer functions
- ``ccl_bcm.c``: code to compute the BCM model for baryonic effects
- ``ccl_eh.c``: code to compute the Eisenstein and Hu (1998) transfer function
  approximation
- ``ccl_musigma.c``: code to properly normalize input linear power spectra for the
  ``mu-Sigma`` modified gravity model
- ``ccl_cls.c``: code to compute angular power spectra from 3-d power spectra
- ``ccl_massfunc.c``: code for the halo mass function and halo bias models, also
  contains code to spline the linear power spectrum variance in top-hat windows
  ``sigma(R)``
- `ccl_neutrinos.c``: code to compute the neutrino masses from their cosmological
  density and vice versa.
- ``ccl_emu17.c``: code to compute the Cosmic Emu emulator for the matter power
  spectrum
- ``ccl_correlation.c``: code to compute correlation functions from 2-d and 3-d
  power spectra
- ``ccl_halomod.c``: code to compute halo model approximations to the power spctrum, also
  contains models for the halo mass-concentration relationship
- ``ccl_halofit.c``: code to compute the HALOFIT approximation for the non-linear
  matter power spectrum
- ``ccl_haloprofile.c``: code to compute common approximations to halo mass density
  profiles
- ``ccl_tracers.c``: code to compute various kernels for tracers of large-scale structure
  (e.g., weak lensing kernels, galaxy clustering kernels, etc.)
- ``ccl_fftlog.c``: an implementation of the ``FFTLog`` algorithm for fast transforms
  between Fourier and real space quantities
