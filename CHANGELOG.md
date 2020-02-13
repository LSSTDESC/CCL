# Unreleased
## Python library
- Generalized Tinker mass function to all SO mass definitions (#736)
- Deprecated old halo model (#736)
- Halo model power spectra #717
- Fixed bug in Duffy 2008 concentration-mass relation #717
- Added a "spline" integration method to carry out Limber integrals.
- Added a function for the angular diameter distance difference between two scale factors (#645)
- Extended implementation of halo profiles (#722).
- Improved FFTLog API (#722).
- Functionality to resample 1D arrays including extrapolation (#720).
- Methods to retrieve internal tracer functions (#713).
- Improved implementation of halo model quantities (n(M), b(M), c(M)) (#668, #655, #656, #657, #636).
- Add straightforward single massive neutrino option (#730).

## C library
- Deprecated old halo model (#736)
- Added a "spline" integration method to carry out Limber integrals.
- Added function for the angular diameter distance difference between two scale factors (#645)
- Removed old C/C++ examples (#725)
- Extended implementation of halo profiles (#722).
- Improved FFTLog API (#722).
- Functionality to resample 1D arrays including extrapolation (#720).
- Improved implementation of halo model quantities (n(M), b(M), c(M)) (#668, #655, #656, #657, #636).

# v2.0.1 Changes

## Bug Fixes
- Fixed a PyPi installation bug where the proper files for `CMAKE` were not
  included in the source distribution.

# v2.0 API changes

## Python library
- Made CAMB the default Boltzmann code (#685).
- Added check to ensure the number of relativistic neutrinos is positive (#684).
- Added massive neutrinos to Omega_m (#680).
- Changed neutrino API options to be intuitive and consistent (#681).
- Enabled CAMB (#677)
- Fixed bug when normalizing the linear power spectrum using sigma8 and
  the mu-Sigma modified gravity model (#677).
- Enabled the mu-Sigma modified gravity model for any linear power spectrum (#677).
- Refactored the power spectrum normalization routines to only run CAMB/CLASS once
  when using sigma8 (#677).
- Fixed a bug where the power spectrum normalization was not being set
  accurately when using sigma8 (#677).
- Added warnings for inconsistent models (#676).
- Moved CLASS interface to python (#652).
- Moved all benchmarks and tests to python (#653).
- Changed IA bias normalization to be consistent with A_IA=1 (#630).
- Implemented generalized models for tracers (#630)
- Improved error reporting for `angular_cl` computations (#567).
- Deprecated the `pyccl.redshifts` module (#579).
- Remove global splines for RSD correlation functions. These are now stored
  per cosmology. Further, they are now rebuilt on-the-fly for a given
  cosmology if a new scale factor is requested. (#582)
- Allow spline, numerical and constant parameters to be set from Python (#557).
- Deprecated transfer function options 'ccl_emulator', 'ccl_fitting_function'
  'ccl_boltzmann', 'ccl_boltzmann_class' and 'ccl_boltzmann_camb' (#610). These
  were either not implemented or aliases for another option.
- Renamed transfer function option 'ccl_none' to 'ccl_transfer_none' to avoid
  ambiguity (#610).
- Refactored transfer function and matter power spectrum options to allow
  any combination, even unphysical ones (#610).
- Added capability to use the halo model power spectrum as the primary
  non-linear power spectrum in the code (#610).
- Fixed infinite loop bug in splitting sum of neutrino masses into individual masses (#605).
- Added custom Halofit code (#611).
- Added `has_density` and `has_shear` tags to `Tracer` constructors.
- Changed TCMB to T_CMB everywhere (#615)
- Deprecate Angpow (#571)
- Added support for modified gravity via mu / Sigma (scale-independent) parameterisation (#442)

## C library
- Added massive neutrinos to Omega_m (#680).
- Moved CLASS interface to python (#652).
- Added OpenMP (#651).
- Removed all benchmarks from C and kept only the C unit tests in C (#653).
- Implemented generalized models for tracers (#630)
- Fixed memory leak in CLASS power spectrum computations (#561, #562).
- Fixed a bug where CLASS would crash due to small rounding errors at z = 0
  when evaluating power spectra (#563, #564).
- Fixed bug in fftlog for some complex arguments (#565, #566).
- Replaced custom gamma function with that from GSL (#570).
- Deprecated the `ccl_redshifts.h` functions (#579).
- Refactored spline and numerical parameters to be allocated per cosmology (#557).
- Allow global physical constants to be changed (#557).
- Fixed memory leaks in `ccl_correlation.c` (#581).
- Deprecated transfer function options 'ccl_emulator', 'ccl_fitting_function'
  'ccl_boltzmann', 'ccl_boltzmann_class' and 'ccl_boltzmann_camb' (#610). These
  were either not implemented or aliases for another option.
- Renamed transfer function option 'ccl_none' to 'ccl_transfer_none' to avoid
  ambiguity (#610).
- Refactored transfer function and matter power spectrum options to allow
  any combination, even unphysical ones (#610).
- Added additional header and source files for clarity (#610).
- Added capability to use the halo model power spectrum as the primary
  non-linear power spectrum in the code (#610).
- Fixed infinite loop bug in splitting sum of neutrino masses into individual masses (#605).
- Added custom Halofit code (#611).
- Separated Limber and Non-Limber C_ell calculations (#614)
- Added `has_density` and `has_shear` flags to ClTracers (#614)
- Simplified C_ell unit tests (#614)
- Changed TCMB to T_CMB everywhere (#615)
- Fixed a small bug in the w_tophat expression and increased precision (#607)
- Deprecated the use of GSL spline accelerators (#626)
- Deprecate Angpow (#571)
- Added support for modified gravity via mu / Sigma (scale-independent) parameterisation (#442)

# v 1.0 API changes :

## C library
- Deprecated the `native` non-Limber angular power spectrum method (#506).
- Renamed `ccl_lsst_specs.c` to `ccl_redshifts.c`, deprecated LSST-specific redshift distribution functionality, introduced user-defined true dNdz (changes in call signature of `ccl_dNdz_tomog`). (#528).

## Python library
- Renamed `lsst_specs.py` to `redshifts.py`, deprecated LSST-specific redshift distribution functionality, introduced user-defined true dNdz (changes in call signature of `dNdz_tomog`). (#528).
- Deprecated the `native` non-Limber angular power spectrum method (#506).
- Deprecated the `Parameters` object in favor of only the `Cosmology` object (#493).
- Renamed the `ClTracer` family of objects (#496).
- Various function parameter name changes and documentation improvements (#464).

# v 0.4 API changes:

Summary: added halo model matter power spectrum calculation and halo mass-concentration relations. Change to sigma(R) function so that it now has time dependence: it is now sigma(R,a). Added a sigmaV(R,a) function, where sigmaV(R,a) is the variance in the displacement field smoothed on scale R at scale-factor a.

## C library
In ccl_halomod.c:

Added this source file. Contains functions to compute halo-model matter power spectra and also mass-concentration relations.

In ccl_power.c

Added sigmaV, changed sigma(R) -> sigma(R,a)

In ccl_massfunc.c

Added Sheth & Tormen (1999) mass function.

## Python library

sigmaR(R) defaults to sigmaR(R,a=1) unless a is specified. sigmaV(R) is also added. New functions ccl.halomodel_matter_power and ccl.halo_concentration.

# v 0.3 API changes:

Summary: the user interface for setting up cosmologies with neutrinos has been altered. Users should from now on pass Neff, the effective number of relativistic neutrino species in the early universe, and mnu, either a sum or neutrino masses or a set of 3 neutrinos masses.

## C library
In ccl_core.c:

In the function, 'ccl\_parameters\_create', the arguements 'double N\_nu\_rel', and 'double N\_nu\_mass' have been removed. The arguments 'double Neff' and 'ccl\_mnu\_convention mnu\_type' have been added. The argument 'mnu' has changed in type from 'double mnu' to 'double* mnu'.

Similar changes apply in 'ccl\_cosmology\_create\_with\_params' and all 'ccl\_parameters\_create...nu' convenience functions.

Additionally, in the function 'ccl\_parameters\_create' and 'ccl\_cosmology\_create\_with\_params', arguments have been added for the parameters of the BCM baryon model; these are 'double bcm\_log10Mc', 'double bcm\_etab', and 'double bcm\_ks'.

In ccl_neutrinos.c:

The function ccl\_Omeganuh2\_to\_Mnu has been renamed ccl\_nu\_masses. The arguments 'double a' and 'gsl\_interp\_accel* accel' have been removed. The argument 'ccl\_neutrino\_mass\_splits mass\_split' has been added.

## Python wrapper
In core.py:

In the Parameters class, the arguments 'N\_nu\_rel', and 'N\_nu\_mass' have been removed. The optional arguments 'Neff', 'mnu\_type', 'bcm\_log10Mc', 'bcm\_etab', and 'bcm\_ks' have been added. Similar changes occur in the Cosmology class.

In neutrinos.py:

In the function Omeganuh2, the argument 'Neff' has been removed. It is now fixed to the length of the argument 'mnu'.
The function 'Omeganuh2\_to\_Mnu' has been renamed 'nu\_masses'. The arguments 'a' and 'Neff' have been removed. The argument 'mass\_split' has been added.
The argument 'TCMB' has been changed to 'T_CMB'.

## Other changes since release 0.2.1 (September 2017):

CLASS is no longer included as part of CCL; it can instead of easily downloaded via the class_install.py script and this procedure is documented.

Tutorial added for usage with MCMC

Added support for BCM baryon model

cpp compatibility improved

Python 3 support added

Added support for computing the nonlinear matter power spectrum with CosmicEmu

Added support for CMB lensing observables, including splines for cosmological quantities to higher redshift

Added the ability to return useful functions such as dNdz for a tracer Cl object.

Clarified license

Values of the physical constants have changed to CODATA2014/IAU2015
