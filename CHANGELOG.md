# Unreleased

# v3.1.2 Changes
- Fixed dynamic versioning

# v3.1.1 Changes
- Update physical constants (#1196)
- Fixed bug in Despali 2016 mass function (#1197)

# v3.1 Changes
- Docstring improvements (#1168, #1177, #1184).
- Minor bugfixes (#1167, #1169, #1170, #1172, #1187, #1188).
- DarkEmulator mass function (#1138).

# v3.0.1 Changes
- Fixed GSL integration segfault (#1155).
- Added `to_dict` utility in cosmology class (#1160).

## Python library
- Documentation fixes (#1122).
- Background calculations (#1149).
- Bacco lbias (#1139).
- Integration of N5K winners into CCL (#1092).
- Non-Gaussian covariances (#869).
- CI fixes (#1121, #1154).

## C library
- Fix GSL integration abort (#1155)

# v3.0.0 Changes

## Python library
- Improvements to Einasto, NFW and Hernquist profiles (#1093, #1096).
- LPT non-linear bias (#1097).
- Power spectrum emulators (#1094, #1102, #1103, #1108).
- Bug fix in virial overdensity (#1100) and streamlined collapse threshold (#1101).
- sigma_8 usable as a parameter for camb when computing non-linear power spectra (#1106).
- Reverted to better value of Neff (#1111).
- New baryonic effects models (#1102, #1110).
- Restored convenience baryonic effects in Cosmology initialisation (#1113).
- Bocquet et al. 2020 mass function emulatr (#1115)
- Modified gravity parameter structures (#1119).

# v2.8.0 Changes

## Python library
- van Daalen+ baryons model (#1110)
- Refactor pyccl under an abstract CCLObject (#934).
- Bump Python version to 3.8 (#1031).
- Optimize HM autocorrelation power spectra (#891).
- `__eq__` methods for all objects (#1033).
- IA halo profile (#1074).
- Lightweight docs revamp for v2.last (#1082).

# v2.7.0 Changes

## Python library
- Numpy 1.24 compatibility (#1012).
- Fix OpenMP again (#1017).

# v2.6.0 Changes

## Python library
- Enhancements for Pk2D, Tk3D, Tracer (#923).
- Useful error for transfer functions with non-monotonic scale factors (#971).
- Better docs for correlations (#974).
- Stop supporting Python 3.6 (#975).
- HaloProfileGNFW bugfix for c500 update (#969).
- Added option to make einasto alpha a free parameter (#987, #989).
- MG parameters in CosmologyCalculator (#977).
- Added functions to check if OpenMP is working (#1000).
- Enhancement to the perturbative modelling (#1002).

# v2.5.0 Changes

## Python library
- Fixed missing terms in Super Sample Covariance (#941).
- Fixed status message string initialisation (#935).
- Fixed WL functions in pyccl.halos.profiles ( #943).
- 10x speed increase in CIB profile by (#917).
- Super Sample Covariance with linear galaxy bias approximation (#947).
- Add action to publish to pypi by (#948).

# v2.4.0 Changes

## Python library
- Fixed high-z halofit behavior (#932)
- Sped up calculation of Ishiyama21 concentration (#930)
- Improved N(z) treatment in tracers and corresponding error messages (#926, #928)
- Enabled accessing Tk3D internal arrays (#924)
- Added CIB profile using Shang et al. 2012 model (#905)
- Added third-order and non-local biases to Fast-PT wrapper (#901)

## C library
- Enabled compilation with debugging symbols (#931)

# v2.3.0 Changes

## Python library
- Fixed bug in calculation of Omega_M in cosmic emulator (#897).
- Eisenstein and Hu no-wiggle power spectrum (#898)
- Many functions now methods of the `Cosmology` class (#887)
- Correction to HM transition regime and small-scale 1-halo suppression (#877, #883)
- Generalized treatment of satellites in HOD profile (#875)
- SSC Non-Gaussian Angular power spectrum covariance (#855)
- Allow access to CAMB nonlinear power spectra (#854) and dark-energy models (#857)
- Write and read configuration settings to yaml (#852)
- Weak lensing profiles (#824)

## C library
- Deprecate C-level yaml reader and writer (#852)
- Fixed bug in emulator implementation (#899)

# v2.2.0 Changes

## Python library
- Added more developer documentation (#776).
- Vanilla LCDM cosmology (#783)
- Sunyaev-Zel'dovich GNFW profile and tracer (#784)
- Added number counts integrals for clusters (#787)
- Scale (and redshift) - dependent mu-Sigma modified gravity support (#790)
- Halo Occupation Distribution (#793)
- kwargs option in `read_yaml` (#829)
- correlation.py now correlations.py (#834)
- Calculator mode (#843)
- z-dependent mass function normalization for Tinker10 (#839)
- Non-Gaussian covariance matrices (#833, #835, #838)

# v2.1.0 Changes

## Python library
- Perturbation theory power spectra #734
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
- Updated MG support via mu / Sigma (scale-independent) with higher accuracy implementation (#737)
- Added function to estimate the scale at which non-linear structure formation becomes
  important for the computation of the matter power spectrum (#760).

## C library
- Deprecated old halo model (#736)
- Added a "spline" integration method to carry out Limber integrals.
- Added function for the angular diameter distance difference between two scale factors (#645)
- Removed old C/C++ examples (#725)
- Extended implementation of halo profiles (#722).
- Improved FFTLog API (#722).
- Functionality to resample 1D arrays including extrapolation (#720).
- Improved implementation of halo model quantities (n(M), b(M), c(M)) (#668, #655, #656, #657, #636).
- Updated MG support via mu / Sigma (scale-independent) with higher accuracy implementation (#737)

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
