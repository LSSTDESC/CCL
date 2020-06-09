.. _models:

***************************************************
Notation, Models and Other Cosmological Conventions
***************************************************

The documentation here is provides a brief description of CCL and its contents.
For a more thorough description of the underlying equations CCL implements, see
the `CCL note <https://github.com/LSSTDESC/CCL/blob/master/doc/0000-ccl_note/0000-ccl_note.pdf>`_
and the `CCL paper <https://arxiv.org/abs/1812.05995>`_.

Cosmological Parameters
-----------------------

CCL uses the following parameters to define the cosmological model.

Background Parameters
~~~~~~~~~~~~~~~~~~~~~

- ``Omega_c``: the density fraction at z=0 of CDM
- ``Omega_b``: the density fraction at z=0 of baryons
- ``h``: the Hubble constant in units of 100 Mpc/km/s
- ``Omega_k``: the curvature density fraction at z=0
- ``Omega_g``: the density of radiation (not including massless neutrinos)
- ``w0``: first order term of the dark energy equation of state
- ``wa``: second order term of the dark energy equation of state

Power Spectrum Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The power spectrum normalization is given either as ``A_s`` (i.e., the primordial
amplitude) or as ``sigma8`` (i.e., a measure of the amplitude today). Note that
not all transfer functions support specifying a primordial amplitude.

- ``sigma8``: the normalization of the power spectrum today, given by the RMS
  variance in spheres of 8 Mpc/h
- ``A_s``: the primordial normalization of the power spectrum at k=0.05 Mpc :math:`^{-1}`

Relativistic Species
~~~~~~~~~~~~~~~~~~~~

- ``Neff``: effective number of massless+massive neutrinos present at recombination
- ``m_nu``: the total mass of massive neutrinos or the masses of the massive neutrinos in eV
- ``m_nu_type``: how to interpret the ``m_nu`` argument, see the :ref:`options <mnutype>` below
- ``T_CMB``: the temperature of the CMB today

Modified Gravity Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``mu_0`` and ``sigma_0``: the parameters of the scale-independent :math:`\mu-\Sigma`
  modified gravity model
- ``df_mg`` and ``z_mg``: arrays of perturbations to the GR growth rate ``f`` as
  a function of redshift


Supported Models for the Power Spectrum, Mass Function, etc.
------------------------------------------------------------

``pyccl`` accepts strings indicating which model to use for various physical
quantities (e.g., the transfer function). The various options are as follows.

``transfer_function`` options

  - ``None`` : do not compute a linear power spectrum
  - 'eisenstein_hu': the Eisenstein and Hu (1998) fitting function
  - 'bbks': the BBKS approximation
  - 'boltzmann_class': use CLASS to compute the transfer function
  - 'boltzmann_camb': use CAMB to compute the transfer function (default)

``matter_power_spectrum`` options

  - 'halo_model': use a halo model
  - 'halofit': use HALOFIT (default)
  - 'linear': neglect non-linear power spectrum contributions
  - 'emu': use the Cosmic Emu

``baryons_power_spectrum`` options

  - 'nobaryons': neglect baryonic contributions to the power spectrum (default)
  - 'bcm': use the baryonic correction model

``mass_function`` options

  - 'tinker': the Tinker et al. (2008) mass function
  - 'tinker10': the Tinker et al. (2010) mass function (default)
  - 'watson': the Watson et al. mass function
  - 'angulo': the Angulo et al. mass function
  - 'shethtormen': the Sheth and Tormen mass function

``halo_concentration`` options

  - 'bhattacharya2011': Bhattacharya et al. (2011) relation
  - 'duffy2008': Duffy et al. (2008) relation (default)
  - 'constant_concentration': use a constant concentration

.. _mnutype:

``m_nu_type`` options

  This parameter specifies the model for massive
  neutrinos.

  - 'list': specify each mass yourself in eV
  - 'normal': use the normal hierarchy to convert total mass to individual
    masses (default)
  - 'inverted': use the inverted hierarchy to convert total mass to
    individual masses
  - 'equal': assume equal masses when converting the total mass to
    individual masses

``emulator_neutrinos`` options

  This parameter specifies how to handle inconsistencies in the treatment of
  neutrinos between the Cosmic Emu (equal masses) and other models.

  - 'strict': fail unless things are absolutely consistent (default)
  - 'equalize': redistribute the total mass equaly before using the Cosmic
    Emu. This option may result in slight internal inconsistencies in the
    physical model assumed for neutrinos.


Controlling Splines and Numerical Accuracy
------------------------------------------

The internal splines and integration accuracy are controlled by the
attributes of :obj:`Cosmology.cosmo.spline_params` and
:obj:`Cosmology.cosmo.gsl_params`. These should be set after instantiation,
but before the object is used. For example, you can set the generic relative
accuracy for integration by executing
``c = Cosmology(...); c.cosmo.gsl_params.INTEGRATION_EPSREL = 1e-5``. The
default values for these parameters are located in ``src/ccl_core.c``.

The internal splines are controlled by the following
parameters.

  - A_SPLINE_NLOG: the number of logarithmically spaced bins between
    A_SPLINE_MINLOG and A_SPLINE_MIN.
  - A_SPLINE_NA: the number of linearly spaced bins between
    A_SPLINE_MIN and A_SPLINE_MAX.
  - A_SPLINE_MINLOG: the minimum value of the scale factor splines used for
    distances, etc.
  - A_SPLINE_MIN: the transition scale factor between logarithmically spaced
    spline points and linearly spaced spline points.
  - A_SPLINE_MAX: the the maximum value of the scale factor splines used for
    distances, etc.
  - LOGM_SPLINE_NM: the number of logarithmically spaced values in mass for
    splines used in the computation of the halo mass function.
  - LOGM_SPLINE_MIN: the base-10 logarithm of the minimum halo mass for
    splines used in the computation of the halo mass function.
  - LOGM_SPLINE_MAX: the base-10 logarithm of the maximum halo mass for
    splines used in the computation of the halo mass function.
  - LOGM_SPLINE_DELTA: the step in base-10 logarithmic units for computing
    finite difference derivatives in the computation of the mass function.
  - A_SPLINE_NLOG_PK: the number of logarithmically spaced bins between
    A_SPLINE_MINLOG_PK and A_SPLINE_MIN_PK.
  - A_SPLINE_NA_PK: the number of linearly spaced bins between
    A_SPLINE_MIN_PK and A_SPLINE_MAX.
  - A_SPLINE_MINLOG_PK: the minimum value of the scale factor used
    for the power spectrum splines.
  - A_SPLINE_MIN_PK: the transition scale factor between logarithmically
    spaced spline points and linearly spaced spline points for the power
    spectrum.
  - K_MIN: the minimum wavenumber for the power spectrum splines for
    analytic models (e.g., BBKS, Eisenstein & Hu, etc.).
  - K_MAX: the maximum wavenumber for the power spectrum splines for
    analytic models (e.g., BBKS, Eisenstein & Hu, etc.).
  - K_MAX_SPLINE: the maximum wavenumber for the power spectrum splines for
    numerical models (e.g., ComsicEmu, CLASS, etc.).
  - N_K: the number of spline nodes per decade for the power spectrum
    splines.
  - N_K_3DCOR: the number of spline points in wavenumber per decade used for
    computing the 3D correlation function.
  - ELL_MIN_CORR: the minimum value of the spline in angular wavenumber for
    correlation function computations with FFTlog.
  - ELL_MAX_CORR: the maximum value of the spline in angular wavenumber for
    correlation function computations with FFTlog.
  - N_ELL_CORR: the number of logarithmically spaced bins in angular
    wavenumber between ELL_MIN_CORR and ELL_MAX_CORR.

The numerical accuracy of GSL computations is controlled by the following
parameters.

  - N_ITERATION: the size of the GSL workspace for numerical
    integration.
  - INTEGRATION_GAUSS_KRONROD_POINTS: the Gauss-Kronrod quadrature rule used
    for adaptive integrations.
  - INTEGRATION_EPSREL: the relative error tolerance for numerical
    integration; used if not specified by a more specific parameter.
  - INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS: the Gauss-Kronrod quadrature
    rule used for adaptive integrations on subintervals for Limber integrals.
  - INTEGRATION_LIMBER_EPSREL: the relative error tolerance for numerical
    integration of Limber integrals.
  - INTEGRATION_DISTANCE_EPSREL: the relative error tolerance for numerical
    integration of distance integrals.
  - INTEGRATION_SIGMAR_EPSREL: the relative error tolerance for numerical
    integration of power spectrum variance intrgals for the mass function.
  - ROOT_EPSREL: the relative error tolerance for root finding used to
    invert the relationship between comoving distance and scale factor.
  - ROOT_N_ITERATION: the maximum number of iterations used to for root
    finding to invert the relationship between comoving distance and
    scale factor.
  - ODE_GROWTH_EPSREL: the relative error tolerance for integrating the
    linear growth ODEs.
  - EPS_SCALEFAC_GROWTH: 10x the starting step size for integrating the
    linear growth ODEs and the scale factor of the initial condition for the
    linear growth ODEs.
  - HM_MMIN: the minimum mass for halo model integrations.
  - HM_MMAX: the maximum mass for halo model integrations.
  - HM_EPSABS: the absolute error tolerance for halo model integrations.
  - HM_EPSREL: the relative error tolerance for halo model integrations.
  - HM_LIMIT: the size of the GSL workspace for halo moodel integrations.
  - HM_INT_METHOD: the Gauss-Kronrod quadrature rule used for adaptive
    integrations for the halo model comptutations.


Specifying Physical Constants
-----------------------------

The values of physical constants are set globally. These can be changed by
assigning a new value to the attributes of ``pyccl.physical_constants``.
The following constants are defined and their default values are located
in ``src/ccl_core.c``. Note that the neutrino mass splittings are taken
from Lesgourgues & Pastor (2012; 1212.6154). Also, see the
`CCL note <https://github.com/LSSTDESC/CCL/blob/master/doc/0000-ccl_note/0000-ccl_note.pdf>`_
for a discussion of the values of these constants from different sources.

basic physical constants

  - CLIGHT_HMPC: speed of light / H0 in units of Mpc/h
  - GNEWT: Newton's gravitational constant in units of m^3/Kg/s^2
  - SOLAR_MASS: solar mass in units of kg
  - MPC_TO_METER: conversion factor for Mpc to meters.
  - PC_TO_METER: conversion factor for parsecs to meters.
  - RHO_CRITICAL: critical density in units of M_sun/h / (Mpc/h)^3
  - KBOLTZ: Boltzmann constant in units of J/K
  - STBOLTZ: Stefan-Boltzmann constant in units of kg/s^3 / K^4
  - HPLANCK: Planck's constant in units kg m^2 / s
  - CLIGHT: speed of light in m/s
  - EV_IN_J: conversion factor between electron volts and Joules
  - T_CMB: temperature of the CMB in K
  - TNCDM: temperature of the cosmological neutrino background in K

neutrino mass splittings

  - DELTAM12_sq: squared mass difference between eigenstates 2 and 1.
  - DELTAM13_sq_pos: squared mass difference between eigenstates 3 and 1 for
    the normal hierarchy.
  - DELTAM13_sq_neg: squared mass difference between eigenstates 3 and 1 for
    the inverted hierarchy.
