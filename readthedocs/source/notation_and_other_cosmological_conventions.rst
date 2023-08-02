.. _models:

***************************************************
Notation, Models and Other Cosmological Conventions
***************************************************

The documentation here is provides a brief description of CCL and its contents.
For a more thorough description of the underlying equations CCL implements, see
the CCL note and the `CCL paper <https://arxiv.org/abs/1812.05995>`_.


Units
-----

The following conventions are used by CCL:

  - **All** units are **non**-h-inverse (e.g., Mpc as opposed to Mpc/h).
  - Distances are in units of Mpc, masses are in :math:`M_\odot` units.


Cosmological Parameters
-----------------------

CCL uses the following parameters to define the cosmological model.

Background Parameters
~~~~~~~~~~~~~~~~~~~~~

- ``Omega_c``: the density fraction at z=0 of CDM.
- ``Omega_b``: the density fraction at z=0 of baryons.
- ``h``: the Hubble constant in units of 100 :math:`{\rm km}/{\rm s}/{\rm Mpc}`.
- ``Omega_k``: the curvature density fraction at :math:`z=0`.
- ``Omega_g``: the density of radiation (not including massless neutrinos).
- ``w0``: first order term of the dark energy equation of state.
- ``wa``: second order term of the dark energy equation of state.

Power Spectrum Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The power spectrum normalization is given either as ``A_s`` (i.e., the primordial
amplitude) or as ``sigma8`` (i.e., a measure of the amplitude today). Note that
not all transfer functions support specifying a primordial amplitude.

- ``sigma8``: the normalization of the power spectrum today, given by the RMS
  variance in spheres of 8 :math:`{\rm Mpc}/h`.
- ``A_s``: the primordial normalization of the power spectrum at :math:`k_p=0.05\,{\rm Mpc}^{-1}`.

Relativistic Species
~~~~~~~~~~~~~~~~~~~~

- ``Neff``: effective number of massless+massive neutrinos present at recombination.
- ``m_nu``: the total mass of massive neutrinos or the masses of the massive neutrinos in eV.
- ``mass_split``: how to interpret the ``m_nu`` argument, see the :ref:`options <masstype>` below.
- ``T_CMB``: the temperature of the CMB today.
- ``T_ncdm``: non-CDM temperature in units of the photon temperature.

Modified Gravity Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``mu_0`` and ``sigma_0``: the parameters of the scale-independent :math:`\mu-\Sigma`
  modified gravity model.
- ``c1_mg``, ``c2_mg``, and ``lambda_mg``: parameters governing the scale dependence of
  the above.


Supported Models for Power Spectra and other options
----------------------------------------------------

``pyccl`` accepts strings indicating which model to use for various physical
quantities (e.g., the transfer function). The various options are as follows.

``transfer_function`` options

  - ``None`` : do not compute a linear power spectrum.
  - 'eisenstein_hu': the Eisenstein and Hu (1998) fitting function.
  - 'bbks': the BBKS approximation.
  - 'boltzmann_class': use CLASS to compute the transfer function.
  - 'boltzmann_camb': use CAMB to compute the transfer function (default).
  - 'boltzmann_isitgr': use ISiTGR to compute the transfer function.
  - An :class:`~pyccl.emulators.emu_base.EmulatorPk` object.

``matter_power_spectrum`` options

  - 'halofit': use HALOFIT (default).
  - 'linear': neglect non-linear power spectrum contributions.
  - An :class:`~pyccl.emulators.emu_base.EmulatorPk` object.

``baryonic_effects``: a :class:`~pyccl.baryons.baryons_base.Baryons` object.

.. _masstype:

``mass_split`` options

  This parameter specifies the model for massive
  neutrinos.

  - 'list': specify each mass yourself in eV
  - 'normal': use the normal hierarchy to convert total mass to individual
    masses (default)
  - 'inverted': use the inverted hierarchy to convert total mass to
    individual masses
  - 'equal': assume equal masses when converting the total mass to
    individual masses


The Calculator Mode
-------------------

Although ``pyccl`` aspires to support a wide variety of models, there will always
be more models out there (e.g. specific modified-gravity models, different prescriptions
for baryonic effects, etc.), which are not implemented, but which you might like to
use, together with CCL in order to calculate specific observables (e.g. weak lensing
power spectra). To enable this, CCL can create cosmologies in *"calculator mode"*.

:class:`~pyccl.cosmology.CosmologyCalculator` objects are versions of the standard
:class:`~pyccl.cosmology.Cosmology` class that can be constructed from building blocks
calculated by external libraries. The core building blocks are the distance-redshift
relation :math:`\chi(z)`, the expansion history :math:`H(z)`, the growth factor and
growth rate :math:`D(z)`, :math:`f(z)`, the linear matter power spectrum, and the
non-linear matter power spectrum. CCL can then use these building blocks to construct
observer-level predictions (angular power spectra, cluster counts,
correlation functions, etc.). Power spectra can be generated and passed to the
``CosmologyCalculator`` in the form of :class:`~pyccl.pk2d.Pk2D` objects.


Controlling Splines and Numerical Accuracy
------------------------------------------

The internal splines and integration accuracy are controlled by the global
instances ``pyccl.spline_params`` and ``pyccl.gsl_params``.
Upon instantiation, the :obj:`~pyccl.cosmology.Cosmology` object assumes the accuracy
parameters from these instances. For example, you can set the generic relative
accuracy for integration by executing
``pyccl.gsl_params["INTEGRATION_EPSREL"] = 1e-5``. To reset the accuracy
parameters to their default valus listed in ``src/ccl_core.c``, you may run
``pyccl.gsl_params.reload()`` or ``pyccl.spline_params.reload()``.

The internal splines are controlled by the following
parameters.

  - ``A_SPLINE_NLOG``: the number of logarithmically spaced bins between
    ``A_SPLINE_MINLOG`` and ``A_SPLINE_MIN``.
  - ``A_SPLINE_NA``: the number of linearly spaced bins between
    ``A_SPLINE_MIN`` and ``A_SPLINE_MAX``.
  - ``A_SPLINE_MINLOG``: the minimum value of the scale factor splines used for
    distances, etc.
  - ``A_SPLINE_MIN``: the transition scale factor between logarithmically spaced
    spline points and linearly spaced spline points.
  - ``A_SPLINE_MAX``: the the maximum value of the scale factor splines used for
    distances, etc.
  - ``LOGM_SPLINE_NM``: the number of logarithmically spaced values in mass for
    splines used in the computation of the halo mass function.
  - ``LOGM_SPLINE_MIN``: the base-10 logarithm of the minimum halo mass for
    splines used in the computation of the halo mass function.
  - ``LOGM_SPLINE_MAX``: the base-10 logarithm of the maximum halo mass for
    splines used in the computation of the halo mass function.
  - ``LOGM_SPLINE_DELTA``: the step in base-10 logarithmic units for computing
    finite difference derivatives in the computation of the mass function.
  - ``A_SPLINE_NLOG_PK``: the number of logarithmically spaced bins between
    ``A_SPLINE_MINLOG_PK`` and ``A_SPLINE_MIN_PK``.
  - ``A_SPLINE_NA_PK``: the number of linearly spaced bins between
    ``A_SPLINE_MIN_PK`` and ``A_SPLINE_MAX``.
  - ``A_SPLINE_MINLOG_PK``: the minimum value of the scale factor used
    for the power spectrum splines.
  - ``A_SPLINE_MIN_PK``: the transition scale factor between logarithmically
    spaced spline points and linearly spaced spline points for the power
    spectrum.
  - ``K_MIN``: the minimum wavenumber for the power spectrum splines for
    analytic models (e.g., BBKS, Eisenstein & Hu, etc.).
  - ``K_MAX``: the maximum wavenumber for the power spectrum splines for
    analytic models (e.g., BBKS, Eisenstein & Hu, etc.).
  - ``K_MAX_SPLINE``: the maximum wavenumber for the power spectrum splines for
    numerical models (e.g., CLASS).
  - ``N_K``: the number of spline nodes per decade for the power spectrum
    splines.
  - ``N_K_3DCOR``: the number of spline points in wavenumber per decade used for
    computing the 3D correlation function.
  - ``ELL_MIN_CORR``: the minimum value of the spline in angular wavenumber for
    correlation function computations with FFTlog.
  - ``ELL_MAX_CORR``: the maximum value of the spline in angular wavenumber for
    correlation function computations with FFTlog.
  - ``N_ELL_CORR``: the number of logarithmically spaced bins in angular
    wavenumber between ``ELL_MIN_CORR`` and ``ELL_MAX_CORR``.

The numerical accuracy of GSL computations is controlled by the following
parameters.

  - ``N_ITERATION``: the size of the GSL workspace for numerical
    integration.
  - ``INTEGRATION_GAUSS_KRONROD_POINTS``: the Gauss-Kronrod quadrature rule used
    for adaptive integrations.
  - ``INTEGRATION_EPSREL``: the relative error tolerance for numerical
    integration; used if not specified by a more specific parameter.
  - ``INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS``: the Gauss-Kronrod quadrature
    rule used for adaptive integrations on subintervals for Limber integrals.
  - ``INTEGRATION_LIMBER_EPSREL``: the relative error tolerance for numerical
    integration of Limber integrals.
  - ``INTEGRATION_DISTANCE_EPSREL``: the relative error tolerance for numerical
    integration of distance integrals.
  - ``INTEGRATION_SIGMAR_EPSREL``: the relative error tolerance for numerical
    integration of power spectrum variance intrgals for the mass function.
  - ``ROOT_EPSREL``: the relative error tolerance for root finding used to
    invert the relationship between comoving distance and scale factor.
  - ``ROOT_N_ITERATION``: the maximum number of iterations used to for root
    finding to invert the relationship between comoving distance and
    scale factor.
  - ``ODE_GROWTH_EPSREL``: the relative error tolerance for integrating the
    linear growth ODEs.
  - ``EPS_SCALEFAC_GROWTH``: 10x the starting step size for integrating the
    linear growth ODEs and the scale factor of the initial condition for the
    linear growth ODEs.
  - ``NZ_NORM_SPLINE_INTEGRATION``: Use spline integration for the normalization of
    the n(z).
  - ``LENSING_KERNEL_SPLINE_INTEGRATION``: Use spline integration for the lensing
    kernel integral.


Specifying Physical Constants
-----------------------------

The values of physical constants are set globally and are frozen. We do not
recommend changing them, as some constants derive from others (such as Newton's
gravitational constant and the solar mass). However, if you know what you are
doing, you can unfreeze with ``pyccl.physical_constants.unfreeze()`` and then
set your desired value to the parameter you would like to change.
The following constants are defined and their default values are located
in ``src/ccl_core.c``. Note that the neutrino mass splittings are taken
from `Lesgourgues & Pastor (2012) <https://arxiv.org/abs/1212.6154>`__. Also, see the
CCL note for a discussion of the values of these constants from different sources.

basic physical constants

  - ``CLIGHT_HMPC``: speed of light divided by :math:`H_0` in units of :math:`{\rm Mpc}/h`.
  - ``GNEWT``: Newton's gravitational constant in units of :math:`{\rm m}^3{\rm kg}^{-1}{\rm s}^{-2}`.
  - ``SOLAR_MASS``: solar mass in units of :math:`{\rm kg}`.
  - ``MPC_TO_METER``: conversion factor for Mpc to meters.
  - ``RHO_CRITICAL``: critical density in units of :math:`M_\odot/h/({\rm Mpc}/h)^3`.
  - ``KBOLTZ``: Boltzmann constant in units of J/K.
  - ``STBOLTZ``: Stefan-Boltzmann constant in units of :math:`{\rm kg}/{\rm s}^3 / {\rm K}^4`.
  - ``HPLANCK``: Planck's constant in units :math:`{\rm kg}\,{\rm m}^2 {\rm s}^{-1}`.
  - ``CLIGHT``: speed of light in m/s.
  - ``EV_IN_J``: conversion factor between electron volts and Joules.

neutrino mass splittings

  - ``DELTAM12_sq``: squared mass difference between eigenstates 2 and 1.
  - ``DELTAM13_sq_pos``: squared mass difference between eigenstates 3 and 1 for
    the normal hierarchy.
  - ``DELTAM13_sq_neg``: squared mass difference between eigenstates 3 and 1 for
    the inverted hierarchy.
