"""The core functionality of ccl, including the core data types. This includes
the cosmology and parameters objects used to instantiate a model from which one
can compute a set of theoretical predictions.

------------------------------------------------------------
Supported Models for the Power Spectrum, Mass Function, etc.
------------------------------------------------------------

The classes in this module accept strings indicating which model to use
for various physical quantities (e.g., the transfer function). The various
options are as follows.

transfer_function options
  - None : do not compute a linear power spectrum
  - 'eisenstein_hu': the Eisenstein and Hu (1998) fitting function
  - 'bbks': the BBKS approximation
  - 'boltzmann_class': use CLASS to compute the transfer function
  - 'boltzmann_camb': use CAMB to compute the transfer function

matter_power_spectrum options
  - 'halo_model': use a halo model
  - 'halofit': use HALOFIT
  - 'linear': neglect non-linear power spectrum contributions
  - 'emu': use the Cosmic Emu

baryons_power_spectrum options
  - 'nobaryons': neglect baryonic contributions to the power spectrum
  - 'bcm': use the baryonic correction model

mass_function options
  - 'tinker': the Tinker et al. (2008) mass function
  - 'tinker10': the Tinker et al. (2010) mass function
  - 'watson': the Watson et al. mass function
  - 'angulo': the Angulo et al. mass function
  - 'shethtormen': the Sheth and Tormen mass function

halo_concentration options
  - 'bhattacharya2011': Bhattacharya et al. (2011) relation
  - 'duffy2008': Duffy et al. (2008) relation
  - 'constant_concentration': use a constant concentration

mnu_type options
  This parameter specifies the model for massive
  neutrinos.
    - 'list': specify each mass yourself in eV
    - 'sum': use the normal hierarchy to convert total mass to individual
      masses
    - 'sum_inverted': use the inverted hierarchy to convert total mass to
      individual masses
    - 'sum_equal': assume equal masses when converting the total mass to
      individual masses

emulator_neutrinos options
  This parameter specifies how to handle inconsistencies in the treatment of
  neutrinos between the Cosmic Emu (equal masses) and other models.
    - 'strict': fail unless things are absolutely consistent
    - 'equalize': redistribute the total mass equaly before using the Cosmic
      Emu. This option may result in slight internal inconsistencies in the
      physical model assumed for neutrinos.

------------------------------------------
Controlling Splines and Numerical Accuracy
------------------------------------------

The internal splines and integration accuracy are controlled by the
attributes of :obj:`Cosmology.cosmo.spline_params` and
:obj:`Cosmology.cosmo.gsl_params`. These should be set after instantiation,
but before the object is used. For example, you can set the generic relative
accuracy for integration by executing
``c = Cosmology(...); c.cosmo.gsl_params.INTEGRATION_EPSREL = 1e-5``. The
default values for these parameters are located in ``src/ccl_core.c``.

The intrnal splines are controlled by the following
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

The numrical accuracy of GSL computations are controlled by the following
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

-----------------------------
Specifying Physical Constants
-----------------------------

The values of physical constants are set globally. These can be changed by
assigning a new value to the attributes of ``pyccl.physical_constants``.
The following constants are defined and their default values are located
in ``src/ccl_core.c``. Note that the neutrino mass splittings are taken
from Lesgourgues & Pastor (2012; 1212.6154).

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
"""
import warnings
import numpy as np
import yaml

from . import ccllib as lib
from .errors import CCLError, CCLWarning
from ._types import error_types
from .boltzmann import get_class_pk_lin, get_camb_pk_lin
from .pyutils import check

# Configuration types
transfer_function_types = {
    None:               lib.transfer_none,
    'eisenstein_hu':    lib.eisenstein_hu,
    'bbks':             lib.bbks,
    'boltzmann_class':  lib.boltzmann_class,
    'boltzmann_camb':   lib.boltzmann_camb,
}

matter_power_spectrum_types = {
    'halo_model':   lib.halo_model,
    'halofit':      lib.halofit,
    'linear':       lib.linear,
    'emu':          lib.emu
}

baryons_power_spectrum_types = {
    'nobaryons':   lib.nobaryons,
    'bcm':         lib.bcm
}

# List which transfer functions can be used with the muSigma_MG
# parameterisation of modified gravity
valid_muSig_transfers = {'boltzmann_class', 'class'}

mass_function_types = {
    'angulo':      lib.angulo,
    'tinker':      lib.tinker,
    'tinker10':    lib.tinker10,
    'watson':      lib.watson,
    'shethtormen': lib.shethtormen
}

halo_concentration_types = {
    'bhattacharya2011':          lib.bhattacharya2011,
    'duffy2008':                 lib.duffy2008,
    'constant_concentration':    lib.constant_concentration,
}

emulator_neutrinos_types = {
    'strict': lib.emu_strict,
    'equalize': lib.emu_equalize
}

mnu_types = {
    'list': lib.mnu_list,
    'sum': lib.mnu_sum,
    'sum_inverted': lib.mnu_sum_inverted,
    'sum_equal': lib.mnu_sum_equal,
}


class Cosmology(object):
    """A cosmology including parameters and associated data.

    .. note:: Although some arguments default to `None`, they will raise a
              ValueError inside this function if not specified, so they are not
              optional.

    .. note:: The parameter Omega_g can be used to set the radiation density
              (not including relativistic neutrinos) to zero. Doing this will
              give you a model that is physically inconsistent since the
              temperature of the CMB will still be non-zero. Note however
              that this approximation is common for late-time LSS computations.

    .. note:: BCM stands for the "baryonic correction model" of Schneider &
              Teyssier (2015; https://arxiv.org/abs/1510.06034). See the
              `DESC Note <https://github.com/LSSTDESC/CCL/blob/master/doc\
/0000-ccl_note/main.pdf>`_
              for details.

    .. note:: After instantiation, you can set parameters related to the
              internal splines and numerical integration accuracy by setting
              the values of the attributes of
              :obj:`Cosmology.cosmo.spline_params` and
              :obj:`Cosmology.cosmo.gsl_params`. For example, you can set
              the generic relative accuracy for integration by executing
              ``c = Cosmology(...); c.cosmo.gsl_params.INTEGRATION_EPSREL \
= 1e-5``.
              See the module level documetaion of `pyccl.core` for details.

    Args:
        Omega_c (:obj:`float`): Cold dark matter density fraction.
        Omega_b (:obj:`float`): Baryonic matter density fraction.
        h (:obj:`float`): Hubble constant divided by 100 km/s/Mpc; unitless.
        A_s (:obj:`float`): Power spectrum normalization. Exactly one of A_s
            and sigma_8 is required.
        sigma8 (:obj:`float`): Variance of matter density perturbations at
            an 8 Mpc/h scale. Exactly one of A_s and sigma_8 is required.
        n_s (:obj:`float`): Primordial scalar perturbation spectral index.
        Omega_k (:obj:`float`, optional): Curvature density fraction.
            Defaults to 0.
        Omega_g (:obj:`float`, optional): Density in relativistic species
            except massless neutrinos. The default of `None` corresponds
            to setting this from the CMB temperature. Note that if a non-`None`
            value is given, this may result in a physically inconsistent model
            because the CMB temperature will still be non-zero in the
            parameters.
        Neff (:obj:`float`, optional): Effective number of massless
            neutrinos present. Defaults to 3.046.
        m_nu (:obj:`float`, optional): Total mass in eV of the massive
            neutrinos present. Defaults to 0.
        mnu_type (:obj:`str`, optional): The type of massive neutrinos.
        w0 (:obj:`float`, optional): First order term of dark energy equation
            of state. Defaults to -1.
        wa (:obj:`float`, optional): Second order term of dark energy equation
            of state. Defaults to 0.
        T_CMB (:obj:`float`): The CMB temperature today. The default of
            ``None`` uses the global CCL value in
            ``pyccl.physical_constants.T_CMB``.
        bcm_log10Mc (:obj:`float`, optional): One of the parameters of the
            BCM model. Defaults to `np.log10(1.2e14)`.
        bcm_etab (:obj:`float`, optional): One of the parameters of the BCM
            model. Defaults to 0.5.
        bcm_ks (:obj:`float`, optional): One of the parameters of the BCM
            model. Defaults to 55.0.
        mu_0 (:obj:`float`, optional): One of the parameters of the mu-Sigma
            modified gravity model. Defaults to 0.0
        sigma_0 (:obj:`float`, optional): One of the parameters of the mu-Sigma
            modified gravity model. Defaults to 0.0
        df_mg (array_like, optional): Perturbations to the GR growth rate as
            a function of redshift :math:`\\Delta f`. Used to implement simple
            modified growth scenarios.
        z_mg (array_like, optional): Array of redshifts corresponding to df_mg.
        transfer_function (:obj:`str`, optional): The transfer function to
            use. Defaults to 'boltzmann_class'.
        matter_power_spectrum (:obj:`str`, optional): The matter power
            spectrum to use. Defaults to 'halofit'.
        baryons_power_spectrum (:obj:`str`, optional): The correction from
            baryonic effects to be implemented. Defaults to 'nobaryons'.
        mass_function (:obj:`str`, optional): The mass function to use.
            Defaults to 'tinker10' (2010).
        halo_concentration (:obj:`str`, optional): The halo concentration
            relation to use. Defaults to Duffy et al. (2008) 'duffy2008'.
        emulator_neutrinos: `str`, optional): If using the emulator for
            the power spectrum, specified treatment of unequal neutrinos.
            Options are 'strict', which will raise an error and quit if the
            user fails to pass either a set of three equal masses or a sum with
            mnu_type = 'sum_equal', and 'equalize', which will redistribute
            masses to be equal right before calling the emualtor but results in
            internal inconsistencies. Defaults to 'strict'.
    """
    def __init__(
            self, Omega_c=None, Omega_b=None, h=None, n_s=None,
            sigma8=None, A_s=None,
            Omega_k=0., Omega_g=None, Neff=3.046, m_nu=0., mnu_type=None,
            w0=-1., wa=0., T_CMB=None,
            bcm_log10Mc=np.log10(1.2e14), bcm_etab=0.5,
            bcm_ks=55., mu_0=0., sigma_0=0., z_mg=None, df_mg=None,
            transfer_function='boltzmann_class',
            matter_power_spectrum='halofit',
            baryons_power_spectrum='nobaryons',
            mass_function='tinker10',
            halo_concentration='duffy2008',
            emulator_neutrinos='strict'):

        # going to save these for later
        self._params_init_kwargs = dict(
            Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=n_s, sigma8=sigma8,
            A_s=A_s, Omega_k=Omega_k, Omega_g=Omega_g, Neff=Neff, m_nu=m_nu,
            mnu_type=mnu_type, w0=w0, wa=wa, T_CMB=T_CMB,
            bcm_log10Mc=bcm_log10Mc,
            bcm_etab=bcm_etab, bcm_ks=bcm_ks, mu_0=mu_0, sigma_0=sigma_0,
            z_mg=z_mg, df_mg=df_mg)

        self._config_init_kwargs = dict(
            transfer_function=transfer_function,
            matter_power_spectrum=matter_power_spectrum,
            baryons_power_spectrum=baryons_power_spectrum,
            mass_function=mass_function,
            halo_concentration=halo_concentration,
            emulator_neutrinos=emulator_neutrinos)

        self._build_cosmo()

    def _build_cosmo(self):
        """Assemble all of the input data into a valid ccl_cosmology object."""
        # We have to make all of the C stuff that goes into a cosmology
        # and then we make the cosmology.
        self._build_parameters(**self._params_init_kwargs)
        self._build_config(**self._config_init_kwargs)
        self.cosmo = lib.cosmology_create(self._params, self._config)

        if self.cosmo.status != 0:
            raise CCLError(
                "(%d): %s"
                % (self.cosmo.status, self.cosmo.status_message))

    def write_yaml(self, filename):
        """Write a YAML representation of the parameters to file.

        Args:
            filename (:obj:`str`) Filename to write parameters to.
        """
        # NOTE: we use the C yaml dump here so that the parameters
        # dumped by this object are compatible with the C yaml load function.
        status = 0
        lib.parameters_write_yaml(self._params, filename, status)

        # Check status
        if status != 0:
            raise IOError("Unable to write YAML file {}".format(filename))

    @classmethod
    def read_yaml(cls, filename):
        """Read the parameters from a YAML file.
        Args:
            filename (:obj:`str`) Filename to read parameters from.
        """
        with open(filename, 'r') as fp:
            params = yaml.load(fp, Loader=yaml.Loader)

        # Now we assemble an init for the object since the CCL YAML has
        # extra info we don't need and different formatting.
        inits = dict(
            Omega_c=params['Omega_c'],
            Omega_b=params['Omega_b'],
            h=params['h'],
            n_s=params['n_s'],
            sigma8=None if params['sigma8'] == 'nan' else params['sigma8'],
            A_s=None if params['A_s'] == 'nan' else params['A_s'],
            Omega_k=params['Omega_k'],
            Neff=params['Neff'],
            w0=params['w0'],
            wa=params['wa'],
            bcm_log10Mc=params['bcm_log10Mc'],
            bcm_etab=params['bcm_etab'],
            bcm_ks=params['bcm_ks'],
            mu_0=params['mu_0'],
            sigma_0=params['sigma_0'])
        if 'z_mg' in params:
            inits['z_mg'] = params['z_mg']
            inits['df_mg'] = params['df_mg']

        if 'mnu' in params:
            inits['m_nu'] = params['mnu']
            inits['mnu_type'] = 'list'

        return cls(**inits)

    def _build_config(
            self, transfer_function=None, matter_power_spectrum=None,
            baryons_power_spectrum=None,
            mass_function=None, halo_concentration=None,
            emulator_neutrinos=None):
        """Build a ccl_configuration struct.

        This function builds C ccl_configuration struct. This structure
        controls which various approximations are used for the transfer
        function, matter power spectrum, baryonic effect in the matter
        power spectrum, mass function, halo concentration relation, and
        neutrino effects in the emulator.

        It also does some error checking on the inputs to make sure they
        are valid and physically consistent.
        """

        # Check validity of configuration-related arguments
        if transfer_function not in transfer_function_types.keys():
            raise ValueError(
                "'%s' is not a valid transfer_function type. "
                "Available options are: %s"
                % (transfer_function,
                   transfer_function_types.keys()))
        if matter_power_spectrum not in matter_power_spectrum_types.keys():
            raise ValueError(
                "'%s' is not a valid matter_power_spectrum "
                "type. Available options are: %s"
                % (matter_power_spectrum,
                   matter_power_spectrum_types.keys()))
        if (baryons_power_spectrum not in
                baryons_power_spectrum_types.keys()):
            raise ValueError(
                "'%s' is not a valid baryons_power_spectrum "
                "type. Available options are: %s"
                % (baryons_power_spectrum,
                   baryons_power_spectrum_types.keys()))
        if mass_function not in mass_function_types.keys():
            raise ValueError(
                "'%s' is not a valid mass_function type. "
                "Available options are: %s"
                % (mass_function,
                   mass_function_types.keys()))
        if halo_concentration not in halo_concentration_types.keys():
            raise ValueError(
                "'%s' is not a valid halo_concentration type. "
                "Available options are: %s"
                % (halo_concentration,
                   halo_concentration_types.keys()))
        if emulator_neutrinos not in emulator_neutrinos_types.keys():
            raise ValueError("'%s' is not a valid emulator neutrinos "
                             "method. Available options are: %s"
                             % (emulator_neutrinos,
                                emulator_neutrinos_types.keys()))

        # Assign values to new ccl_configuration object
        config = lib.configuration()

        config.transfer_function_method = \
            transfer_function_types[transfer_function]
        config.matter_power_spectrum_method = \
            matter_power_spectrum_types[matter_power_spectrum]
        config.baryons_power_spectrum_method = \
            baryons_power_spectrum_types[baryons_power_spectrum]
        config.mass_function_method = \
            mass_function_types[mass_function]
        config.halo_concentration_method = \
            halo_concentration_types[halo_concentration]
        config.emulator_neutrinos_method = \
            emulator_neutrinos_types[emulator_neutrinos]

        # Store ccl_configuration for later access
        self._config = config

    def _build_parameters(
            self, Omega_c=None, Omega_b=None, h=None, n_s=None, sigma8=None,
            A_s=None, Omega_k=None, Neff=None, m_nu=None, mnu_type=None,
            w0=None, wa=None, T_CMB=None,
            bcm_log10Mc=None, bcm_etab=None, bcm_ks=None,
            mu_0=None, sigma_0=None, z_mg=None, df_mg=None, Omega_g=None):
        """Build a ccl_parameters struct"""

        # Set nz_mg (no. of redshift bins for modified growth fns.)
        if z_mg is not None and df_mg is not None:
            # Get growth array size and do sanity check
            z_mg = np.atleast_1d(z_mg)
            df_mg = np.atleast_1d(df_mg)
            if z_mg.size != df_mg.size:
                raise ValueError(
                    "The parameters `z_mg` and `dF_mg` are "
                    "not the same shape!")
            nz_mg = z_mg.size
        else:
            # If one or both of the MG growth arrays are set to zero, disable
            # all of them
            if z_mg is not None or df_mg is not None:
                raise ValueError("Must specify both z_mg and df_mg.")
            z_mg = None
            df_mg = None
            nz_mg = -1

        # Check to make sure specified amplitude parameter is consistent
        if ((A_s is None and sigma8 is None) or
                (A_s is not None and sigma8 is not None)):
            raise ValueError("Must set either A_s or sigma8 and not both.")

        # Set norm_pk to either A_s or sigma8
        norm_pk = A_s if A_s is not None else sigma8

        # The C library decides whether A_s or sigma8 was the input parameter
        # based on value, so we need to make sure this is consistent too
        if norm_pk >= 1e-5 and A_s is not None:
            raise ValueError("A_s must be less than 1e-5.")

        if norm_pk < 1e-5 and sigma8 is not None:
            raise ValueError("sigma8 must be greater than 1e-5.")

        # Make sure the neutrino parameters are consistent.
        if hasattr(m_nu, "__len__"):
            if (len(m_nu) != 3):
                raise ValueError("m_nu must be a float or array-like object "
                                 "with length 3.")
            elif ((mnu_type == 'sum') or
                    (mnu_type == 'sum_inverted') or
                    (mnu_type == 'sum_equal')):
                raise ValueError(
                    "mnu type '%s' cannot be passed with a list "
                    "of neutrino masses, only with a sum." % mnu_type)
            elif mnu_type is None:
                mnu_type = 'list'  # False

        else:
            try:
                m_nu = float(m_nu)
            except Exception:
                raise ValueError(
                    "m_nu must be a float or array-like object with "
                    "length 3.")

            if mnu_type is None:
                mnu_type = 'sum'
            m_nu = [m_nu]
            if (mnu_type == 'sum'
                    and m_nu[0] < (np.sqrt(7.62E-5) + np.sqrt(2.55E-3))
                    and (m_nu[0] > 1e-15)):
                raise ValueError("if mnu_type= sum, we are using the "
                                 "normal hierarchy and so m_nu must "
                                 "be less than (~)0.0592")
            elif (mnu_type == 'sum_inverted' and
                  m_nu[0] < (np.sqrt(2.43e-3 - 7.62e-5) + np.sqrt(2.43e-3))
                  and (m_nu[0] > 1e-15)):
                raise ValueError("if mnu_type= sum_inverted, we are using the "
                                 "inverted hierarchy and so m_nu must "
                                 "be less than (~)0.0978")

        # Check if any compulsory parameters are not set
        compul = [Omega_c, Omega_b, Omega_k, w0, wa, h, norm_pk,
                  n_s]
        names = ['Omega_c', 'Omega_b', 'Omega_k', 'w0', 'wa',
                 'h', 'norm_pk', 'n_s']
        for nm, item in zip(names, compul):
            if item is None:
                raise ValueError("Necessary parameter '%s' was not set "
                                 "(or set to None)." % nm)

        # Create new instance of ccl_parameters object
        # Create an internal status variable; needed to check massive neutrino
        # integral.
        T_CMB_old = lib.cvar.constants.T_CMB
        try:
            if T_CMB is not None:
                lib.cvar.constants.T_CMB = T_CMB
            status = 0
            if nz_mg == -1:
                # Create ccl_parameters without modified growth

                self._params, status = lib.parameters_create_nu(
                   Omega_c, Omega_b, Omega_k, Neff,
                   w0, wa, h, norm_pk,
                   n_s, bcm_log10Mc, bcm_etab, bcm_ks,
                   mu_0, sigma_0, mnu_types[mnu_type],
                   m_nu, status)
            else:
                # Create ccl_parameters with modified growth arrays
                self._params, status = lib.parameters_create_nu_vec(
                   Omega_c, Omega_b, Omega_k, Neff,
                   w0, wa, h, norm_pk,
                   n_s, bcm_log10Mc, bcm_etab, bcm_ks,
                   mu_0, sigma_0, z_mg, df_mg,
                   mnu_types[mnu_type], m_nu, status)
            check(status)
        finally:
            lib.cvar.constants.T_CMB = T_CMB_old

        if Omega_g is not None:
            total = self._params.Omega_g + self._params.Omega_l
            self._params.Omega_g = Omega_g
            self._params.Omega_l = total - Omega_g

    def __getitem__(self, key):
        """Access parameter values by name."""
        try:
            if key == 'mnu':
                val = lib.parameters_get_nu_masses(self._params, 3)
            else:
                val = getattr(self._params, key)
        except AttributeError:
            raise KeyError("Parameter '%s' not recognized." % key)
        return val

    def __setitem__(self, key, val):
        """Set parameter values by name."""
        raise NotImplementedError("Cosmology objects are immutable; create a "
                                  "new Cosmology() instance instead.")

    def __del__(self):
        """Free the C memory this object is managing as it is being garbage
        collected (hopefully)."""
        if hasattr(self, "cosmo"):
            if (self.cosmo is not None and
                    hasattr(lib, 'cosmology_free') and
                    lib.cosmology_free is not None):
                lib.cosmology_free(self.cosmo)
        if hasattr(self, "_params"):
            if (self._params is not None and
                    hasattr(lib, 'parameters_free') and
                    lib.parameters_free is not None):
                lib.parameters_free(self._params)

        # finally delete some attributes we don't want to be around for safety
        # when the context manager exits or if __del__ is called twice
        if hasattr(self, "cosmo"):
            delattr(self, "cosmo")
        if hasattr(self, "_params"):
            delattr(self, "_params")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """Free the C memory this object is managing when the context manager
        exits."""
        self.__del__()

    def __getstate__(self):
        # we are removing any C data before pickling so that the
        # is pure python when pickled.
        state = self.__dict__.copy()
        state.pop('cosmo', None)
        state.pop('_params', None)
        state.pop('_config', None)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        # we removed the C data when it was pickled, so now we unpickle
        # and rebuild the C data
        self._build_cosmo()

    def __repr__(self):
        """Make an eval-able string.

        This feature can be used like this:

        >>> import pyccl
        >>> cosmo = pyccl.Cosmology(...)
        >>> cosmo2 = eval(repr(cosmo))
        """
        string = "pyccl.Cosmology("
        string += ", ".join(
            "%s=%s" % (k, v)
            for k, v in self._params_init_kwargs.items()
            if k not in ['m_nu', 'mnu_type', 'z_mg', 'df_mg'])

        if hasattr(self._params_init_kwargs['m_nu'], '__len__'):
            string += ", m_nu=[%s, %s, %s]" % tuple(
                self._params_init_kwargs['m_nu'])
        else:
            string += ', m_nu=%s' % self._params_init_kwargs['m_nu']

        if self._params_init_kwargs['mnu_type'] is not None:
            string += ", mnu_type='%s'" % self._params_init_kwargs['mnu_type']
        else:
            string += ', mnu_type=None'

        if self._params_init_kwargs['z_mg'] is not None:
            vals = ", ".join(
                ["%s" % v for v in self._params_init_kwargs['z_mg']])
            string += ", z_mg=[%s]" % vals
        else:
            string += ", z_mg=%s" % self._params_init_kwargs['z_mg']

        if self._params_init_kwargs['df_mg'] is not None:
            vals = ", ".join(
                ["%s" % v for v in self._params_init_kwargs['df_mg']])
            string += ", df_mg=[%s]" % vals
        else:
            string += ", df_mg=%s" % self._params_init_kwargs['df_mg']

        string += ", "
        string += ", ".join(
            "%s='%s'" % (k, v) for k, v in self._config_init_kwargs.items())
        string += ")"

        return string

    def compute_distances(self):
        """Compute the distance splines."""
        if self.has_distances:
            return
        status = 0
        status = lib.cosmology_compute_distances(self.cosmo, status)
        check(status, self)

    def compute_growth(self):
        """Compute the growth function."""
        if self.has_growth:
            return

        if self['N_nu_mass'] > 0:
            warnings.warn(
                "CCL does not properly compute the linear growth rate in "
                "cosmological models with massive neutrinos!",
                category=CCLWarning)

            if self._params_init_kwargs['df_mg'] is not None:
                warnings.warn(
                    "Modified growth rates via the `df_mg` keyword argument "
                    "cannot be consistently combined with cosmological models "
                    "with massive neutrinos in CCL!",
                    category=CCLWarning)

            if (self._params_init_kwargs['mu_0'] > 0 or
                    self._params_init_kwargs['sigma_0'] > 0):
                warnings.warn(
                    "Modified growth rates via the mu-Sigma model "
                    "cannot be consistently combined with cosmological models "
                    "with massive neutrinos in CCL!",
                    category=CCLWarning)

        status = 0
        status = lib.cosmology_compute_growth(self.cosmo, status)
        check(status, self)

    def compute_linear_power(self):
        """Compute the linear power spectrum."""
        if self.has_linear_power:
            return

        if (self['N_nu_mass'] > 0 and
                self._config_init_kwargs['transfer_function'] in
                ['bbks', 'eisenstein_hu']):
            warnings.warn(
                "The '%s' linear power spectrum model does not properly "
                "account for massive neutrinos!" %
                self._config_init_kwargs['transfer_function'],
                category=CCLWarning)

        if self._config_init_kwargs['matter_power_spectrum'] == 'emu':
            warnings.warn(
                "None of the linear power spectrum models in CCL are "
                "consistent with that implictly used in the emulated "
                "non-linear power spectrum!",
                category=CCLWarning)

        # needed to init some models
        self.compute_growth()

        if ((self._config_init_kwargs['transfer_function'] ==
                'boltzmann_class') and not self.has_linear_power):
            pk_lin = get_class_pk_lin(self)
            psp = pk_lin.psp
        elif ((self._config_init_kwargs['transfer_function'] ==
                'boltzmann_camb') and not self.has_linear_power):
            pk_lin = get_camb_pk_lin(self)
            psp = pk_lin.psp
        else:
            psp = None

        if (psp is None and not self.has_linear_power and (
                self._config_init_kwargs['transfer_function'] in
                ['boltzmann_camb', 'boltzmann_class'])):
            raise CCLError("Either the CAMB or CLASS computation "
                           "failed silently! CCL could not compute the "
                           "transfer function!")

        # first do the linear matter power
        status = 0
        status = lib.cosmology_compute_linear_power(self.cosmo, psp, status)
        check(status, self)

    def compute_nonlin_power(self):
        """Compute the non-linear power spectrum."""
        if self.has_nonlin_power:
            return

        if self._config_init_kwargs['matter_power_spectrum'] != 'linear':
            if self._params_init_kwargs['df_mg'] is not None:
                warnings.warn(
                    "Modified growth rates via the `df_mg` keyword argument "
                    "cannot be consistently combined with '%s' for "
                    "computing the non-linear power spectrum!" %
                    self._config_init_kwargs['matter_power_spectrum'],
                    category=CCLWarning)

            if (self._params_init_kwargs['mu_0'] != 0 or
                    self._params_init_kwargs['sigma_0'] != 0):
                warnings.warn(
                    "mu-Sigma modified cosmologies "
                    "cannot be consistently combined with '%s' "
                    "for computing the non-linear power spectrum!" %
                    self._config_init_kwargs['matter_power_spectrum'],
                    category=CCLWarning)

        if (self['N_nu_mass'] > 0 and
                self._config_init_kwargs['baryons_power_spectrum'] == 'bcm'):
            warnings.warn(
                "The BCM baryonic correction model's default parameters "
                "were not calibrated for cosmological models with "
                "massive neutrinos!",
                category=CCLWarning)

        self.compute_distances()

        # needed for halofit, halomodel and linear options
        if self._config_init_kwargs['matter_power_spectrum'] != 'emu':
            self.compute_linear_power()

        # for the halo model we need to init the mass function stuff
        if self._config_init_kwargs['matter_power_spectrum'] == 'halo_model':
            self.compute_sigma()

        status = 0
        status = lib.cosmology_compute_nonlin_power(self.cosmo, status)
        check(status, self)

    def compute_sigma(self):
        """Compute the sigma(M) and mass function splines."""
        if self.has_sigma:
            return

        # we need these things before building the mass function splines
        if self['N_nu_mass'] > 0:
            # these are not consistent with anything - fun
            warnings.warn(
                "All of the halo mass function, concentration, and bias "
                "models in CCL are not properly calibrated for cosmological "
                "models with massive neutrinos!",
                category=CCLWarning)

        if self._config_init_kwargs['baryons_power_spectrum'] != 'nobaryons':
            warnings.warn(
                "All of the halo mass function, concentration, and bias "
                "models in CCL are not consistently adjusted for baryons "
                "when the power spectrum is via the BCM model!",
                category=CCLWarning)

        self.compute_growth()
        self.compute_linear_power()
        status = 0
        status = lib.cosmology_compute_sigma(self.cosmo, status)
        status = lib.cosmology_compute_hmfparams(self.cosmo, status)
        check(status, self)

    @property
    def has_distances(self):
        """Checks if the distances have been precomputed."""
        return bool(self.cosmo.computed_distances)

    @property
    def has_growth(self):
        """Checks if the growth function has been precomputed."""
        return bool(self.cosmo.computed_growth)

    @property
    def has_linear_power(self):
        """Checks if the linear power spectra have been precomputed."""
        return bool(self.cosmo.computed_linear_power)

    @property
    def has_nonlin_power(self):
        """Checks if the non-linear power spectra have been precomputed."""
        return bool(self.cosmo.computed_nonlin_power)

    @property
    def has_sigma(self):
        """Checks if sigma(M) and mass function splines are precomputed."""
        return (
            bool(self.cosmo.computed_sigma) and
            bool(self.cosmo.computed_hmfparams))

    def status(self):
        """Get error status of the ccl_cosmology object.

        .. note:: The error statuses are currently under development and
                  may not be fully descriptive.

        Returns:
            :obj:`str` containing the status message.
        """
        # Get status ID string if one exists
        if self.cosmo.status in error_types.keys():
            status = error_types[self.cosmo.status]
        else:
            status = self.cosmo.status

        # Get status message
        msg = self.cosmo.status_message

        # Return status information
        return "status(%s): %s" % (status, msg)
