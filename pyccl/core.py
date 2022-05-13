"""The core functionality of ccl, including the core data types. This includes
the cosmology and parameters objects used to instantiate a model from which one
can compute a set of theoretical predictions.
"""
import warnings
import numpy as np
import yaml
from inspect import getmembers, isfunction, signature

from . import ccllib as lib
from .errors import CCLError, CCLWarning, CCLDeprecationWarning
from ._types import error_types
from ._core import _docstring_extra_parameters
from .boltzmann import get_class_pk_lin, get_camb_pk_lin, get_isitgr_pk_lin
from .pyutils import check
from .pk2d import Pk2D
from .base import CCLObject, cache, unlock_instance, warn_api
from .parameters import CCLParameters, physical_constants

# Configuration types
transfer_function_types = {
    None: lib.transfer_none,
    'eisenstein_hu': lib.eisenstein_hu,
    'eisenstein_hu_nowiggles': lib.eisenstein_hu_nowiggles,
    'bbks': lib.bbks,
    'boltzmann_class': lib.boltzmann_class,
    'boltzmann_camb': lib.boltzmann_camb,
    'boltzmann_isitgr': lib.boltzmann_isitgr,
    'calculator': lib.pklin_from_input,
    'bacco': lib.pklin_from_input,
}

matter_power_spectrum_types = {
    None: lib.pknl_none,
    'halo_model': lib.halo_model,
    'halofit': lib.halofit,
    'linear': lib.linear,
    'emu': lib.emu,
    'calculator': lib.pknl_from_input,
    'camb': lib.pknl_from_boltzman,
    'bacco': lib.pknl_from_input,
}

baryons_power_spectrum_types = {
    'nobaryons': lib.nobaryons,
    'bcm': lib.bcm,
    'bacco': lib.nobaryons,
}

# List which transfer functions can be used with the muSigma_MG
# parameterisation of modified gravity

mass_function_types = {
    'angulo': lib.angulo,
    'tinker': lib.tinker,
    'tinker10': lib.tinker10,
    'watson': lib.watson,
    'shethtormen': lib.shethtormen,
}

halo_concentration_types = {
    'bhattacharya2011': lib.bhattacharya2011,
    'duffy2008': lib.duffy2008,
    'constant_concentration': lib.constant_concentration,
}

emulator_neutrinos_types = {
    'strict': lib.emu_strict,
    'equalize': lib.emu_equalize,
}


class Cosmology(CCLObject):
    """A cosmology including parameters and associated data.

    .. note:: Although some arguments default to `None`, they will raise a
              ValueError inside this function if not specified, so they are not
              optional.

    .. note:: The parameter Omega_g can be used to set the radiation density
              (not including relativistic neutrinos) to zero. Doing this will
              give you a model that is physically inconsistent since the
              temperature of the CMB will still be non-zero. Note however
              that this approximation is common for late-time LSS computations.

    Args:
        Omega_c (:obj:`float`):
            Cold dark matter density fraction.
        Omega_b (:obj:`float`):
            Baryonic matter density fraction.
        h (:obj:`float`):
            Hubble constant divided by 100 km/s/Mpc; unitless.
        A_s (:obj:`float`):
            Power spectrum normalization.
            Exactly one of A_s and sigma_8 is required.
        sigma8 (:obj:`float`):
            Variance of matter density perturbations at an 8 Mpc/h scale.
            Exactly one of A_s and sigma_8 is required.
        n_s (:obj:`float`):
            Primordial scalar perturbation spectral index.
        Omega_k (:obj:`float`, optional):
            Curvature density fraction. Defaults to 0.
        Omega_g (:obj:`float`, optional):
            Density in relativistic species except massless neutrinos.
            The default of `None` corresponds to setting this from the CMB
            temperature. Note that if a non-`None` value is given, it may
            result in a physically inconsistent model because the CMB
            temperature will still be non-zero in the parameters.
        Neff (:obj:`float`, optional):
            Effective number of massless neutrinos present.
            Defaults to 3.046.
        m_nu (:obj:`float`, optional):
            Total mass in eV of the massive neutrinos present. Defaults to 0.
        m_nu_type (:obj:`str`, optional):
            The type of massive neutrinos. Accepted types are ``'inverted'``,
            ``'normal'``, ``'equal'``, ``'single'``, and ``'list'``.
            The default of None is the same as 'normal'.
        w0 (:obj:`float`, optional):
            First order term of dark energy equation of state. Defaults to -1.
        wa (:obj:`float`, optional):
            Second order term of dark energy equation of state. Defaults to 0.
        T_CMB (:obj:`float`):
            The CMB temperature today.
            The default of ``None`` uses the global CCL value in
            ``pyccl.physical_constants.T_CMB``.
        bcm_log10Mc (:obj:`float`, optional):
            Deprecated; pass via ``extra parameters``.
            One of the parameters of the BCM model.
            Defaults to ``log10(1.2e14)``.
        bcm_etab (:obj:`float`, optional):
            Deprecated; pass via ``extra parameters``.
            One of the parameters of the BCM model. Defaults to 0.5.
        bcm_ks (:obj:`float`, optional):
            Deprecated; pass via ``extra parameters``.
            One of the parameters of the BCM model. Defaults to 55.0.
        mu_0 (:obj:`float`, optional):
            One of the parameters of the mu-Sigma modified gravity model.
            Defaults to 0.0
        sigma_0 (:obj:`float`, optional):
            One of the parameters of the mu-Sigma modified gravity model.
            Defaults to 0.0
        c1_mg (:obj:`float`, optional):
            Deprecated; pass via ``extra_parameters``.
            MG parameter that enters in the scale dependence of mu affecting
            its large scale behavior. Defaults to 1.
            See, e.g., Eqs. (46) in Ade et al. 2015, arXiv:1502.01590
            where their f1 and f2 functions are set equal to the commonly used
            ratio of dark energy density parameter at scale factor a over
            the dark energy density parameter today
        c2_mg (:obj:`float`, optional):
            Deprecated; pass via ``extra_parameters``.
            MG parameter that enters in the scale dependence of Sigma
            affecting its large scale behavior. Defaults to 1.
            See, e.g., Eqs. (47) in Ade et al. 2015, arXiv:1502.01590
            where their f1 and f2 functions are set equal to the commonly used
            ratio of dark energy density parameter at scale factor a over
            the dark energy density parameter today
        lambda_mg (:obj:`float`, optional):
            Deprecated; pass via ``extra_parameters``.
            MG parameter that sets the start of dependance on c1 and c2 MG
            parameters. Defaults to 0.0.
            See, e.g., Eqs. (46) & (47) in Ade et al. 2015, arXiv:1502.01590
            where their f1 and f2 functions are set equal to the commonly used
            ratio of dark energy density parameter at scale factor a over
            the dark energy density parameter today
        df_mg (array_like, optional):
            Deprecated; will be removed in a future release.
            Perturbations to the GR growth rate as a function of redshift
            :math:`\\Delta f`. Used to implement simple modified growth
            scenarios.
        z_mg (array_like, optional):
            Deprecated; will be removed in a future release.
            Redshifts corresponding to df_mg.
        transfer_function (:obj:`str`, optional):
            The transfer function to use. Defaults to ``'boltzmann_camb'``.
        matter_power_spectrum (:obj:`str`, optional):
            The matter power spectrum to use. Defaults to ``'halofit'``.
        baryons_power_spectrum (:obj:`str`, optional):
            The correction from baryonic effects to be implemented.
            Defaults to ``'nobaryons'``.
        mass_function (:obj:`str`, optional):
            Deprecated; pass via `extra_parameters` or use the `halos`
            sub-package.
            The mass function to use. Defaults to ``'tinker10'``.
        halo_concentration (:obj:`str`, optional):
            Deprecated; pass via `extra_parameters` or use the `halos`
            sub-package.
            The halo concentration relation to use.
            Defaults to ``'duffy2008'``.
        emulator_neutrinos (:obj:`str`, optional):
            If using CosmicEmu for the power spectrum, specified treatment
            of unequal neutrinos.
            Options are ``'strict'``, which will raise an error and quit if
            the user fails to pass either a set of three equal masses or a sum
            with ``m_nu_type = 'equal'``, and ``'equalize'``, which will
            redistribute masses to be equal right before calling the emulator,
            but results in internal inconsistencies. Defaults to ``'strict'``.
        extra_parameters (:obj:`dict`, optional):
            Dictionary holding extra parameters.
            Accepted keys are detailed below.

    """
    __doc__ += _docstring_extra_parameters
    from ._repr import _build_string_Cosmology as __repr__

    # Go through all functions in the main package and the subpackages
    # and make every function that takes `cosmo` as its first argument
    # an attribute of this class.
    from . import (background, baryons, boltzmann, cells,
                   correlations, covariances, neutrinos,
                   pk2d, power, tk3d, tracers, halos, nl_pt)
    subs = [background, boltzmann, baryons, cells, correlations, covariances,
            neutrinos, pk2d, power, tk3d, tracers, halos, nl_pt]
    funcs = [getmembers(sub, isfunction) for sub in subs]
    funcs = [func for sub in funcs for func in sub]
    for name, func in funcs:
        pars = list(signature(func).parameters)
        if pars and pars[0] == "cosmo":
            vars()[name] = func
    # clear unnecessary locals
    del (background, boltzmann, baryons, cells, correlations, covariances,
         neutrinos, pk2d, power, tk3d, tracers, halos, nl_pt,
         subs, funcs, func, name, pars)

    @warn_api
    def __init__(
            self, *, Omega_c=None, Omega_b=None, h=None, n_s=None,
            sigma8=None, A_s=None,
            Omega_k=0., Omega_g=None, Neff=3.046, m_nu=0., m_nu_type=None,
            w0=-1., wa=0., T_CMB=None,
            bcm_log10Mc=None, bcm_etab=None, bcm_ks=None,
            mu_0=0., sigma_0=0.,
            c1_mg=None, c2_mg=None, lambda_mg=None, z_mg=None, df_mg=None,
            transfer_function='boltzmann_camb',
            matter_power_spectrum='halofit',
            baryons_power_spectrum='nobaryons',
            mass_function=None, halo_concentration=None,
            emulator_neutrinos='strict',
            extra_parameters=None):

        # going to save these for later
        self._params_init_kwargs = dict(
            Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=n_s, sigma8=sigma8,
            A_s=A_s, Omega_k=Omega_k, Omega_g=Omega_g, Neff=Neff, m_nu=m_nu,
            m_nu_type=m_nu_type, w0=w0, wa=wa, T_CMB=T_CMB,
            bcm_log10Mc=bcm_log10Mc, bcm_etab=bcm_etab, bcm_ks=bcm_ks,
            mu_0=mu_0, sigma_0=sigma_0,
            c1_mg=c1_mg, c2_mg=c2_mg, lambda_mg=lambda_mg,
            z_mg=z_mg, df_mg=df_mg,
            extra_parameters=extra_parameters)

        self._config_init_kwargs = dict(
            transfer_function=transfer_function,
            matter_power_spectrum=matter_power_spectrum,
            baryons_power_spectrum=baryons_power_spectrum,
            mass_function=mass_function,
            halo_concentration=halo_concentration,
            emulator_neutrinos=emulator_neutrinos)

        self._build_cosmo()

        self._has_pk_lin = False
        self._pk_lin = {}
        self._has_pk_nl = False
        self._pk_nl = {}

    def _build_cosmo(self):
        """Assemble all of the input data into a valid ccl_cosmology object."""
        # We have to make all of the C stuff that goes into a cosmology
        # and then we make the cosmology.
        self._build_parameters(**self._params_init_kwargs)
        self._build_config(**self._config_init_kwargs)
        self.cosmo = lib.cosmology_create(self._params, self._config)
        self._spline_params = CCLParameters.get_params_dict("spline_params")
        self._gsl_params = CCLParameters.get_params_dict("gsl_params")
        self._accuracy_params = {**self._spline_params, **self._gsl_params}

        if self.cosmo.status != 0:
            raise CCLError(
                "(%d): %s"
                % (self.cosmo.status, self.cosmo.status_message))

    def write_yaml(self, filename):
        """Write a YAML representation of the parameters to file.

        Args:
            filename (:obj:`str`) Filename, file pointer, or stream to write "
                "parameters to."
        """
        def make_yaml_friendly(d):
            for k, v in d.items():
                if isinstance(v, np.floating):
                    d[k] = float(v)
                elif isinstance(v, np.integer):
                    d[k] = int(v)
                elif isinstance(v, bool):
                    d[k] = bool(v)
                elif isinstance(v, dict):
                    make_yaml_friendly(v)

        params = {**self._params_init_kwargs,
                  **self._config_init_kwargs}
        make_yaml_friendly(params)

        if isinstance(filename, str):
            with open(filename, "w") as fp:
                yaml.dump(params, fp,
                          default_flow_style=False, sort_keys=False)
        else:
            yaml.dump(params, filename,
                      default_flow_style=False, sort_keys=False)

    @classmethod
    def read_yaml(cls, filename, **kwargs):
        """Read the parameters from a YAML file.

        Args:
            filename (:obj:`str`) Filename, file pointer, or stream to read
                parameters from.
            **kwargs (dict) Additional keywords that supersede file contents
        """
        if isinstance(filename, str):
            with open(filename, 'r') as fp:
                params = yaml.load(fp, Loader=yaml.Loader)
        else:
            params = yaml.load(filename, Loader=yaml.Loader)

        if "sigma8" in params and params["sigma8"] == "nan":
            del params["sigma8"]
        if "A_s" in params and params["A_s"] == "nan":
            del params["A_s"]

        # Get the call signature of Cosmology (i.e., the names of
        # all arguments)
        init_param_names = signature(cls).parameters.keys()

        # Read the values we need from the loaded yaml dictionary. Missing
        # values take their default values from Cosmology.__init__
        inits = {k: params[k] for k in init_param_names if k in params}

        # Overwrite with extra values
        inits.update(kwargs)

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
        if (mass_function, halo_concentration) != (None, None):
            warnings.warn(
                "Arguments `mass_function` and `halo_concentration` are "
                "deprecated in `pyccl.Cosmology` and will be removed in a "
                "future release. To compute the Halo Model power spectrum "
                "refer to the 'halo_model' key in `extra_parameters`.",
                CCLDeprecationWarning)
            if ((mass_function not in mass_function_types.keys()) and
                    (mass_function is not None)):
                raise ValueError(
                    "'%s' is not a valid mass_function type. "
                    "Available options are: %s or None."
                    % (mass_function, mass_function_types.keys()))
            if ((halo_concentration not in halo_concentration_types.keys()) and
                    (halo_concentration is not None)):
                raise ValueError(
                    "'%s' is not a valid halo_concentration type. "
                    "Available options are: %s or None."
                    % (halo_concentration, halo_concentration_types.keys()))
        if emulator_neutrinos not in emulator_neutrinos_types.keys():
            raise ValueError(
                "'%s' is not a valid emulator neutrinos "
                "method. Available options are: %s"
                % (emulator_neutrinos, emulator_neutrinos_types.keys()))

        # Assign values to new ccl_configuration object
        # TODO: remove mass function and concentration from config
        if mass_function is None:
            mass_function = "tinker10"
        if halo_concentration is None:
            halo_concentration = "duffy2008"

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
            A_s=None, Omega_k=None, Neff=None, m_nu=None, m_nu_type=None,
            w0=None, wa=None, T_CMB=None,
            bcm_log10Mc=None, bcm_etab=None, bcm_ks=None,
            mu_0=None, sigma_0=None, c1_mg=None, c2_mg=None, lambda_mg=None,
            z_mg=None, df_mg=None, Omega_g=None,
            extra_parameters=None):
        """Build a ccl_parameters struct"""

        # Check to make sure Omega_k is within reasonable bounds.
        if Omega_k is not None and Omega_k < -1.0135:
            raise ValueError("Omega_k must be more than -1.0135.")

        # Set nz_mg (no. of redshift bins for modified growth fns.)
        if z_mg is not None and df_mg is not None:
            warnings.warn(
                "Arguments `z_mg` and `df_mg` are deprecated and will be "
                "removed in a future release.", CCLDeprecationWarning)
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
        if not (A_s is None) ^ (sigma8 is None):
            raise ValueError("Must set either A_s or sigma8 and not both.")

        # Set norm_pk to either A_s or sigma8
        norm_pk = A_s if A_s is not None else sigma8

        # The C library decides whether A_s or sigma8 was the input parameter
        # based on value, so we need to make sure this is consistent too
        if norm_pk >= 1e-5 and A_s is not None:
            raise ValueError("A_s must be less than 1e-5.")

        if norm_pk < 1e-5 and sigma8 is not None:
            raise ValueError("sigma8 must be greater than 1e-5.")

        # Make sure the neutrino parameters are consistent
        # and if a sum is given for mass, split into three masses.
        if hasattr(m_nu, "__len__"):
            if (len(m_nu) != 3):
                raise ValueError("m_nu must be a float or array-like object "
                                 "with length 3.")
            elif m_nu_type in ['normal', 'inverted', 'equal']:
                raise ValueError(
                    "m_nu_type '%s' cannot be passed with a list "
                    "of neutrino masses, only with a sum." % m_nu_type)
            elif m_nu_type is None:
                m_nu_type = 'list'  # False
            mnu_list = [0]*3
            for i in range(0, 3):
                mnu_list[i] = m_nu[i]

        else:
            try:
                m_nu = float(m_nu)
            except Exception:
                raise ValueError(
                    "m_nu must be a float or array-like object with "
                    "length 3.")

            if m_nu_type is None:
                m_nu_type = 'normal'
            m_nu = [m_nu]
            if (m_nu_type == 'normal'):
                if (m_nu[0] < (np.sqrt(7.62E-5) + np.sqrt(2.55E-3))
                        and (m_nu[0] > 1e-15)):
                    raise ValueError("if m_nu_type is 'normal', we are "
                                     "using the normal hierarchy and so "
                                     "m_nu must be greater than (~)0.0592 "
                                     "(or zero)")

                # Split the sum into 3 masses under normal hierarchy.
                if (m_nu[0] > 1e-15):
                    mnu_list = [0]*3
                    # This is a starting guess.
                    mnu_list[0] = 0.
                    mnu_list[1] = np.sqrt(7.62E-5)
                    mnu_list[2] = np.sqrt(2.55E-3)
                    sum_check = mnu_list[0] + mnu_list[1] + mnu_list[2]
                    # This is the Newton's method
                    while (np.abs(m_nu[0] - sum_check) > 1e-15):
                        dsdm1 = (1. + mnu_list[0] / mnu_list[1]
                                 + mnu_list[0] / mnu_list[2])
                        mnu_list[0] = mnu_list[0] - (sum_check
                                                     - m_nu[0]) / dsdm1
                        mnu_list[1] = np.sqrt(mnu_list[0]*mnu_list[0]
                                              + 7.62E-5)
                        mnu_list[2] = np.sqrt(mnu_list[0]*mnu_list[0]
                                              + 2.55E-3)
                        sum_check = mnu_list[0] + mnu_list[1] + mnu_list[2]

            elif (m_nu_type == 'inverted'):
                if (m_nu[0] < (np.sqrt(2.43e-3 - 7.62e-5) + np.sqrt(2.43e-3))
                        and (m_nu[0] > 1e-15)):
                    raise ValueError("if m_nu_type is 'inverted', we "
                                     "are using the inverted hierarchy "
                                     "and so m_nu must be greater than "
                                     "(~)0.0978 (or zero)")
                # Split the sum into 3 masses under inverted hierarchy.
                if (m_nu[0] > 1e-15):
                    mnu_list = [0]*3
                    mnu_list[0] = 0.  # This is a starting guess.
                    mnu_list[1] = np.sqrt(2.43e-3 - 7.62E-5)
                    mnu_list[2] = np.sqrt(2.43e-3)
                    sum_check = mnu_list[0] + mnu_list[1] + mnu_list[2]
                    # This is the Newton's method
                    while (np.abs(m_nu[0] - sum_check) > 1e-15):
                        dsdm1 = (1. + (mnu_list[0] / mnu_list[1])
                                 + (mnu_list[0] / mnu_list[2]))
                        mnu_list[0] = mnu_list[0] - (sum_check
                                                     - m_nu[0]) / dsdm1
                        mnu_list[1] = np.sqrt(mnu_list[0]*mnu_list[0]
                                              + 7.62E-5)
                        mnu_list[2] = np.sqrt(mnu_list[0]*mnu_list[0]
                                              - 2.43e-3)
                        sum_check = mnu_list[0] + mnu_list[1] + mnu_list[2]
            elif (m_nu_type == 'equal'):
                mnu_list = [0]*3
                mnu_list[0] = m_nu[0]/3.
                mnu_list[1] = m_nu[0]/3.
                mnu_list[2] = m_nu[0]/3.
            elif (m_nu_type == 'single'):
                mnu_list = [0]*3
                mnu_list[0] = m_nu[0]
                mnu_list[1] = 0.
                mnu_list[2] = 0.

        # Check which of the neutrino species are non-relativistic today
        N_nu_mass = 0
        if (np.abs(np.amax(m_nu) > 1e-15)):
            for i in range(0, 3):
                if (mnu_list[i] > 0.00017):  # Lesgourges et al. 2012
                    N_nu_mass = N_nu_mass + 1
            N_nu_rel = Neff - (N_nu_mass * 0.71611**4 * (4./11.)**(-4./3.))
            if N_nu_rel < 0.:
                raise ValueError("Neff and m_nu must result in a number "
                                 "of relativistic neutrino species greater "
                                 "than or equal to zero.")

        # Fill an array with the non-relativistic neutrino masses
        if N_nu_mass > 0:
            mnu_final_list = [0]*N_nu_mass
            relativistic = [0]*3
            for i in range(0, N_nu_mass):
                for j in range(0, 3):
                    if (mnu_list[j] > 0.00017 and relativistic[j] == 0):
                        relativistic[j] = 1
                        mnu_final_list[i] = mnu_list[j]
                        break
        else:
            mnu_final_list = [0.]

        # Check if any compulsory parameters are not set
        compul = [Omega_c, Omega_b, Omega_k, w0, wa, h, norm_pk, n_s]
        names = ['Omega_c', 'Omega_b', 'Omega_k',
                 'w0', 'wa', 'h', 'norm_pk', 'n_s']
        for nm, item in zip(names, compul):
            if item is None:
                raise ValueError("Necessary parameter '%s' was not set "
                                 "(or set to None)." % nm)

        # BCM parameters: deprecate old usage and sub-in defaults if needed
        if (extra_parameters is not None) and ("bcm" in extra_parameters):
            bcm = extra_parameters["bcm"]
        else:
            bcm = {"log10Mc": None, "etab": None, "ks": None}

        if any([par is not None for par in [bcm_log10Mc, bcm_etab, bcm_ks]]):
            warnings.warn(
                "BCM parameters as arguments of Cosmology are deprecated "
                "and will be removed in a future release. Specify them in "
                "`extra_parameters` instead, using the model key 'bcm', "
                "and omitting the 'bcm_' prefix from the parameter name.",
                CCLDeprecationWarning)
            bcm = {"log10Mc": bcm_log10Mc, "etab": bcm_etab, "ks": bcm_ks}

        bcm_defaults = {"log10Mc": np.log10(1.2e14), "etab": 0.5, "ks": 55}
        for par, val in bcm.items():
            if val is None:
                bcm[par] = bcm_defaults[par]
        log10Mc, etab, ks = bcm["log10Mc"], bcm["etab"], bcm["ks"]

        # Planck MG params: deprecate old usage and sub-in defaults if needed
        if (extra_parameters is not None) and \
                ("PlanckMG" in extra_parameters):
            planckMG = extra_parameters["PlanckMG"]
        else:
            planckMG = {"c1": None, "c2": None, "lambda": None}

        if any([par is not None for par in [c1_mg, c2_mg, lambda_mg]]):
            warnings.warn(
                "MG parameters [c1, c2, lambda] as arguments of Cosmology "
                "are deprecated and will be removed in a future release. "
                "Specify them in `extra_parameters` instead, using the model "
                "key 'PlanckMG', and omitting the '_mg' suffix from the "
                "parameter name.", CCLDeprecationWarning)
            planckMG = {"c1": c1_mg, "c2": c2_mg, "lambda": lambda_mg}

        planckMG_defaults = {"c1": 1.0, "c2": 1.0, "lambda": 0.0}
        for par, val in planckMG.items():
            if val is None:
                planckMG[par] = planckMG_defaults[par]
        c1, c2, lambda_ = planckMG["c1"], planckMG["c2"], planckMG["lambda"]

        # Create new instance of ccl_parameters object
        # Create an internal status variable; needed to check massive neutrino
        # integral.
        T_CMB_old = physical_constants.T_CMB
        try:
            if T_CMB is not None:
                physical_constants.T_CMB = T_CMB
            status = 0
            if nz_mg == -1:
                # Create ccl_parameters without modified growth
                self._params, status = lib.parameters_create_nu(
                    Omega_c, Omega_b, Omega_k, Neff, w0, wa, h,
                    norm_pk, n_s, log10Mc, etab, ks, mu_0, sigma_0,
                    c1, c2, lambda_, mnu_final_list, status)
            else:
                # Create ccl_parameters with modified growth arrays
                self._params, status = lib.parameters_create_nu_vec(
                    Omega_c, Omega_b, Omega_k, Neff, w0, wa, h,
                    norm_pk, n_s, log10Mc, etab, ks, mu_0, sigma_0,
                    c1, c2, lambda_,
                    z_mg, df_mg, mnu_final_list, status)
            check(status)
        finally:
            physical_constants.T_CMB = T_CMB_old

        if Omega_g is not None:
            total = self._params.Omega_g + self._params.Omega_l
            self._params.Omega_g = Omega_g
            self._params.Omega_l = total - Omega_g

    def __getitem__(self, key):
        """Access parameter values by name."""
        try:
            if key == 'm_nu':
                val = lib.parameters_get_nu_masses(self._params, 3)
            elif key == 'extra_parameters':
                val = self._params_init_kwargs["extra_parameters"]
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
        # This will create a new `Cosmology` object so we create another lock.
        state["_object_lock"] = type(state.pop("_object_lock"))()
        self.__dict__ = state
        # we removed the C data when it was pickled, so now we unpickle
        # and rebuild the C data
        self._build_cosmo()
        self._object_lock.lock()  # Lock on exit.

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

    @cache(maxsize=3)
    def _compute_linear_power(self):
        """Return the linear power spectrum."""
        if (self['N_nu_mass'] > 0 and
                self._config_init_kwargs['transfer_function'] in
                ['bbks', 'eisenstein_hu', 'eisenstein_hu_nowiggles', ]):
            warnings.warn(
                "The '%s' linear power spectrum model does not properly "
                "account for massive neutrinos!" %
                self._config_init_kwargs['transfer_function'],
                category=CCLWarning)

        if self._config_init_kwargs['matter_power_spectrum'] == 'emu':
            warnings.warn(
                "None of the linear power spectrum models in CCL are "
                "consistent with that implicitly used in the emulated "
                "non-linear power spectrum!",
                category=CCLWarning)

        # needed to init some models
        self.compute_growth()

        # Populate power spectrum splines
        trf = self._config_init_kwargs['transfer_function']
        pk = None
        rescale_s8 = True
        rescale_mg = True
        if trf is None:
            raise CCLError("You want to compute the linear power spectrum, "
                           "but you selected `transfer_function=None`.")
        elif trf == 'boltzmann_class':
            pk = get_class_pk_lin(self)
        elif trf == 'boltzmann_isitgr':
            rescale_mg = False
            pk = get_isitgr_pk_lin(self)
        elif trf == 'boltzmann_camb':
            pk_nl_from_camb = False
            if self._config_init_kwargs['matter_power_spectrum'] == "camb":
                pk_nl_from_camb = True
            pk = get_camb_pk_lin(self, nonlin=pk_nl_from_camb)
            if pk_nl_from_camb:
                pk, pk_nl = pk
                self._pk_nl['delta_matter:delta_matter'] = pk_nl
                self._has_pk_nl = True
                rescale_mg = False
                rescale_s8 = False
                if abs(self["mu_0"]) > 1e-14:
                    warnings.warn("You want to compute the non-linear power "
                                  "spectrum using CAMB. This cannot be "
                                  "consistently done with mu_0 > 0.",
                                  category=CCLWarning)
                if np.isfinite(self["sigma8"]) \
                        and not np.isfinite(self["A_s"]):
                    raise CCLError("You want to compute the non-linear "
                                   "power spectrum using CAMB and specified "
                                   "sigma8 but the non-linear power spectrum "
                                   "cannot be consistenty rescaled.")
        elif trf in ['bbks', 'eisenstein_hu', 'eisenstein_hu_nowiggles',
                     'bacco']:
            rescale_s8 = False
            rescale_mg = False
            pk = Pk2D.from_model(self, model=trf)

        # Rescale by sigma8/mu-sigma if needed
        if pk:
            status = 0
            status = lib.rescale_linpower(self.cosmo, pk.psp,
                                          int(rescale_mg),
                                          int(rescale_s8),
                                          status)
            check(status, self)

        return pk

    @unlock_instance(mutate=False)
    def compute_linear_power(self):
        """Compute the linear power spectrum."""
        if self.has_linear_power:
            return
        pk = self._compute_linear_power()
        # Assign
        self._pk_lin['delta_matter:delta_matter'] = pk
        if pk:
            self._has_pk_lin = True

    def _get_halo_model_nonlin_power(self):
        from . import halos as hal
        HM = {"mass_def": None, "mass_def_strict": None,
              "mass_function": None, "halo_bias": None,
              "concentration": None}
        try:
            extras = self._params_init_kwargs["extra_parameters"]
            HM.update(extras["halo_model"])
        except (KeyError, TypeError):
            warnings.warn(
                "You want to compute the Halo Model power spectrum but the "
                "`halo_model` parameters are not specified in "
                "`extra_parameters`. Refer to the documentation for the "
                "default values. "
                "Defaults will be overriden by the deprecated "
                "`mass_function` and `halo_concentration`, if specified.",
                CCLWarning)

        # override with deprecated Cosmology arguments
        mass_function = self._config_init_kwargs["mass_function"]
        halo_concentration = self._config_init_kwargs["halo_concentration"]
        if mass_function is not None:
            mfs = {"angulo": "Angulo12", "tinker": "Tinker08",
                   "tinker10": "Tinker10", "watson": "Watson13",
                   "shethtormen": "Sheth99"}
            HM["mass_function"] = mfs[mass_function]
        if halo_concentration is not None:
            cms = {"bhattacharya2011": "Bhattacharya13",
                   "duffy2008": "Duffy08",
                   "constant_concentration": "Constant"}
            HM["concentration"] = cms[halo_concentration]

        if (HM["halo_bias"] is None and
                HM["mass_function"] in ["Tinker10", "Sheth99"]):
            HM["halo_bias"] = HM["mass_function"]

        HM_defaults = {"mass_def": "200m", "mass_def_strict": False,
                       "mass_function": "Tinker10", "halo_bias": "Tinker10",
                       "concentration": "Duffy08"}
        for par, val in HM.items():
            if val is None:
                HM[par] = HM_defaults[par]

        hmd = hal.MassDef.from_name(HM["mass_def"])()
        mf_pars = {"mass_def": hmd, "mass_def_strict": HM["mass_def_strict"]}
        hb_pars = mf_pars.copy()
        if HM["mass_function"] == "Sheth99":
            mf_pars["use_delta_c_fit"] = True
        hmf = hal.MassFunc.from_name(HM["mass_function"])(**mf_pars)
        hbf = hal.HaloBias.from_name(HM["halo_bias"])(**hb_pars)
        hmc = hal.HMCalculator(mass_function=hmf, halo_bias=hbf, mass_def=hmd)
        cM_pars = {"mass_def": hmd}
        if HM["concentration"] == "Constant":
            cM_pars["c"] = 4.
        cM = hal.Concentration.from_name(HM["concentration"])(**cM_pars)
        prof = hal.HaloProfileNFW(c_m_relation=cM)
        return hal.halomod_Pk2D(self, hmc, prof, normprof=True)

    @cache(maxsize=3)
    def _compute_nonlin_power(self):
        """Return the non-linear power spectrum."""
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

        # Populate power spectrum splines
        mps = self._config_init_kwargs['matter_power_spectrum']
        # needed for halofit, halomodel and linear options
        if (mps != 'emu') and (mps is not None):
            self.compute_linear_power()

        if mps == "camb" and self._has_pk_nl:
            # Already computed
            return self._pk_nl['delta_matter:delta_matter']

        pk = None
        if mps is None:
            raise CCLError("You want to compute the non-linear power "
                           "spectrum, but you selected "
                           "`matter_power_spectrum=None`.")
        elif mps == 'halo_model':
            pk = self._get_halo_model_nonlin_power()
        elif mps == 'halofit':
            pkl = self._pk_lin['delta_matter:delta_matter']
            if pkl is None:
                raise CCLError("The linear power spectrum is a "
                               "necessary input for halofit")
            pk = pkl.apply_halofit(self)
        elif mps == 'emu':
            pk = Pk2D.from_model(self, model='emu')
        elif mps == 'linear':
            pk = self._pk_lin['delta_matter:delta_matter']
        elif mps in ['bacco', ]:  # other emulators go in here
            pkl = self._pk_lin['delta_matter:delta_matter']
            pk = pkl.apply_nonlin_model(self, model=mps)

        # Correct for baryons if required
        bps = self._config_init_kwargs['baryons_power_spectrum']
        if bps in ['bcm', 'bacco', ]:
            pk = pk.include_baryons(self, model=bps)

        return pk

    @unlock_instance(mutate=False)
    def compute_nonlin_power(self):
        """Compute the non-linear power spectrum."""
        if self.has_nonlin_power:
            return
        pk = self._compute_nonlin_power()
        # Assign
        self._pk_nl['delta_matter:delta_matter'] = pk
        if pk:
            self._has_pk_nl = True

    def compute_sigma(self):
        """Compute the sigma(M) spline."""
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

        self.compute_linear_power()
        pk = self._pk_lin['delta_matter:delta_matter']
        if pk is None:
            raise CCLError("Linear power spectrum can't be None")
        status = 0
        status = lib.cosmology_compute_sigma(self.cosmo, pk.psp, status)
        check(status, self)

    def get_linear_power(self, name='delta_matter:delta_matter'):
        """Get the :class:`~pyccl.pk2d.Pk2D` object associated with
        the linear power spectrum with name `name`.

        Args:
            name (:obj:`str` or `None`): name of the power spectrum to
                return. If `None`, `'delta_matter:delta_matter'` will
                be used.

        Returns:
            :class:`~pyccl.pk2d.Pk2D` object containing the linear
            power spectrum with name `name`.
        """
        if name is None:
            name = 'delta_matter:delta_matter'
        if name not in self._pk_lin:
            raise KeyError("Unknown power spectrum %s." % name)
        return self._pk_lin[name]

    def get_nonlin_power(self, name='delta_matter:delta_matter'):
        """Get the :class:`~pyccl.pk2d.Pk2D` object associated with
        the non-linear power spectrum with name `name`.

        Args:
            name (:obj:`str` or `None`): name of the power spectrum to
                return. If `None`, `'delta_matter:delta_matter'` will
                be used.

        Returns:
            :class:`~pyccl.pk2d.Pk2D` object containing the non-linear
            power spectrum with name `name`.
        """
        if name is None:
            name = 'delta_matter:delta_matter'
        if name not in self._pk_nl:
            raise KeyError("Unknown power spectrum %s." % name)
        return self._pk_nl[name]

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
        return self._has_pk_lin

    @property
    def has_nonlin_power(self):
        """Checks if the non-linear power spectra have been precomputed."""
        return self._has_pk_nl

    @property
    def has_sigma(self):
        """Checks if sigma(M) is precomputed."""
        return bool(self.cosmo.computed_sigma)

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


def CosmologyVanillaLCDM(**kwargs):
    """A cosmology with typical flat Lambda-CDM parameters (`Omega_c=0.25`,
    `Omega_b = 0.05`, `Omega_k = 0`, `sigma8 = 0.81`, `n_s = 0.96`, `h = 0.67`,
    no massive neutrinos).

    Arguments:
        **kwargs (dict): a dictionary of parameters passed as arguments
            to the `Cosmology` constructor. It should not contain any of
            the LambdaCDM parameters (`"Omega_c"`, `"Omega_b"`, `"n_s"`,
            `"sigma8"`, `"A_s"`, `"h"`), since these are fixed.
    """
    p = {'Omega_c': 0.25,
         'Omega_b': 0.05,
         'h': 0.67,
         'n_s': 0.96,
         'sigma8': 0.81,
         'A_s': None}
    if set(p).intersection(set(kwargs)):
        raise ValueError(
            f"You cannot change the LCDM parameters: {list(p.keys())}.")
    # TODO py39+: dictionary union operator `(p | kwargs)`.
    return Cosmology(**{**p, **kwargs})


class CosmologyCalculator(Cosmology):
    """A "calculator-mode" CCL `Cosmology` object.
    This allows users to build a cosmology from a set of arrays
    describing the background expansion, linear growth factor and
    linear and non-linear power spectra, which can then be used
    to compute more complex observables (e.g. angular power
    spectra or halo-model quantities). These are stored in
    `background`, `growth`, `pk_linear` and `pk_nonlin`.

    .. note:: Although in principle these arrays should suffice
              to compute most observable quantities some
              calculations implemented in CCL (e.g. the halo
              mass function) requires knowledge of basic
              cosmological parameters such as :math:`\\Omega_M`.
              For this reason, users must pass a minimal set
              of :math:`\\Lambda` CDM cosmological parameters.

    Args:
        Omega_c (:obj:`float`): Cold dark matter density fraction.
        Omega_b (:obj:`float`): Baryonic matter density fraction.
        h (:obj:`float`): Hubble constant divided by 100 km/s/Mpc;
            unitless.
        A_s (:obj:`float`): Power spectrum normalization. Exactly
            one of A_s and sigma_8 is required.
        sigma8 (:obj:`float`): Variance of matter density
            perturbations at an 8 Mpc/h scale. Exactly one of A_s
            and sigma_8 is required.
        n_s (:obj:`float`): Primordial scalar perturbation spectral
            index.
        Omega_k (:obj:`float`, optional): Curvature density fraction.
            Defaults to 0.
        Omega_g (:obj:`float`, optional): Density in relativistic species
            except massless neutrinos. The default of `None` corresponds
            to setting this from the CMB temperature. Note that if a
            non-`None` value is given, this may result in a physically
            inconsistent model because the CMB temperature will still
            be non-zero in the parameters.
        Neff (:obj:`float`, optional): Effective number of massless
            neutrinos present. Defaults to 3.046.
        m_nu (:obj:`float`, optional): Total mass in eV of the massive
            neutrinos present. Defaults to 0.
        m_nu_type (:obj:`str`, optional): The type of massive neutrinos.
            Should be one of 'inverted', 'normal', 'equal', 'single', or
            'list'. The default of None is the same as 'normal'.
        w0 (:obj:`float`, optional): First order term of dark energy
            equation of state. Defaults to -1.
        wa (:obj:`float`, optional): Second order term of dark energy
            equation of state. Defaults to 0.
        T_CMB (:obj:`float`): The CMB temperature today. The default of
            ``None`` uses the global CCL value in
            ``pyccl.physical_constants.T_CMB``.
        background (:obj:`dict`): a dictionary describing the background
            expansion. It must contain three mandatory entries: `'a'`: an
            array of monotonically ascending scale-factor values. `'chi'`:
            an array containing the values of the comoving radial distance
            (in units of Mpc) at the scale factor values stored in `a`.
            '`h_over_h0`': an array containing the Hubble expansion rate at
            the scale factor values stored in `a`, divided by its value
            today (at `a=1`).
        growth (:obj:`dict`): a dictionary describing the linear growth of
            matter fluctuations. It must contain three mandatory entries:
            `'a'`: an array of monotonically ascending scale-factor
            values. `'growth_factor'`: an array containing the values of
            the linear growth factor :math:`D(a)` at the scale factor
            values stored in `a`. '`growth_rate`': an array containing the
            growth rate :math:`f(a)\\equiv d\\log D/d\\log a` at the scale
            factor values stored in `a`.
        pk_linear (:obj:`dict`): a dictionary containing linear power
            spectra. It must contain the following mandatory entries:
            `'a'`: an array of scale factor values. `'k'`: an array of
            comoving wavenumbers in units of inverse Mpc.
            `'delta_matter:delta_matter'`: a 2D array of shape
            `(n_a, n_k)`, where `n_a` and `n_k` are the lengths of
            `'a'` and `'k'` respectively, containing the linear matter
            power spectrum :math:`P(k,a)`. This dictionary may also
            contain other entries with keys of the form `'q1:q2'`,
            containing other cross-power spectra between quantities
            `'q1'` and `'q2'`.
        pk_nonlin (:obj:`dict`): a dictionary containing non-linear
            power spectra. It must contain the following mandatory
            entries: `'a'`: an array of scale factor values.
            `'k'`: an array of comoving wavenumbers in units of
            inverse Mpc. If `nonlinear_model` is `None`, it should also
            contain `'delta_matter:delta_matter'`: a 2D array of
            shape `(n_a, n_k)`, where `n_a` and `n_k` are the lengths
            of `'a'` and `'k'` respectively, containing the non-linear
            matter power spectrum :math:`P(k,a)`. This dictionary may
            also contain other entries with keys of the form `'q1:q2'`,
            containing other cross-power spectra between quantities
            `'q1'` and `'q2'`.
        nonlinear_model (:obj:`str`, :obj:`dict` or `None`): model to
            compute non-linear power spectra. If a string, the associated
            non-linear model will be applied to all entries in `pk_linear`
            which do not appear in `pk_nonlin`. If a dictionary, it should
            contain entries of the form `'q1:q2': model`, where `model`
            is a string designating the non-linear model to apply to the
            `'q1:q2'` power spectrum, which must also be present in
            `pk_linear`. If `model` is `None`, this non-linear power
            spectrum will not be calculated. If `nonlinear_model` is
            `None`, no additional non-linear power spectra will be
            computed. The only non-linear model supported is `'halofit'`,
            corresponding to the "HALOFIT" transformation of
            Takahashi et al. 2012 (arXiv:1208.2701).
    """
    @warn_api
    def __init__(
            self, *, Omega_c=None, Omega_b=None, h=None, n_s=None,
            sigma8=None, A_s=None, Omega_k=0., Omega_g=None,
            Neff=3.046, m_nu=0., m_nu_type=None, w0=-1., wa=0.,
            T_CMB=None, background=None, growth=None,
            pk_linear=None, pk_nonlin=None, nonlinear_model=None):
        if pk_linear:
            transfer_function = 'calculator'
        else:
            transfer_function = None
        if pk_nonlin or nonlinear_model:
            matter_power_spectrum = 'calculator'
        else:
            matter_power_spectrum = None

        # Cosmology
        super(CosmologyCalculator, self).__init__(
            Omega_c=Omega_c, Omega_b=Omega_b, h=h,
            n_s=n_s, sigma8=sigma8, A_s=A_s,
            Omega_k=Omega_k, Omega_g=Omega_g,
            Neff=Neff, m_nu=m_nu, m_nu_type=m_nu_type,
            w0=w0, wa=wa, T_CMB=T_CMB,
            transfer_function=transfer_function,
            matter_power_spectrum=matter_power_spectrum)

        # Parse arrays
        has_bg = background is not None
        has_dz = growth is not None
        has_pklin = pk_linear is not None
        has_pknl = pk_nonlin is not None
        has_nonlin_model = nonlinear_model is not None

        if has_bg:
            self._init_bg(background)
        if has_dz:
            self._init_growth(growth)
        if has_pklin:
            self._init_pklin(pk_linear)
        if has_pknl:
            self._init_pknl(pk_nonlin, has_nonlin_model)
        self._apply_nonlinear_model(nonlinear_model)

    def _init_bg(self, background):
        # Background
        if not isinstance(background, dict):
            raise TypeError("`background` must be a dictionary.")
        if (('a' not in background) or ('chi' not in background) or
                ('h_over_h0' not in background)):
            raise ValueError("`background` must contain keys "
                             "'a', 'chi' and 'h_over_h0'")
        a = background['a']
        chi = background['chi']
        hoh0 = background['h_over_h0']
        # Check that input arrays have the same size.
        if not (a.shape == chi.shape == hoh0.shape):
            raise ValueError("Input arrays must have the same size.")
        # Check that `a` is a monotonically increasing array.
        if not np.array_equal(a, np.sort(a)):
            raise ValueError("Input scale factor array is not "
                             "monotonically increasing.")
        # Check that the last element of a_array_back is 1:
        if np.abs(a[-1]-1.0) > 1e-5:
            raise ValueError("The last element of the input scale factor"
                             "array must be 1.0.")
        status = 0
        status = lib.cosmology_distances_from_input(self.cosmo,
                                                    a, chi, hoh0,
                                                    status)
        check(status, self)

    def _init_growth(self, growth):
        # Growth
        if not isinstance(growth, dict):
            raise TypeError("`growth` must be a dictionary.")
        if (('a' not in growth) or ('growth_factor' not in growth) or
                ('growth_rate' not in growth)):
            raise ValueError("`growth` must contain keys "
                             "'a', 'growth_factor' and 'growth_rate'")
        a = growth['a']
        dz = growth['growth_factor']
        fz = growth['growth_rate']
        # Check that input arrays have the same size.
        if not (a.shape == dz.shape
                == fz.shape):
            raise ValueError("Input arrays must have the same size.")
        # Check that a_array_grth is a monotonically increasing array.
        if not np.array_equal(a,
                              np.sort(a)):
            raise ValueError("Input scale factor array is not "
                             "monotonically increasing.")
        # Check that the last element of a is 1:
        if np.abs(a[-1]-1.0) > 1e-5:
            raise ValueError("The last element of the input scale factor"
                             "array must be 1.0.")
        status = 0
        status = lib.cosmology_growth_from_input(self.cosmo,
                                                 a, dz, fz,
                                                 status)
        check(status, self)

    def _init_pklin(self, pk_linear):
        # Linear power spectrum
        if not isinstance(pk_linear, dict):
            raise TypeError("`pk_linear` must be a dictionary")
        if (('delta_matter:delta_matter' not in pk_linear) or
                ('a' not in pk_linear) or ('k' not in pk_linear)):
            raise ValueError("`pk_linear` must contain keys 'a', 'k' "
                             "and 'delta_matter:delta_matter' "
                             "(at least)")

        # Check that `a` is a monotonically increasing array.
        if not np.array_equal(pk_linear['a'], np.sort(pk_linear['a'])):
            raise ValueError("Input scale factor array in `pk_linear` is not "
                             "monotonically increasing.")

        # needed for high-z extrapolation
        self.compute_growth()

        na = len(pk_linear['a'])
        nk = len(pk_linear['k'])
        lk = np.log(pk_linear['k'])
        pk_names = [key for key in pk_linear if key not in ('a', 'k')]
        for n in pk_names:
            qs = n.split(':')
            if len(qs) != 2:
                raise ValueError("Power spectrum label %s could " % n +
                                 "not be parsed. Label must be of the " +
                                 "form 'q1:q2'")
            pk = pk_linear[n]
            if pk.shape != (na, nk):
                raise ValueError("Power spectrum %s has shape " % n +
                                 str(pk.shape) + " but shape " +
                                 "(%d, %d) was expected." % (na, nk))
            # Spline in log-space if the P(k) is positive-definite
            use_log = np.all(pk > 0)
            if use_log:
                pk = np.log(pk)
            # Initialize and store
            pk = Pk2D(pkfunc=None,
                      a_arr=pk_linear['a'], lk_arr=lk, pk_arr=pk,
                      is_logp=use_log, extrap_order_lok=1,
                      extrap_order_hik=2, cosmo=None)
            self._pk_lin[n] = pk
        # Set linear power spectrum as initialized
        self._has_pk_lin = True

    def _init_pknl(self, pk_nonlin, has_nonlin_model):
        # Non-linear power spectrum
        if not isinstance(pk_nonlin, dict):
            raise TypeError("`pk_nonlin` must be a dictionary")
        if (('a' not in pk_nonlin) or ('k' not in pk_nonlin)):
            raise ValueError("`pk_nonlin` must contain keys "
                             "'a' and 'k' (at least)")
        # Check that `a` is a monotonically increasing array.
        if not np.array_equal(pk_nonlin['a'], np.sort(pk_nonlin['a'])):
            raise ValueError("Input scale factor array in `pk_nonlin` is not "
                             "monotonically increasing.")

        if ((not has_nonlin_model) and
                ('delta_matter:delta_matter' not in pk_nonlin)):
            raise ValueError("`pk_nonlin` must contain key "
                             "'delta_matter:delta_matter' or "
                             "use halofit to compute it")
        na = len(pk_nonlin['a'])
        nk = len(pk_nonlin['k'])
        lk = np.log(pk_nonlin['k'])
        pk_names = [key for key in pk_nonlin if key not in ('a', 'k')]
        for n in pk_names:
            qs = n.split(':')
            if len(qs) != 2:
                raise ValueError("Power spectrum label %s could " % n +
                                 "not be parsed. Label must be of the " +
                                 "form 'q1:q2'")
            pk = pk_nonlin[n]
            if pk.shape != (na, nk):
                raise ValueError("Power spectrum %s has shape " % n +
                                 str(pk.shape) + " but shape " +
                                 "(%d, %d) was expected." % (na, nk))
            # Spline in log-space if the P(k) is positive-definite
            use_log = np.all(pk > 0)
            if use_log:
                pk = np.log(pk)
            # Initialize and store
            pk = Pk2D(pkfunc=None,
                      a_arr=pk_nonlin['a'], lk_arr=lk, pk_arr=pk,
                      is_logp=use_log, extrap_order_lok=1,
                      extrap_order_hik=2, cosmo=None)
            self._pk_nl[n] = pk
        # Set non-linear power spectrum as initialized
        self._has_pk_nl = True

    def _apply_nonlinear_model(self, nonlin_model):
        if nonlin_model is None:
            return
        elif isinstance(nonlin_model, str):
            if not self._pk_lin:
                raise ValueError("You asked to use the non-linear "
                                 "model " + nonlin_model + " but "
                                 "provided no linear power spectrum "
                                 "to apply it to.")
            nld = {n: nonlin_model
                   for n in self._pk_lin}
        elif isinstance(nonlin_model, dict):
            nld = nonlin_model
        else:
            raise TypeError("`nonlinear_model` must be a string, "
                            "a dictionary or `None`")

        for name, model in nld.items():
            if name in self._pk_nl:
                continue

            if name not in self._pk_lin:
                raise KeyError(name + " is not a "
                               "known linear power spectrum")

            if ((name == 'delta_matter:delta_matter') and
                    (model is None)):
                raise ValueError("The non-linear matter power spectrum "
                                 "can't be `None`")

            if model == 'halofit':
                pkl = self._pk_lin[name]
                self._pk_nl[name] = pkl.apply_halofit(self)
            elif model is None:
                pass
            else:
                raise KeyError(model + " is not a valid "
                               "non-linear model.")
        # Set non-linear power spectrum as initialized
        self._has_pk_nl = True
