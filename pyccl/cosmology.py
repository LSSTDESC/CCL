"""The core functionality of ccl, including the core data types. This includes
the cosmology and parameters objects used to instantiate a model from which one
can compute a set of theoretical predictions.
"""
__all__ = ("TransferFunctions", "MatterPowerSpectra",
           "Cosmology", "CosmologyVanillaLCDM", "CosmologyCalculator",)

import warnings
import yaml
from enum import Enum
from inspect import getmembers, isfunction, signature
from numbers import Real
from typing import Iterable

import numpy as np

from . import (
    CCLError, CCLDeprecationWarning, CCLObject, CCLParameters, CLevelErrors,
    CosmologyParams, DEFAULT_POWER_SPECTRUM, DefaultParams, Pk2D, cache, check,
    lib, unlock_instance, warn_api, deprecated)
from . import physical_constants as const


class TransferFunctions(Enum):
    BBKS = "bbks"
    EISENSTEIN_HU = "eisenstein_hu"
    EISENSTEIN_HU_NOWIGGLES = "eisenstein_hu_nowiggles"
    BOLTZMANN_CLASS = "boltzmann_class"
    BOLTZMANN_CAMB = "boltzmann_camb"
    BOLTZMANN_ISITGR = "boltzmann_isitgr"
    CALCULATOR = "calculator"


class MatterPowerSpectra(Enum):
    LINEAR = "linear"
    HALOFIT = "halofit"
    HALOMODEL = "halomodel"
    EMU = "emu"
    CAMB = "camb"
    CALCULATOR = "calculator"


# Configuration types
transfer_function_types = {
    'eisenstein_hu': lib.eisenstein_hu,
    'eisenstein_hu_nowiggles': lib.eisenstein_hu_nowiggles,
    'bbks': lib.bbks,
    'boltzmann_class': lib.boltzmann_class,
    'boltzmann_camb': lib.boltzmann_camb,
    'boltzmann_isitgr': lib.boltzmann_isitgr,
    'calculator': lib.pklin_from_input
}


matter_power_spectrum_types = {
    'halo_model': lib.halo_model,
    'halofit': lib.halofit,
    'linear': lib.linear,
    'emu': lib.emu,
    'calculator': lib.pknl_from_input,
    'camb': lib.pknl_from_boltzman
}

baryons_power_spectrum_types = {
    'nobaryons': lib.nobaryons,
    'bcm': lib.bcm
}

mass_function_types = {
    'angulo': lib.angulo,
    'tinker': lib.tinker,
    'tinker10': lib.tinker10,
    'watson': lib.watson,
    'shethtormen': lib.shethtormen
}

halo_concentration_types = {
    'bhattacharya2011': lib.bhattacharya2011,
    'duffy2008': lib.duffy2008,
    'constant_concentration': lib.constant_concentration,
}

emulator_neutrinos_types = {
    'strict': lib.emu_strict,
    'equalize': lib.emu_equalize
}

_TOP_LEVEL_MODULES = ("",)


def _make_methods(cls=None, *, modules=_TOP_LEVEL_MODULES, name=None):
    """Assign all functions in ``modules`` which take ``name`` as their
    first argument as methods of the class ``cls``.
    """
    import functools
    from importlib import import_module

    if cls is None:
        # called with parentheses
        return functools.partial(_make_methods, modules=modules)

    pkg = __name__.rsplit(".")[0]
    modules = [import_module(f".{module}", pkg) for module in modules]
    funcs = [getmembers(module, isfunction) for module in modules]
    funcs = [func for sublist in funcs for func in sublist]

    for name, func in funcs:
        pars = signature(func).parameters
        if pars and list(pars)[0] == "cosmo":
            setattr(cls, name, func)

    return cls


@_make_methods(modules=("", "halos", "nl_pt",), name="cosmo")
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
              See the module level documentation of `pyccl.core` for details.

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
        m_nu (:obj:`float` or array_like, optional):
            Mass in eV of the massive neutrinos present. Defaults to 0.
            If a sequence is passed, it is assumed that the elements of the
            sequence represent the individual neutrino masses.
        mass_split (:obj:`str`, optional): Type of massive neutrinos. Should
            be one of 'single', 'equal', 'normal', 'inverted'. 'single' treats
            the mass as being held by one massive neutrino. The other options
            split the mass into 3 massive neutrinos. Ignored if a sequence is
            passed in m_nu. Default is 'normal'.
        w0 (:obj:`float`, optional): First order term of dark energy equation
            of state. Defaults to -1.
        wa (:obj:`float`, optional): Second order term of dark energy equation
            of state. Defaults to 0.
        T_CMB (:obj:`float`): The CMB temperature today. The default of
            is 2.725.
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
        c1_mg (:obj:`float`, optional): MG parameter that enters in the scale
            dependence of mu affecting its large scale behavior. Default to 1.
            See, e.g., Eqs. (46) in Ade et al. 2015, arXiv:1502.01590
            where their f1 and f2 functions are set equal to the commonly used
            ratio of dark energy density parameter at scale factor a over
            the dark energy density parameter today
        c2_mg (:obj:`float`, optional): MG parameter that enters in the scale
            dependence of Sigma affecting its large scale behavior. Default 1.
            See, e.g., Eqs. (47) in Ade et al. 2015, arXiv:1502.01590
            where their f1 and f2 functions are set equal to the commonly used
            ratio of dark energy density parameter at scale factor a over
            the dark energy density parameter today
        lambda_mg (:obj:`float`, optional): MG parameter that sets the start
            of dependance on c1 and c2 MG parameters. Defaults to 0.0
            See, e.g., Eqs. (46) & (47) in Ade et al. 2015, arXiv:1502.01590
            where their f1 and f2 functions are set equal to the commonly used
            ratio of dark energy density parameter at scale factor a over
            the dark energy density parameter today
        df_mg (array_like, optional): Perturbations to the GR growth rate as
            a function of redshift :math:`\\Delta f`. Used to implement simple
            modified growth scenarios.
        z_mg (array_like, optional): Array of redshifts corresponding to df_mg.
        transfer_function (:obj:`str`, optional): The transfer function to
            use. Defaults to 'boltzmann_camb'.
        matter_power_spectrum (:obj:`str`, optional): The matter power
            spectrum to use. Defaults to 'halofit'.
        baryons_power_spectrum (:obj:`str`, optional): The correction from
            baryonic effects to be implemented. Defaults to 'nobaryons'.
        mass_function (:obj:`str`, optional): The mass function to use.
            Defaults to 'tinker10' (2010).
        halo_concentration (:obj:`str`, optional): The halo concentration
            relation to use. Defaults to Duffy et al. (2008) 'duffy2008'.
        emulator_neutrinos (:obj:`str`, optional): If using the emulator for
            the power spectrum, specified treatment of unequal neutrinos.
            Options are 'strict', which will raise an error and quit if the
            user fails to pass either a set of three equal masses or a sum with
            mass_split = 'equal', and 'equalize', which will redistribute
            masses to be equal right before calling the emulator but results in
            internal inconsistencies. Defaults to 'strict'.
        extra_parameters (:obj:`dict`, optional): Dictionary holding extra
            parameters. Currently supports extra parameters for CAMB, with
            details described below. Defaults to None.
        T_ncdm (:obj:`float`): Non-CDM temperature in units of photon
            temperature. The default is 0.71611.

    Currently supported extra parameters for CAMB are:

        * `halofit_version`
        * `HMCode_A_baryon`
        * `HMCode_eta_baryon`
        * `HMCode_logT_AGN`
        * `kmax`
        * `lmax`
        * `dark_energy_model`

    Consult the CAMB documentation for their usage. These parameters are passed
    in a :obj:`dict` to `extra_parameters` as::

        extra_parameters = {"camb": {"halofit_version": "mead2020_feedback",
                                     "HMCode_logT_AGN": 7.8}}

    """
    from .base.repr_ import build_string_Cosmology as __repr__
    __eq_attrs__ = ("_params_init_kwargs", "_config_init_kwargs",
                    "_accuracy_params",)

    @warn_api(pairs=[("m_nu_type", "mass_split")])
    def __init__(
            self, *, Omega_c=None, Omega_b=None, h=None, n_s=None,
            sigma8=None, A_s=None, Omega_k=0., Omega_g=None,
            Neff=None, m_nu=0., mass_split='normal', w0=-1., wa=0.,
            T_CMB=DefaultParams.T_CMB,
            bcm_log10Mc=None, bcm_etab=None, bcm_ks=None,
            mu_0=0, sigma_0=0, c1_mg=1, c2_mg=1, lambda_mg=0,
            z_mg=None, df_mg=None,
            transfer_function='boltzmann_camb',
            matter_power_spectrum='halofit',
            baryons_power_spectrum=None,
            mass_function=None,
            halo_concentration=None,
            emulator_neutrinos=None,
            extra_parameters=None,
            T_ncdm=DefaultParams.T_ncdm):

        # DEPRECATIONS
        warn = warnings.warn
        msg = lambda par: f"{par} is deprecated in Cosmology."  # noqa

        if Neff is None:
            warn("Neff will change from 3.046 to 3.044 in CCLv3.0.0.",
                 CCLDeprecationWarning)
            Neff = 3.046
        if mass_function is None:
            mass_function = "tinker10"
        else:
            warn(f"{msg('mass_function')} Use the halo model functionality.",
                 CCLDeprecationWarning)
        if halo_concentration is None:
            halo_concentration = "duffy2008"
        else:
            warn(f"{msg('halo_concentration')} Use the halo model "
                 "functionality.", CCLDeprecationWarning)
        if bcm_log10Mc is None:
            bcm_log10Mc = np.log10(1.2e14)
        else:
            warn(f"{msg('bcm_log10Mc')} Use the baryons functionality.",
                 CCLDeprecationWarning)
        if bcm_etab is None:
            bcm_etab = 0.5
        else:
            warn(f"{msg('bcm_etab')} Use the baryons functionality.",
                 CCLDeprecationWarning)
        if bcm_ks is None:
            bcm_ks = 55.
        else:
            warn(f"{msg('bcm_ks')} Use the baryons functionality.",
                 CCLDeprecationWarning)
        if baryons_power_spectrum is None:
            baryons_power_spectrum = "nobaryons"
        else:
            warn(f"{msg('baryons_power_spectrum')} Use the baryons "
                 "functionality.", CCLDeprecationWarning)
        if z_mg is not None or df_mg is not None:
            warn("z_mg and df_mg are deprecated from Cosmology. Custom growth "
                 "arrays can be passed to CosmologyCalculator.",
                 CCLDeprecationWarning)
        extra_parameters = extra_parameters or {}
        if emulator_neutrinos is not None:
            warn("emulator_neutrinos has been moved to extra_parameters  "
                 "and can be specified using e.g. "
                 "{'emu': {'neutrinos': 'strict'}}.", CCLDeprecationWarning)
            extra_parameters["emu"] = {"neutrinos": emulator_neutrinos}
        if "emu" not in extra_parameters:
            extra_parameters["emu"] = {"neutrinos": "strict"}

        # going to save these for later
        self._params_init_kwargs = dict(
            Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=n_s, sigma8=sigma8,
            A_s=A_s, Omega_k=Omega_k, Omega_g=Omega_g, Neff=Neff, m_nu=m_nu,
            mass_split=mass_split, w0=w0, wa=wa, T_CMB=T_CMB, T_ncdm=T_ncdm,
            bcm_log10Mc=bcm_log10Mc,
            bcm_etab=bcm_etab, bcm_ks=bcm_ks, mu_0=mu_0, sigma_0=sigma_0,
            c1_mg=c1_mg, c2_mg=c2_mg, lambda_mg=lambda_mg,
            z_mg=z_mg, df_mg=df_mg,
            extra_parameters=extra_parameters)

        self._config_init_kwargs = dict(
            transfer_function=transfer_function,
            matter_power_spectrum=matter_power_spectrum,
            baryons_power_spectrum=baryons_power_spectrum,
            mass_function=mass_function,
            halo_concentration=halo_concentration,
            extra_parameters=extra_parameters)

        self._build_cosmo()

        self._pk_lin = {}
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
            raise CCLError(f"{self.cosmo.status}: {self.cosmo.status_message}")

    def write_yaml(self, filename, *, sort_keys=False):
        """Write a YAML representation of the parameters to file.

        Args:
            filename (:obj:`str`) Filename, file pointer, or stream to write "
                "parameters to."
        """
        def make_yaml_friendly(d):
            # serialize numpy types and dicts
            for k, v in d.items():
                if isinstance(v, int):
                    d[k] = int(v)
                elif isinstance(v, float):
                    d[k] = float(v)
                elif isinstance(v, dict):
                    make_yaml_friendly(v)

        params = {**self._params_init_kwargs, **self._config_init_kwargs}
        make_yaml_friendly(params)

        if isinstance(filename, str):
            with open(filename, "w") as fp:
                return yaml.dump(params, fp, sort_keys=sort_keys)
        return yaml.dump(params, filename, sort_keys=sort_keys)

    @classmethod
    def read_yaml(cls, filename, **kwargs):
        """Read the parameters from a YAML file.

        Args:
            filename (:obj:`str`) Filename, file pointer, or stream to read
                parameters from.
            **kwargs (dict) Additional keywords that supersede file contents
        """
        loader = yaml.Loader
        if isinstance(filename, str):
            with open(filename, 'r') as fp:
                return cls(**{**yaml.load(fp, Loader=loader), **kwargs})
        return cls(**{**yaml.load(filename, Loader=loader), **kwargs})

    def _build_config(
            self, transfer_function=None, matter_power_spectrum=None,
            baryons_power_spectrum=None,
            mass_function=None, halo_concentration=None,
            extra_parameters=None):
        """Build a ccl_configuration struct.

        This function builds C ccl_configuration struct. This structure
        controls which various approximations are used for the transfer
        function, matter power spectrum, baryonic effect in the matter
        power spectrum, mass function, halo concentration relation, and
        neutrino effects in the emulator.

        It also does some error checking on the inputs to make sure they
        are valid and physically consistent.
        """
        if (matter_power_spectrum == "camb"
                and transfer_function != "boltzmann_camb"):
            raise CCLError(
                "To compute the non-linear matter power spectrum with CAMB "
                "the transfer function should be 'boltzmann_camb'.")

        config = lib.configuration()
        tf = transfer_function_types[transfer_function]
        config.transfer_function_method = tf
        mps = matter_power_spectrum_types[matter_power_spectrum]
        config.matter_power_spectrum_method = mps
        # TODO: Remove for CCLv3.
        bps = baryons_power_spectrum_types[baryons_power_spectrum]
        config.baryons_power_spectrum_method = bps
        # TODO: Remove for CCLv3.
        mf = mass_function_types[mass_function]
        config.mass_function_method = mf
        # TODO: Remove for CCLv3.
        cm = halo_concentration_types[halo_concentration]
        config.halo_concentration_method = cm
        ent = extra_parameters["emu"]["neutrinos"]
        config.emulator_neutrinos_method = emulator_neutrinos_types[ent]

        # Store ccl_configuration for later access
        self._config = config

    def _build_parameters(
            self, Omega_c=None, Omega_b=None, h=None, n_s=None, sigma8=None,
            A_s=None, Omega_k=None, Neff=None, m_nu=None, mass_split=None,
            w0=None, wa=None, T_CMB=None, T_ncdm=None,
            bcm_log10Mc=None, bcm_etab=None, bcm_ks=None,
            mu_0=None, sigma_0=None, c1_mg=None, c2_mg=None, lambda_mg=None,
            z_mg=None, df_mg=None, Omega_g=None,
            extra_parameters=None):
        """Build a ccl_parameters struct"""
        # Fill-in defaults (SWIG converts `numpy.nan` to `NAN`)
        A_s = np.nan if A_s is None else A_s
        sigma8 = np.nan if sigma8 is None else sigma8
        Omega_g = np.nan if Omega_g is None else Omega_g

        # Check to make sure Omega_k is within reasonable bounds.
        if Omega_k < -1.0135:
            raise ValueError("Omega_k must be more than -1.0135.")

        # Modified growth.
        if (z_mg is None) != (df_mg is None):
            raise ValueError("Both z_mg and df_mg must be arrays or None.")
        if z_mg is not None:
            z_mg, df_mg = map(np.atleast_1d, [z_mg, df_mg])
            if z_mg.shape != df_mg.shape:
                raise ValueError("Shape mismatch for z_mg and df_mg.")

        # Check to make sure specified amplitude parameter is consistent.
        if [A_s, sigma8].count(np.nan) != 1:
            raise ValueError("Set either A_s or sigma8 and not both.")

        # Check if any compulsory parameters are not set.
        compul = {"Omega_c": Omega_c, "Omega_b": Omega_b, "h": h, "n_s": n_s}
        for param, value in compul.items():
            if value is None:
                raise ValueError(f"Must set parameter {param}.")

        # Make sure the neutrino parameters are consistent.
        if not isinstance(m_nu, (Real, Iterable)):
            raise ValueError("m_nu must be float or sequence")

        c = const
        # Initialize curvature.
        k_sign = -np.sign(Omega_k) if np.abs(Omega_k) > 1e-6 else 0
        sqrtk = np.sqrt(np.abs(Omega_k)) * h / c.CLIGHT_HMPC

        # Initialize radiation.
        rho_g = 4 * c.STBOLTZ / c.CLIGHT**3 * T_CMB**4
        rho_crit = c.RHO_CRITICAL * c.SOLAR_MASS / c.MPC_TO_METER**3 * h**2

        # Initialize neutrinos.
        g = (4/11)**(1/3)
        T_nu = g * T_CMB
        massless_limit = T_nu * c.KBOLTZ / c.EV_IN_J

        from .neutrinos import nu_masses
        mnu_list = nu_masses(m_nu=m_nu, mass_split=mass_split)
        nu_mass = mnu_list[mnu_list > massless_limit]
        N_nu_mass = len(nu_mass)
        N_nu_rel = Neff - N_nu_mass * (T_ncdm/g)**4

        rho_nu_rel = N_nu_rel * (7/8) * 4 * c.STBOLTZ / c.CLIGHT**3 * T_nu**4
        Omega_nu_rel = rho_nu_rel / rho_crit
        Omega_nu_mass = self._OmNuh2(nu_mass, N_nu_mass, T_CMB, T_ncdm) / h**2

        if N_nu_rel < 0:
            raise ValueError("Unphysical Neff and m_nu combination results to "
                             "negative number of relativistic neutrinos.")

        # Initialize matter & dark energy.
        Omega_m = Omega_b + Omega_c + Omega_nu_mass
        Omega_l = 1 - Omega_m - rho_g/rho_crit - Omega_nu_rel - Omega_k
        if np.isnan(Omega_g):
            # No value passed for Omega_g
            Omega_g = rho_g/rho_crit
        else:
            # Omega_g was passed - modify Omega_l
            Omega_l += rho_g/rho_crit - Omega_g

        self._fill_params(
            m_nu=nu_mass, sum_nu_masses=sum(nu_mass), N_nu_mass=N_nu_mass,
            N_nu_rel=N_nu_rel, Neff=Neff, Omega_nu_mass=Omega_nu_mass,
            Omega_nu_rel=Omega_nu_rel, Omega_m=Omega_m, Omega_c=Omega_c,
            Omega_b=Omega_b, Omega_k=Omega_k, sqrtk=sqrtk, k_sign=int(k_sign),
            T_CMB=T_CMB, T_ncdm=T_ncdm, Omega_g=Omega_g, w0=w0, wa=wa,
            Omega_l=Omega_l, h=h, H0=h*100, A_s=A_s, sigma8=sigma8, n_s=n_s,
            mu_0=mu_0, sigma_0=sigma_0, c1_mg=c1_mg, c2_mg=c2_mg,
            lambda_mg=lambda_mg,
            bcm_log10Mc=bcm_log10Mc, bcm_etab=bcm_etab, bcm_ks=bcm_ks)

        # Modified growth (deprecated)
        if z_mg is not None:
            self._params.mgrowth = [z_mg, df_mg]

    def _OmNuh2(self, m_nu, N_nu_mass, T_CMB, T_ncdm):
        # Compute OmNuh2 today.
        ret, st = lib.Omeganuh2_vec(N_nu_mass, T_CMB, T_ncdm, [1], m_nu, 1, 0)
        check(st)
        return ret[0]

    def _fill_params(self, **kwargs):
        if not hasattr(self, "_params"):
            self._params = CosmologyParams()
        [setattr(self._params, par, val) for par, val in kwargs.items()]

    def __getitem__(self, key):
        if key == 'extra_parameters':
            return self._params_init_kwargs["extra_parameters"]
        return getattr(self._params, key)

    def __del__(self):
        """Free the C memory this object is managing as it is being garbage
        collected (hopefully)."""
        if hasattr(self, "cosmo"):
            lib.cosmology_free(self.cosmo)
            delattr(self, "cosmo")
        if hasattr(self, "_params"):
            lib.parameters_free(self._params)
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
        status = 0
        status = lib.cosmology_compute_growth(self.cosmo, status)
        check(status, self)

    @cache(maxsize=3)
    def _compute_linear_power(self):
        """Return the linear power spectrum."""
        self.compute_growth()

        # Populate power spectrum splines
        trf = self._config_init_kwargs['transfer_function']
        pk = None
        rescale_s8 = True
        rescale_mg = True
        if trf == 'boltzmann_class':
            pk = self.get_class_pk_lin()
        elif trf == 'boltzmann_isitgr':
            rescale_mg = False
            pk = self.get_isitgr_pk_lin()
        elif trf in ['bbks', 'eisenstein_hu', 'eisenstein_hu_nowiggles']:
            rescale_s8 = False
            rescale_mg = False
            pk = Pk2D.from_model(self, model=trf)

        # Compute the CAMB nonlin power spectrum if needed,
        # to avoid repeating the code in `compute_nonlin_power`.
        # Because CAMB power spectra come in pairs with pkl always computed,
        # we set the nonlin power spectrum first, but keep the linear via a
        # status variable to use it later if the transfer function is CAMB too.
        pkl = None
        if self._config_init_kwargs["matter_power_spectrum"] == "camb":
            if not np.isfinite(self["A_s"]):
                raise CCLError("CAMB doesn't rescale non-linear power spectra "
                               "consistently without A_s.")
            # no rescaling because A_s is necessarily provided
            rescale_mg = rescale_s8 = False
            name = "delta_matter:delta_matter"
            pkl, self._pk_nl[name] = self.get_camb_pk_lin(nonlin=True)

        if trf == "boltzmann_camb":
            pk = pkl if pkl is not None else self.get_camb_pk_lin()

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
        self._pk_lin[DEFAULT_POWER_SPECTRUM] = self._compute_linear_power()

    def _get_halo_model_nonlin_power(self):
        warnings.warn(
            "The halo model option for the internal CCL matter power "
            "spectrum is deprecated. Use the more general functionality "
            "in the `halos` module.", category=CCLDeprecationWarning)

        from . import halos as hal
        mdef = hal.MassDef('vir', 'matter')
        conc = self._config.halo_concentration_method
        mfm = self._config.mass_function_method

        if conc == lib.bhattacharya2011:
            c = hal.ConcentrationBhattacharya13(mass_def=mdef)
        elif conc == lib.duffy2008:
            c = hal.ConcentrationDuffy08(mass_def=mdef)
        elif conc == lib.constant_concentration:
            c = hal.ConcentrationConstant(c=4., mass_def=mdef)

        if mfm == lib.tinker10:
            hmf = hal.MassFuncTinker10(mass_def=mdef,
                                       mass_def_strict=False)
            hbf = hal.HaloBiasTinker10(mass_def=mdef,
                                       mass_def_strict=False)
        elif mfm == lib.shethtormen:
            hmf = hal.MassFuncSheth99(mass_def=mdef,
                                      mass_def_strict=False,
                                      use_delta_c_fit=True)
            hbf = hal.HaloBiasSheth99(mass_def=mdef,
                                      mass_def_strict=False)
        else:
            raise ValueError("Halo model spectra not available for your "
                             "current choice of mass function with the "
                             "deprecated implementation.")
        prf = hal.HaloProfileNFW(concentration=c)
        hmc = hal.HMCalculator(mass_function=hmf, halo_bias=hbf,
                               mass_def=mdef)
        return hal.halomod_Pk2D(self, hmc, prf)

    @cache(maxsize=3)
    def _compute_nonlin_power(self):
        """Return the non-linear power spectrum."""
        self.compute_distances()

        # Populate power spectrum splines
        mps = self._config_init_kwargs['matter_power_spectrum']
        # needed for halofit, halomodel and linear options
        if (mps != 'emu') and (mps is not None):
            self.compute_linear_power()

        if mps == "camb" and self.has_nonlin_power:
            # Already computed
            return self._pk_nl[DEFAULT_POWER_SPECTRUM]

        if mps == 'halo_model':
            pk = self._get_halo_model_nonlin_power()
        elif mps == 'halofit':
            pkl = self._pk_lin[DEFAULT_POWER_SPECTRUM]
            pk = pkl.apply_halofit(self)
        elif mps == 'emu':
            pk = Pk2D.from_model(self, model='emu')
        elif mps == 'linear':
            pk = self._pk_lin[DEFAULT_POWER_SPECTRUM]

        # Correct for baryons if required
        if self._config_init_kwargs['baryons_power_spectrum'] == 'bcm':
            warnings.warn("Adding baryonic effects to the non-linear matter "
                          "power spectrum automatically is deprecated in "
                          "Cosmology. Use the functionality in baryons.",
                          CCLDeprecationWarning)
            self.bcm_correct_pk2d(pk)

        return pk

    @unlock_instance(mutate=False)
    def compute_nonlin_power(self):
        """Compute the non-linear power spectrum."""
        if self.has_nonlin_power:
            return
        self._pk_nl[DEFAULT_POWER_SPECTRUM] = self._compute_nonlin_power()

    def compute_sigma(self):
        """Compute the sigma(M) spline."""
        if self.has_sigma:
            return

        pk = self.get_linear_power()
        status = 0
        status = lib.cosmology_compute_sigma(self.cosmo, pk.psp, status)
        check(status, self)

    def get_linear_power(self, name=DEFAULT_POWER_SPECTRUM):
        """Get the :class:`~pyccl.pk2d.Pk2D` object associated with
        the linear power spectrum with name `name`.

        Args:
            name (:obj:`str` or `None`): name of the power spectrum to
                return.

        Returns:
            :class:`~pyccl.pk2d.Pk2D` object containing the linear
            power spectrum with name `name`.
        """
        if name == DEFAULT_POWER_SPECTRUM:
            self.compute_linear_power()
        pk = self._pk_lin.get(name)
        if pk is None:
            raise KeyError(f"Power spectrum {name} does not exist.")
        return pk

    def get_nonlin_power(self, name=DEFAULT_POWER_SPECTRUM):
        """Get the :class:`~pyccl.pk2d.Pk2D` object associated with
        the non-linear power spectrum with name `name`.

        Args:
            name (:obj:`str` or `None`): name of the power spectrum to
                return.

        Returns:
            :class:`~pyccl.pk2d.Pk2D` object containing the non-linear
            power spectrum with name `name`.
        """
        if name == DEFAULT_POWER_SPECTRUM:
            self.compute_nonlin_power()
        pk = self._pk_nl.get(name)
        if pk is None:
            raise KeyError(f"Power spectrum {name} does not exist.")
        return pk

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
        return DEFAULT_POWER_SPECTRUM in self._pk_lin

    @property
    def has_nonlin_power(self):
        """Checks if the non-linear power spectra have been precomputed."""
        return DEFAULT_POWER_SPECTRUM in self._pk_nl

    @property
    def has_sigma(self):
        """Checks if sigma(M) is precomputed."""
        return bool(self.cosmo.computed_sigma)

    @deprecated()
    def status(self):
        """Get error status of the ccl_cosmology object.
        .. note:: The error statuses are currently under development and
                  may not be fully descriptive.
        Returns:
            :obj:`str` containing the status message.
        """
        # Get status ID string if one exists
        if self.cosmo.status in CLevelErrors.keys():
            status = CLevelErrors[self.cosmo.status]
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
            f"You cannot change the ΛCDM parameters: {list(p.keys())}.")
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
        m_nu (:obj:`float` or array_like, optional):
            Mass in eV of the massive neutrinos present. Defaults to 0.
            If a sequence is passed, it is assumed that the elements of the
            sequence represent the individual neutrino masses.
        mass_split (:obj:`str`, optional): Type of massive neutrinos. Should
            be one of 'single', 'equal', 'normal', 'inverted'. 'single' treats
            the mass as being held by one massive neutrino. The other options
            split the mass into 3 massive neutrinos. Ignored if a sequence is
            passed in m_nu. Default is 'normal'.
        w0 (:obj:`float`, optional): First order term of dark energy
            equation of state. Defaults to -1.
        wa (:obj:`float`, optional): Second order term of dark energy
            equation of state. Defaults to 0.
        T_CMB (:obj:`float`): The CMB temperature today. The default is the
            same as in the Cosmology base class.
        mu_0 (:obj:`float`, optional): One of the parameters of the mu-Sigma
            modified gravity model. Defaults to 0.0
        sigma_0 (:obj:`float`, optional): One of the parameters of the mu-Sigma
            modified gravity model. Defaults to 0.0
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
        T_ncdm (:obj:`float`): Non-CDM temperature in units of photon
            temperature. The default is the same as in the base class
    """
    # TODO: Docstring - Move T_ncdm after T_CMB for CCLv3.
    __eq_attrs__ = ("_params_init_kwargs", "_config_init_kwargs",
                    "_accuracy_params", "_input_arrays",)

    @warn_api(pairs=[("m_nu_type", "mass_split")])
    def __init__(
            self, *, Omega_c=None, Omega_b=None, h=None, n_s=None,
            sigma8=None, A_s=None, Omega_k=0., Omega_g=None,
            Neff=None, m_nu=0., mass_split="normal", w0=-1., wa=0.,
            T_CMB=DefaultParams.T_CMB, mu_0=0., sigma_0=0.,
            background=None, growth=None,
            pk_linear=None, pk_nonlin=None, nonlinear_model=None,
            T_ncdm=DefaultParams.T_ncdm):

        super().__init__(
            Omega_c=Omega_c, Omega_b=Omega_b, h=h, n_s=n_s, sigma8=sigma8,
            A_s=A_s, Omega_k=Omega_k, Omega_g=Omega_g, Neff=Neff, m_nu=m_nu,
            mass_split=mass_split, w0=w0, wa=wa, T_CMB=T_CMB, T_ncdm=T_ncdm,
            mu_0=mu_0, sigma_0=sigma_0,
            transfer_function="calculator", matter_power_spectrum="calculator")

        self._input_arrays = {"background": background, "growth": growth,
                              "pk_linear": pk_linear, "pk_nonlin": pk_nonlin,
                              "nonlinear_model": nonlinear_model}

        if background is not None:
            self._init_background(background)
        if growth is not None:
            self._init_growth(growth)
        if pk_linear is not None:
            self._init_pk_linear(pk_linear)
        if pk_nonlin is not None:
            self._init_pk_nonlinear(pk_nonlin, nonlinear_model)
        self._apply_nonlinear_model(nonlinear_model)

    def _check_scale_factor(self, a):
        if not (np.diff(a) > 0).all():
            raise ValueError("Scale factor not monotonically increasing.")
        if np.abs(a[-1] - 1) > 1e-5:
            raise ValueError("Scale factor should end at 1.")

    def _check_input(self, a, arr1, arr2):
        self._check_scale_factor(a)
        if not a.shape == arr1.shape == arr2.shape:
            raise ValueError("Shape mismatch of input arrays.")

    def _check_label(self, name):
        if len(name.split(":")) != 2:
            raise ValueError(f"Could not parse power spectrum {name}. "
                             "Label must be of the form 'q1:q2'.")

    def _init_background(self, background):
        a, chi, E = background["a"], background["chi"], background["h_over_h0"]
        self._check_input(a, chi, E)
        status = 0
        status = lib.cosmology_distances_from_input(self.cosmo, a, chi, E,
                                                    status)
        check(status, self)

    def _init_growth(self, growth):
        a, gz, fz = growth["a"], growth["growth_factor"], growth["growth_rate"]
        self._check_input(a, gz, fz)
        status = 0
        status = lib.cosmology_growth_from_input(self.cosmo, a, gz, fz, status)
        check(status, self)

    def _init_pk_linear(self, pk_linear):
        a, lk = pk_linear["a"], np.log(pk_linear["k"])
        self._check_scale_factor(a)
        self.compute_growth()  # needed for high-z extrapolation
        na, nk = a.size, lk.size

        if DEFAULT_POWER_SPECTRUM not in pk_linear:
            raise ValueError("pk_linear does not contain "
                             f"{DEFAULT_POWER_SPECTRUM}")

        pk_names = set(pk_linear.keys()) - set(["a", "k"])
        for name in pk_names:
            self._check_label(name)
            pk = pk_linear[name]
            if pk.shape != (na, nk):
                raise ValueError("Power spectrum shape mismatch. "
                                 f"Expected {(na, nk)}. Received {pk.shape}.")
            # Spline in log-space if the P(k) is positive-definite
            use_log = (pk > 0).all()
            if use_log:
                pk = np.log(pk)
            self._pk_lin[name] = Pk2D(a_arr=a, lk_arr=lk, pk_arr=pk,
                                      is_logp=use_log)

    def _init_pk_nonlinear(self, pk_nonlin, nonlinear_model):
        a, lk = pk_nonlin["a"], np.log(pk_nonlin["k"])
        self._check_scale_factor(a)
        na, nk = a.size, lk.size

        if DEFAULT_POWER_SPECTRUM not in pk_nonlin and nonlinear_model is None:
            raise ValueError(f"{DEFAULT_POWER_SPECTRUM} not specified in "
                             "`pk_nonlin` and `nonlinear_model` is None")

        pk_names = set(pk_nonlin.keys()) - set(["a", "k"])
        for name in pk_names:
            self._check_label(name)
            pk = pk_nonlin[name]
            if pk.shape != (na, nk):
                raise ValueError("Power spectrum shape mismatch. "
                                 f"Expected {(na, nk)}. Received {pk.shape}.")
            # Spline in log-space if the P(k) is positive-definite
            use_log = (pk > 0).all()
            if use_log:
                pk = np.log(pk)
            self._pk_nl[name] = Pk2D(a_arr=a, lk_arr=lk, pk_arr=pk,
                                     is_logp=use_log)

    def _apply_nonlinear_model(self, nonlin_model):
        if nonlin_model is None:
            return

        if not isinstance(nonlin_model, (str, dict)):
            raise ValueError("`nonlin_model` must be str, dict, or None.")
        if isinstance(nonlin_model, str) and not self.has_linear_power:
            raise ValueError("Linear power spectrum is empty.")

        if isinstance(nonlin_model, str):
            nonlin_model = {name: nonlin_model for name in self._pk_lin}

        for name, model in nonlin_model.items():
            if name in self._pk_nl:
                continue

            if name not in self._pk_lin:
                raise KeyError(f"Linear power spectrum {name} does not exist.")
            if model not in [None, "halofit"]:
                raise KeyError(f"{model} is not a valid non-linear model.")
            if name == DEFAULT_POWER_SPECTRUM and model is None:
                raise ValueError("The non-linear matter power spectrum must "
                                 "not be None.")

            if model == 'halofit':
                pkl = self._pk_lin[name]
                self._pk_nl[name] = pkl.apply_halofit(self)
