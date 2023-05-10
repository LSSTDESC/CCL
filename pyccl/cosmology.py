"""
==================================
Cosmology (:mod:`pyccl.cosmology`)
==================================

Classes that store cosmological parameters. Main functionality of CCL.

.. note::

    All of the standalone functions in other modules, which take `cosmo` as
    their first argument, are methods of :class:`~Cosmology`.
"""

__all__ = ("TransferFunctions", "MatterPowerSpectra",
           "Cosmology", "CosmologyVanillaLCDM", "CosmologyCalculator",)

import warnings
import yaml
from enum import Enum
from inspect import getmembers, isfunction, signature
from numbers import Real
from typing import Dict, Iterable, Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from . import (
    CCLError, CCLDeprecationWarning, CCLObject, CLevelErrors, CosmologyParams,
    DEFAULT_POWER_SPECTRUM, Pk2D, cache, gsl_params, lib, spline_params,
    warn_api, deprecated)
from . import physical_constants as const
from .pyutils import check as check_


class TransferFunctions(Enum):
    """Available choices for the computation of the linear power spectrum."""
    BBKS = "bbks"
    """Fitting formula of :footcite:t:`Bardeen86` (BBKS).

    .. footbibliography::
    """

    EISENSTEIN_HU = "eisenstein_hu"
    """Model of :footcite:t:`Eisenstein99` (with wiggles).

    .. footbibliography::
    """

    EISENSTEIN_HU_NOWIGGLES = "eisenstein_hu_nowiggles"
    """Model of :footcite:t:`Eisenstein99` (without wiggles).

    .. footbibliography::
    """

    BOLTZMANN_CLASS = "boltzmann_class"
    """``CLASS`` :footcite:p:`Blas11` Boltzmann solver .

    .. footbibliography::
    """

    BOLTZMANN_CAMB = "boltzmann_camb"
    """``CAMB`` :footcite:p:`LewisCAMB` Boltzmann solver.

    .. footbibliography::
    """

    BOLTZMANN_ISITGR = "boltzmann_isitgr"
    """``ISiTGR`` :footcite:p:`Garcia19` modified gravity solver.

    .. footbibliography::
    """

    CALCULATOR = "calculator"
    """:class:`~CosmologyCalculator` with input power spectra."""


class MatterPowerSpectra(Enum):
    """Available choices for the computation of the non-linear power spectrum.
    """
    LINEAR = "linear"
    """Linear power spectrum."""

    HALOFIT = "halofit"
    """``HALOFIT`` transformation of :footcite:t:`Takahashi12`.

    .. footbibliography::
    """

    HALOMODEL = "halomodel"
    """Halo model.

    .. deprecated:: 2.1.0

        This option is deprecated and will be removed in the next major
        release.
    """

    EMU = "emu"
    """``Cosmic Emu`` :footcite:p:`Lawrence17` matter power spectrum emulator.

    .. footbibliography::
    """

    CAMB = "camb"
    """``CAMB`` :footcite:p:`LewisCAMB`. Supports ``HALOFIT``
    :footcite:p:`Takahashi12` and ``HMCode-2020`` :footcite:p:`Mead21`.

    .. footbibliography::
    """

    CALCULATOR = "calculator"
    """:class:`~CosmologyCalculator` with input power spectra."""


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
    """Stores the cosmological parameters, and associated data.

    A list of the cosmological parameters stored in new instances of this class
    lives in :class:`~pyccl.base.parameters.cosmology_params.CosmologyParams`
    Note that only a subset of these parameters exist in the signature of
    :class:`~Cosmology`. The rest are calculated during initialization.

    Parameters may be looked up by name (e.g. ``cosmo["sigma8"]``).

    .. note::

        Setting `Omega_g` to zero, yields a physically inconsistent model
        (since `T_CMB` is non-zero). However, this approximation is common for
        late-time LSS computations.

    Parameters
    ----------
    transfer_function
        Transfer function. Available choices in :class:`~TransferFunctions`.
    matter_power_spectrum
        Matter power spectrum. Available choices in
        :class:`~MatterPowerSpectra`.
    baryons_power_spectrum
        Baryonic feedback correction to the matter power spectrum.

        .. deprecated:: 2.8.0

            Use the :mod:`~pyccl.baryons` functionality.

    mass_function
        Halo mass function for the halo model power spectrum.

        .. deprecated:: 2.8.0

            Use the :mod:`~pyccl.halos` functionality.

    halo_concentration
        Halo concentration-mass relation.

        .. deprecated:: 2.8.0

            Use the :mod:`~pyccl.halos` functionality.

    emulator_neutrinos : {'strict', 'equalize'}
        ``CosmicEmu`` behavior for unequal neutrino masses. `'strict'` raises
        an exception if the resulting neutrino masses are unequal. `'equalize'`
        redistributes the masses to force them equal, but may lead to internal
        inconsistencies.

        .. deprecated:: 2.8.0

            Moved to `extra_parameters`. Specified as e.g.
            ``extra_parameters={"emu": {"neutrinos": "strict"}}``.

    extra_parameters
        Model-specific parameters. The key is the name of the model and the
        value is a (nested) dictionary of parameters and their values
        (e.g. ``extra_parameters={"camb": {"kmax": 5, "lmax": 1000}}``).


    Models
    ------
    Supported models in `extra_parameters` are:

        * ``'camb'`` - Options listed in :func:`~get_camb_pk_lin`.
        * ``'emu'`` - Options listed in
          :py:data:`~pyccl.cosmology.emulator_neutrinos_types`.
    """
    # TODO: Docstring - Move T_ncdm after T_CMB for CCLv3.
    from .base.repr_ import build_string_Cosmology as __repr__
    __eq_attrs__ = ("_params_init_kwargs", "_config_init_kwargs",
                    "_gsl_params", "_spline_params",)
    cosmo: lib.cosmology
    """The associated C-level cosmology struct."""

    @warn_api(pairs=[("m_nu_type", "mass_split")])
    def __init__(
            self,
            *,
            Omega_c: Real,
            Omega_b: Real,
            h: Real,
            n_s: Real,
            sigma8: Optional[Real] = None,
            A_s: Optional[Real] = None,
            Omega_k: Real = 0,
            Omega_g: Optional[Real] = None,
            Neff: Real = None,  # TODO: Default value from CosmologyParams (v3)
            m_nu: Union[Real, Sequence[Real]] = 0,
            mass_split: str = 'normal',
            w0: Real = -1,
            wa: Real = 0,
            T_CMB: Real = CosmologyParams.T_CMB,
            bcm_log10Mc: Optional[Real] = None,
            bcm_etab: Optional[Real] = None,
            bcm_ks: Optional[Real] = None,
            mu_0: Real = 0,
            sigma_0: Real = 0,
            c1_mg: Real = 1,
            c2_mg: Real = 1,
            lambda_mg: Real = 0,
            z_mg: Optional[NDArray[Real]] = None,  # TODO: deprecate in v3
            df_mg: Optional[NDArray[Real]] = None,  # TODO: deprecate in v3
            transfer_function: str = 'boltzmann_camb',
            matter_power_spectrum: str = 'halofit',
            baryons_power_spectrum: Optional[str] = None,  # TODO: depr in v3
            mass_function: Optional[str] = None,  # TODO: depr in v3
            halo_concentration: Optional[str] = None,  # TODO: depr in v3
            emulator_neutrinos: Optional[str] = None,  # TODO: depr in v3
            extra_parameters: Optional[dict] = None,
            T_ncdm: Real = CosmologyParams.T_ncdm  # TODO: v3 after T_CMB
    ):
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

    @property
    def has_distances(self) -> bool:
        """Check whether the distance splines exist."""
        return bool(self.cosmo.computed_distances)

    @property
    def has_growth(self) -> bool:
        """Check whether the growth splines exist."""
        return bool(self.cosmo.computed_growth)

    @property
    def has_linear_power(self) -> bool:
        """Check whether the linear power spectrum exists."""
        return DEFAULT_POWER_SPECTRUM in self._pk_lin

    @property
    def has_nonlin_power(self) -> bool:
        """Check whether the non-linear power spectrum exists."""
        return DEFAULT_POWER_SPECTRUM in self._pk_nl

    @property
    def has_sigma(self) -> bool:
        r"""Check whether the :math:`\sigma(M)` splines exist."""
        return bool(self.cosmo.computed_sigma)

    def _build_cosmo(self):
        """Assemble all of the input data into a valid ccl_cosmology object."""
        # We have to make all of the C stuff that goes into a cosmology
        # and then we make the cosmology.
        self._build_parameters(**self._params_init_kwargs)
        self._build_config(**self._config_init_kwargs)
        self.cosmo = lib.cosmology_create(self._params._instance, self._config)
        self._gsl_params = gsl_params.copy()
        self._spline_params = spline_params.copy()

        if self.cosmo.status != 0:
            raise CCLError(f"{self.cosmo.status}: {self.cosmo.status_message}")

    def _pretty_print(self):
        """Pretty print for `yaml` export and `repr`."""
        def make_pretty(d):
            # serialize numpy types and dicts
            for k, v in d.items():
                if isinstance(v, int):
                    d[k] = int(v)
                elif isinstance(v, float):
                    d[k] = float(v)
                elif isinstance(v, dict):
                    make_pretty(v)

        params = {**self._params_init_kwargs, **self._config_init_kwargs}
        make_pretty(params)
        return params

    def write_yaml(self, filename: str, *, sort_keys: bool = False) -> None:
        """Write a YAML representation of the parameters to file.

        Arguments
        ---------
        filename
            Filename, file pointer, or stream to write parameters to.
        """
        params = self._pretty_print()

        if isinstance(filename, str):
            with open(filename, "w") as fp:
                return yaml.dump(params, fp, sort_keys=sort_keys)
        return yaml.dump(params, filename, sort_keys=sort_keys)

    @classmethod
    def read_yaml(cls, filename: str, **kwargs):
        """Read the parameters from a YAML file.

        Arguments
        ---------
        filename
            Filename, file pointer, or stream to read parameters from.
        **kwargs
            Additional keywords that supersede file contents
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
        """Build a ccl_configuration struct."""
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
        Omega_nu_mass = self._OmNuh2(nu_mass, T_CMB, T_ncdm) / h**2

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

        self._params = CosmologyParams(
            Omega_k=Omega_k, k_sign=int(k_sign), sqrtk=sqrtk,
            n_s=n_s, sigma8=sigma8, A_s=A_s,
            N_nu_mass=N_nu_mass, N_nu_rel=N_nu_rel, Neff=Neff,
            Omega_nu_mass=Omega_nu_mass, Omega_nu_rel=Omega_nu_rel,
            sum_nu_masses=sum(nu_mass), m_nu=nu_mass, mass_split=mass_split,
            T_nu=T_nu, Omega_g=Omega_g, T_CMB=T_CMB, T_ncdm=T_ncdm,
            Omega_c=Omega_c, Omega_b=Omega_b, Omega_m=Omega_m,
            h=h, H0=h*100,
            Omega_l=Omega_l, w0=w0, wa=wa,
            mu_0=mu_0, sigma_0=sigma_0,
            c1_mg=c1_mg, c2_mg=c2_mg, lambda_mg=lambda_mg,
            bcm_log10Mc=bcm_log10Mc, bcm_etab=bcm_etab, bcm_ks=bcm_ks)

        # Modified growth (deprecated)
        if z_mg is not None:
            self._params.mgrowth = [z_mg, df_mg]

    def _OmNuh2(self, m_nu, T_CMB, T_ncdm):
        # Compute OmNuh2 today.
        ret, st = lib.Omeganuh2_vec(len(m_nu), T_CMB, T_ncdm, [1], m_nu, 1, 0)
        check_(st)
        return ret[0]

    def __getitem__(self, key):
        if key == 'extra_parameters':
            return self._params_init_kwargs["extra_parameters"]
        return getattr(self._params, key)

    def __del__(self):
        if hasattr(self, "cosmo"):
            lib.cosmology_free(self.cosmo)
            delattr(self, "cosmo")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.__del__()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove C data.
        state.pop('cosmo', None)
        state.pop('_params', None)
        state.pop('_config', None)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._build_cosmo()  # rebuild C data from `state`

    def compute_distances(self) -> None:
        """Compute the distance splines."""
        if self.has_distances:
            return
        status = 0
        status = lib.cosmology_compute_distances(self.cosmo, status)
        self.check(status)

    def compute_growth(self) -> None:
        """Compute the growth splines."""
        if self.has_growth:
            return
        status = 0
        status = lib.cosmology_compute_growth(self.cosmo, status)
        self.check(status)

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
            name = DEFAULT_POWER_SPECTRUM
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
            self.check(status)

        return pk

    def compute_linear_power(self) -> None:
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
        return hal.halomod_Pk2D(self, hmc, prf, normprof1=True)

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

    def compute_nonlin_power(self) -> None:
        """Compute the non-linear power spectrum."""
        if self.has_nonlin_power:
            return
        self._pk_nl[DEFAULT_POWER_SPECTRUM] = self._compute_nonlin_power()

    def compute_sigma(self) -> None:
        r"""Compute the :math:`\sigma(M)` spline."""
        if self.has_sigma:
            return

        pk = self.get_linear_power()
        status = 0
        status = lib.cosmology_compute_sigma(self.cosmo, pk.psp, status)
        self.check(status)

    def get_linear_power(self, name: str = DEFAULT_POWER_SPECTRUM) -> Pk2D:
        """Get the linear power spectrum. (Compute if necessary.)

        Arguments
        ---------
        name
            Name of the power spectrum.

        Returns
            Linear power spectrum.
        """
        if name == DEFAULT_POWER_SPECTRUM:
            self.compute_linear_power()
        pk = self._pk_lin.get(name)
        if pk is None:
            raise KeyError(f"Power spectrum {name} does not exist.")
        return pk

    def get_nonlin_power(self, name: str = DEFAULT_POWER_SPECTRUM) -> Pk2D:
        """Get the non-linear power spectrum. (Compute if necessary.)

        Arguments
        ---------
        name
            Name of the power spectrum.

        Returns
            Non-linear power spectrum.
        """
        if name == DEFAULT_POWER_SPECTRUM:
            self.compute_nonlin_power()
        pk = self._pk_nl.get(name)
        if pk is None:
            raise KeyError(f"Power spectrum {name} does not exist.")
        return pk

    def check(self, status: int) -> None:
        """Check the status returned by a :mod:`~pyccl.ccllib` function.

        Arguments
        ---------
        status
            Error type flag. The dictionary mapping is in
            :py:data:`~pyccl.pyutils.CLevelErrors`.
        """
        return check_(status=status, cosmo=self)

    @deprecated
    def status(self) -> str:
        """Error status of the ccl_cosmology object.

        Returns
        -------
            Status message.
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


def CosmologyVanillaLCDM(**kwargs) -> Cosmology:
    r"""Create a cosmology with typical flat :math:`\rm \Lambda CDM` parameters
    (``Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81``) -
    and with no massive neutrinos.

    Arguments
    ---------
    **kwargs
        Additional cosmological parameters to pass to :class:`Cosmology`. The
        :math:`\rm \Lambda CDM` parameters, and `A_s` cannot be overridden.

    Returns
    -------

        Typical :math:`\rm \Lambda CDM` cosmology.

    Raises
    ------
    ValuError
        Trying to override the :math:`\rm \Lambda CDM` parameters, or `A_s`.
    """
    p = {'Omega_c': 0.25, 'Omega_b': 0.05, 'h': 0.67, 'n_s': 0.96,
         'sigma8': 0.81, 'A_s': None}
    if set(p).intersection(set(kwargs)):
        raise ValueError(
            f"You cannot change the ΛCDM parameters: {list(p.keys())}.")
    # TODO py39+: dictionary union operator `(p | kwargs)`.
    return Cosmology(**{**p, **kwargs})


class CosmologyCalculator(Cosmology):
    r"""Cosmology calculator mode.

    Construct a cosmology from arrays describing

    * background expansion
      (override :meth:`~Cosmology.compute_distances`),
    * growth
      (override :meth:`~Cosmology.compute_growth`),
    * linear power spectra
      (override :meth:`~Cosmology.compute_linear_power`),
    * non-linear power spectra
      (override :meth:`~Cosmology.compute_nonlin_power`).


    While the input arrays are generally adequate for computing the majority of
    observables, a basic set of :math:`\rm \Lambda CDM` parameters is required
    for some calculations implemented in CCL (e.g. halo mass function).

    :class:`~CosmologyCalculator` accepts a subset of the cosmological
    parameters in the signature of :class:`~Cosmology`. Additional parameters
    are described below.

    Parameters
    ----------
    background
        Background expansion. Dictionary entries:

            * ``'a'`` - Monotonically increasing array of scale factor,
            * ``'chi'`` - Comoving radial distance :math:`\chi(a)`
              (in :math:`\rm Mpc`),
            * ``'h_over_h0'`` - :math:`E(a) := \frac{H(a)}{H(a=1)}`.

    growth
        Linear growth of matter fluctuations. Dictionary entries:

            * ``'a'`` - Monotonically increasing array of scale factor,
            * ``'growth_factor'`` - Linear growth factor, :math:`D(a)`,
            * ``'growth_rate'`` - Growth rate,
              :math:`f(a) := \frac{{\rm d}\log D(a)}{{\rm d}\log a}`.

    pk_linear
        Linear power spectra. Dictionary entries:

            * ``'a'`` - Monotonically increasing array of scale factor,
            * ``'k'`` - Comoving wavenumber (in :math:`\rm Mpc^{-1}`),
            * :py:data:`~pyccl.base.parameters.cosmology_params.\
              DEFAULT_POWER_SPECTRUM` - Array-like `(na, nk)` with the linear
              matter power spectrum, :math:`P_{\rm L}(k, a)` (in
              :math:`\rm Mpc^3`).
            * ``q1:q2`` - Arrays of additional cross-power spectra between
              quantities `q1` and `q2`.

    pk_nonlin
        Non-linear power spectra. Dictionary entries:

            * ``'a'`` - Monotonically increasing array of scale factor,
            * ``'k'`` - Comoving wavenumber (in :math:`\rm Mpc^{-1}`),
            * :py:data:`~pyccl.base.parameters.cosmology_params.\
              DEFAULT_POWER_SPECTRUM` - If `nonlinear_model` is not specified,
              array-like `(na, nk)` with the non-linear matter power spectrum,
              :math:`P_{\rm NL}(k, a)` (in :math:`\rm Mpc^3`).
            * ``q1:q2`` - Arrays of additional cross-power spectra between
              quantities `q1` and `q2`.

    nonlinear_model
        Model to compute non-linear power spectra:

            * `str` - Apply the non-linear model to all spectra in
              `pk_linear` that are also not in `pk_nonlin`,
            * `dict` - Dictionary keys of the power spectrum entries in
              `pk_linear`. Dictionary values of the non-linear model to apply
              to each linear power spectrum. If None, the non-linear power
              spectrum is not calculated. Available models are the
              `'transformations'` listed in :class:`~MatterPowerSpectra`.
    """
    # TODO: Docstring - Move T_ncdm after T_CMB for CCLv3.
    __eq_attrs__ = ("_params_init_kwargs", "_config_init_kwargs",
                    "_gsl_params", "_spline_params", "_input_arrays",)

    @warn_api(pairs=[("m_nu_type", "mass_split")])
    def __init__(
            self,
            *,
            Omega_c: Real,
            Omega_b: Real,
            h: Real,
            n_s: Real,
            sigma8: Optional[Real] = None,
            A_s: Optional[Real] = None,
            Omega_k: Real = 0,
            Omega_g: Optional[Real] = None,
            Neff: Real = None,  # TODO: Default from CosmologyParams in CCLv3.
            m_nu: Union[Real, Sequence[Real]] = 0,
            mass_split: str = "normal",
            w0: Real = -1,
            wa: Real = 0,
            T_CMB: Real = CosmologyParams.T_CMB,
            mu_0: Real = 0,
            sigma_0: Real = 0,
            background: Optional[
                Dict[
                    Literal["a", "chi", "h_over_h0"],
                    NDArray[Real]]
            ] = None,
            growth: Optional[
                Dict[
                    Literal["a", "growth_factor", "growth_rate"],
                    NDArray[Real]]
            ] = None,
            pk_linear: Optional[
                Dict[
                    Union[
                        Literal["a", "k", DEFAULT_POWER_SPECTRUM],
                        str],
                    NDArray[Real]]
            ] = None,
            pk_nonlin: Optional[
                Dict[
                    Union[
                        Literal["a", "k"],
                        str],
                    NDArray[Real]]
            ] = None,
            nonlinear_model: Optional[
                Union[str, Dict[str, Union[str, None]]]
            ] = None,
            T_ncdm: Real = CosmologyParams.T_ncdm  # TODO: v3 after T_CMB
    ):

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
        self.check(status)

    def _init_growth(self, growth):
        a, gz, fz = growth["a"], growth["growth_factor"], growth["growth_rate"]
        self._check_input(a, gz, fz)
        status = 0
        status = lib.cosmology_growth_from_input(self.cosmo, a, gz, fz, status)
        self.check(status)

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
