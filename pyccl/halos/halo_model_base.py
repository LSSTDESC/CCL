from .. import ccllib as lib
from ..core import check
from ..parameters import physical_constants as const
from ..base import (CCLAutoRepr, CCLNamedClass,
                    warn_api, deprecated, deprecate_attr)
import numpy as np
import functools
from abc import abstractmethod, abstractproperty


__all__ = ("HMIngredients",)


class HMIngredients(CCLAutoRepr, CCLNamedClass):
    """Base class for halo model ingredients.

    This class contains methods that check for consistency of mass definition
    and model at initialization. Subclasses must define a
    ``_mass_def_strict_always`` class attribute and a
    ``_check_mass_def_strict`` class method.

    Parameters
    ----------
    mass_def : :obj:`~pyccl.halos.MassDef`
        Mass definition.
    mass_def_strict : bool
        Whether the instance will allow for incompatible mass definitions.

    Raises
    ------
    ValueError
        If the mass definition is incompatible with the model setup.

    Attributes
    ----------
    mass_def
    mass_def_strict
    """
    __repr_attrs__ = ("mass_def", "mass_def_strict",)
    __getattr__ = deprecate_attr(pairs=[('mdef', 'mass_def')]
                                 )(super.__getattribute__)

    @warn_api
    def __init__(self, *, mass_def, mass_def_strict=True):
        # Check mass definition consistency.
        from .massdef import MassDef
        mass_def = MassDef.initialize_from_input(mass_def)
        self.mass_def_strict = mass_def_strict
        self._check_mass_def(mass_def)
        self.mass_def = mass_def
        self._setup()

    @abstractproperty
    def _mass_def_strict_always(self) -> bool:
        """Property that dictates whether ``mass_def_strict`` can be set
        as False on initialization.

        Some models are set up in a way so that the set of fitted parameters
        depends on the mass definition, (i.e. no one universal model exists
        to cover all cases). Setting this to True fixes ``mass_def_strict``
        to True irrespective of what the user passes.

        Set this propery to False to allow users to override strict checks.
        """

    @abstractmethod
    def _check_mass_def_strict(self, mass_def) -> bool:
        """Check if a mass definition is compatible with the model.

        Arguments
        ---------
        mass_def : :class:`~pyccl.halos.MassDef`
            Mass definition to check for compatibility.

        Returns
        -------
        compatibility : bool
            Flag denoting whether the mass definition is compatible.
        """

    def _setup(self) -> None:
        """Initialize any internal attributes of this object.
        This function is the equivalent of a post-init method, called at
        the end of initialization.
        """

    def _check_mass_def(self, mass_def) -> None:
        """Check if a mass definition is compatible with the model and the
        initialization parameters.

        Arguments
        ---------
        mass_def : :class:`~pyccl.halos.MassDef`
            Mass definition to check for compatibility.

        Raises
        ------
        ValueError
            If the mass definition is incompatible with the model setup.
        """
        classname = self.__class__.__name__
        msg = f"{classname} is not defined for {mass_def.name} masses"

        if self._check_mass_def_strict(mass_def):
            # Passed mass is incompatible with model.

            if self._mass_def_strict_always:
                # Class has no universal model and mass is incompatible.
                raise ValueError(
                    f"{msg} and this requirement cannot be relaxed.")

            if self.mass_def_strict:
                # Strict mass_def check enabled and mass is incompatible.
                raise ValueError(
                    f"{msg}. To relax this check set `mass_def_strict=False`.")

    def _get_logM_sigM(self, cosmo, M, a, *, return_dlns=False):
        """Compute ``logM``, ``sigM``, and (optionally) ``dlns_dlogM``."""
        cosmo.compute_sigma()  # initialize sigma(M) splines if needed
        logM = np.log10(M)

        # sigma(M)
        status = 0
        sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM, len(logM), status)
        check(status, cosmo=cosmo)
        if not return_dlns:
            return logM, sigM

        # dlogsigma(M)/dlog10(M)
        dlns_dlogM, status = lib.dlnsigM_dlogM_vec(cosmo.cosmo, a, logM,
                                                   len(logM), status)
        check(status, cosmo=cosmo)
        return logM, sigM, dlns_dlogM


class MassFunc(HMIngredients):
    r"""Base class for halo mass functions.

    Implementation
    --------------
    - Subclasses must include arguments ``mass_def`` and ``mass_def_strict``
      in their :meth:`__init__()` methods.
    - Subclasses must define a :meth:`_get_fsigma()` method which implements
      the mass function.
    - Subclasses must define a :meth:`_check_mass_def_strict()`` method which
      flags if the input mass definition is incompatible with the model.
    - :meth:`_setup()` may be used to set up the model post-init.
      See :meth:`HMIngredients._setup()` for details.
    - Boolean class attribute ``_mass_def_strict_always`` may be set to prevent
      relaxing the mass definition consistency checks. Do not omit argument
      ``mass_def_strict`` in :meth:`__init__()` as this will depart from the
      uniform way to instantiate mass functions.

    Theory
    ------
    We assume that all mass functions can be written as

    .. math::

        \frac{{\rm d}n}{{\rm d}\log_{10}M} = f(\sigma_M) \, \frac{\rho_M}{M} \,
        \frac{{\rm d}\log \sigma_M}{{\rm d}\log_{10} M}

    where :math:`\sigma_M^2` is the overdensity variance on spheres with a
    radius given by the Lagrangian radius for mass :math:`M`.
    """
    _mass_def_strict_always = False

    @abstractmethod
    def _get_fsigma(self, cosmo, sigM, a, lnM):
        r"""Compute :math:`f(\sigma_M)`.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        sigM : (nM,) ``numpy.ndarray``
            Standard deviation in the overdensity field on the scale of
            halos of mass :math:`M`. Input will always be an array, and there
            is no need for the implemented function to convert it to one.
        a : float
            Scale factor.
        lnM : (nM,) ``numpy.ndarray``
            Natural logarithm of the halo mass in units of :math:`\rm M_\odot`
            (provided in addition to sigM for convenience in some mass function
            parametrizations). Input will always be an array, and there is no
            need for the implemented function to convert it to one.

        Returns
        -------
        f_sigma : (nM,) ``numpy.ndarray``
            Values of :math:`f(\sigma_M)`. Output is expected to be an array
            and there is no need to squeeze extra dimensions.
        """

    def __call__(self, cosmo, M, a):
        r"""Call the mass function.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : float or (nM,) array_like
            Halo mass in units of :math:`\rm M_\odot`.
        a : float
            Scale factor.

        Returns
        -------
        mass_function : float or (nM,) ``numpy.ndarray``
            Mass function :math:`\frac{{\rm d}n}{{\rm d}\log_{10}M}`
            in units of comoving :math:`\rm Mpc^{-3}`.
        """
        M_use = np.atleast_1d(M)
        logM, sigM, dlns_dlogM = self._get_logM_sigM(
            cosmo, M_use, a, return_dlns=True)

        rho = (const.RHO_CRITICAL * cosmo['Omega_m'] * cosmo['h']**2)
        f = self._get_fsigma(cosmo, sigM, a, 2.302585092994046 * logM)
        mf = f * rho * dlns_dlogM / M_use
        if np.ndim(M) == 0:
            return mf[0]
        return mf

    get_mass_function = __call__


class HaloBias(HMIngredients):
    r"""Base class for halo bias functions.

    Implementation
    --------------
    - Subclasses must include arguments ``mass_def`` and ``mass_def_strict``
      in their :meth:`__init__()` methods.
    - Subclasses must define a :meth:`_get_bsigma()` method which implements
      the halo bias.
    - Subclasses must define a :meth:`_check_mass_def_strict()`` method which
      flags if the input mass definition is incompatible with the model.
    - :meth:`_setup()` may be used to set up the model post-init.
      See :meth:`HMIngredients._setup()` for details.
    - Boolean class attribute ``_mass_def_strict_always`` may be set to prevent
      relaxing the mass definition consistency checks. Do not omit argument
      ``mass_def_strict`` in :meth:`__init__()` as this will depart from the
      uniform way to instantiate mass functions.

    Theory
    ------
    We assume that all halo bias parametrizations can be written as functions
    that depend only on :math:`M` through :math:`\sigma_M`, the overdensity
    variance on spheres of the Lagrangian radius that corresponds to mass
    :math:`M`.
    """
    _mass_def_strict_always = False

    @abstractmethod
    def _get_bsigma(self, cosmo, sigM, a):
        r"""Compute :math:`b(\sigma_M)`.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        sigM : (nM,) ``numpy.ndarray``
            Standard deviation in the overdensity field on the scale of
            halos of mass :math:`M`. Input will always be an array, and there
            is no need for the implemented function to convert it to one.
        a : float
            Scale factor.

        Returns
        -------
        b_sigma : (nM,) ``numpy.ndarray``
            Values of :math:`b(\sigma_M)`. Output is expected to be an array
            and there is no need to squeeze extra dimensions.
        """

    def __call__(self, cosmo, M, a):
        r"""Call the halo bias function.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : float or (nM,) array_like
            Halo mass in units of :math:`\rm M_\odot`.
        a : float
            Scale factor.

        Returns
        -------
        halo_bias : float or (nM,) ``numpy.ndarray``
            Value(s) of the halo bias function at ``M`` and ``a``.
        """
        M_use = np.atleast_1d(M)
        logM, sigM = self._get_logM_sigM(cosmo, M_use, a)
        b = self._get_bsigma(cosmo, sigM, a)
        if np.ndim(M) == 0:
            return b[0]
        return b

    get_halo_bias = __call__


class Concentration(HMIngredients):
    r"""Base class for halo mass-concentration relations.

    Implementation
    --------------
    - Subclasses must include arguments ``mass_def`` and ``mass_def_strict``
      in their :meth:`__init__()` methods.
    - Subclasses must define a :meth:`_concentration()` method which implements
      the concentration relation.
    - :meth:`_setup()` may be used to set up the model post-init.
      See :meth:`HMIngredients._setup()` for details.
    - The mass definition checks are always strict for the mass-concentration
      relations.

    Theory
    ------
    Halo mass-concentration relations are typically defined through the NFW
    profile,

    .. math::

        \rho(r) = \rho_{\rm c} \frac{\delta_c}{(r/r_s)(1 + r/r_s)^2},

    where :math:`r_s` is a scale radius. The concentration is then defined as

    .. math::

        c_\Delta \equiv \frac{r_\Delta}{r_s},

    where :math:`\Delta` is the density contrast.
    """
    _mass_def_strict_always = True

    @warn_api
    def __init__(self, *, mass_def):
        super().__init__(mass_def=mass_def, mass_def_strict=True)

    @abstractmethod
    def _concentration(self, cosmo, M, a):
        r"""Compute :math:`c(M)`.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : (nM,) ``numpy.ndarray``
            Halo mass in units of :math:`\rm M_\odot`.
            Input will always be an  array, and there is no need for the
            implemented function to convert it to one.
        a : float
            Scale factor.

        Returns
        -------
        concentration : (nM,) ``numpy.ndarray``
            Values of :math:`c(M)`. Output is expected to be an array
            and there is no need to squeeze extra dimensions.
        """

    def __call__(self, cosmo, M, a):
        r"""Call the concentration relation.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        M : float or (nM,) array_like
            Halo mass in units of :math:`\rm M_\odot`.
        a : float
            Scale factor.

        Returns
        -------
        concentration : float or (nM,) ``numpy.ndarray``
            Value(s) of the concentration :math:`c(M)` at ``M`` and ``a``.
        """
        M_use = np.atleast_1d(M)
        c = self._concentration(cosmo, M_use, a)
        if np.ndim(M) == 0:
            return c[0]
        return c

    get_concentration = __call__


@functools.wraps(MassFunc.from_name)
@deprecated(new_function=MassFunc.from_name)
def mass_function_from_name(name):
    return MassFunc.from_name(name)


@functools.wraps(HaloBias.from_name)
@deprecated(new_function=HaloBias.from_name)
def halo_bias_from_name(name):
    return HaloBias.from_name(name)


@functools.wraps(Concentration.from_name)
@deprecated(new_function=Concentration.from_name)
def concentration_from_name(name):
    return Concentration.from_name(name)
