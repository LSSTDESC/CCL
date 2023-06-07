"""
===========================================================
Halo model ingredients (:mod:`pyccl.halos.halo_model_base`)
===========================================================

Abstract base class for halo mass functions, halo bias functions, and
mass-concentration relations.
"""

from __future__ import annotations

__all__ = ("HMIngredients", "MassFunc", "HaloBias", "Concentration",
           "mass_function_from_name", "halo_bias_from_name",
           "concentration_from_name",)

import functools
from abc import abstractmethod
from numbers import Real
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray

from .. import CCLNamedClass, lib
from .. import deprecate_attr, deprecated, warn_api, mass_def_api
from .. import physical_constants as const

if TYPE_CHECKING:
    from .. import Cosmology
    from . import MassDef


class HMIngredients(CCLNamedClass):
    r"""Abstract base class for halo model ingredients. Ensures consistency of
    mass definition.

    Parameters
    ----------
    mass_def
        Mass definition.
    mass_def_strict
        Whether the instance will allow for incompatible mass definitions.

    Raises
    ------
    ValueError
        If the mass definition is incompatible with the model setup.
    """
    __repr_attrs__ = __eq_attrs__ = ("mass_def", "mass_def_strict",)
    __getattr__ = deprecate_attr(pairs=[('mdef', 'mass_def')]
                                 )(super.__getattribute__)
    mass_def: MassDef
    mass_def_strict: bool

    @warn_api
    def __init__(
            self,
            *,
            mass_def: Union[MassDef, str],
            mass_def_strict: bool = True
    ):
        # Check mass definition consistency.
        from .massdef import MassDef  # noqa
        mass_def = MassDef.create_instance(mass_def)
        self.mass_def_strict = mass_def_strict
        self._check_mass_def(mass_def)
        self.mass_def = mass_def
        self._setup()

    @property
    @abstractmethod
    def _mass_def_strict_always(self) -> bool:
        r"""Property that dictates whether `mass_def_strict` can be set
        as False on initialization.

        Some models are set up in a way so that the set of fitted parameters
        depends on the mass definition, (i.e. no one universal model exists
        to cover all cases). Setting this to True fixes `mass_def_strict`
        to True irrespective of what the user passes.

        Set this propery to False to allow users to override strict checks.

        :meta public:
        """

    @abstractmethod
    def _check_mass_def_strict(self, mass_def: MassDef) -> bool:
        r"""Check if a mass definition is compatible with the model.

        :meta public:

        Arguments
        ---------
        mass_def
            Mass definition to check for compatibility.

        Returns
        -------

            Flag denoting whether the mass definition is compatible.
        """

    def _setup(self) -> None:
        r"""Initialize any internal attributes of this object.
        This function is the equivalent of a post-init method, called at
        the end of initialization.

        :meta public:
        """

    def _check_mass_def(self, mass_def: MassDef) -> None:
        r"""Check if a mass definition is compatible with the model and the
        initialization parameters.

        :meta public:

        Arguments
        ---------
        mass_def
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
        # Compute `logM`, `sigM`, and (optionally) `dlns_dlogM`.
        cosmo.compute_sigma()  # initialize sigma(M) splines if needed
        logM = np.log10(M)

        # sigma(M)
        status = 0
        sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM, len(logM), status)
        cosmo.check(status)
        if not return_dlns:
            return logM, sigM

        # dlogsigma(M)/dlog10(M)
        dlns_dlogM, status = lib.dlnsigM_dlogM_vec(cosmo.cosmo, a, logM,
                                                   len(logM), status)
        cosmo.check(status)
        return logM, sigM, dlns_dlogM


class MassFunc(HMIngredients):
    r"""Base class for halo mass functions.

    We assume that all mass functions can be written as

    .. math::

        \frac{{\rm d}n}{{\rm d}\log_{10}M} = f(\sigma_M) \, \frac{\rho_M}{M} \,
        \frac{{\rm d}\log \sigma_M}{{\rm d}\log_{10} M}

    where :math:`\sigma_M^2` is the overdensity variance on spheres with a
    radius given by the Lagrangian radius for mass :math:`M`.
    """
    _mass_def_strict_always = False

    def _get_fsigma(
            self,
            cosmo: Cosmology,
            sigM: NDArray[float],
            a: Real,
            lnM: NDArray[float]
    ) -> NDArray[float]:
        r"""Compute :math:`f(\sigma_M)`.

        :meta public:

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        sigM : ndarray (nM,)
            Standard deviation in the overdensity field on the scale of
            halos of mass :math:`M`. Input will always be an array, and there
            is no need for the implemented function to convert it to one.
        a
            Scale factor.
        lnM : ndarray (nM,)
            Natural logarithm of the halo mass in units of :math:`\rm M_\odot`
            (provided in addition to sigM for convenience in some mass function
            parametrizations). Input will always be an array, and there is no
            need for the implemented function to convert it to one.

        Returns
        -------
        f_sigma : ndarray (nM,)
            Values of :math:`f(\sigma_M)`. Output is expected to be an array
            and there is no need to squeeze extra dimensions.
        """

    def __call__(
            self,
            cosmo: Cosmology,
            M: Union[Real, NDArray[Real]],
            a: Real
    ) -> Union[float, NDArray[float]]:
        r"""Call the mass function. Calls :meth:`MassFunc._get_fsigma`.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        M : array_like (nM,)
            Halo mass in units of :math:`\rm M_\odot`.
        a
            Scale factor.

        Returns
        -------
        mass_function : array_like (nM,)
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

    @deprecated(new_api=__call__)
    @mass_def_api
    def get_mass_function(self, cosmo, M, a):
        """Call the mass function.

        .. deprecated:: 2.8.0

            Use :meth:`~MassFunc.__call__`.
        """
        return self(cosmo, M, a)


class HaloBias(HMIngredients):
    r"""Base class for halo bias functions.

    We assume that all halo bias parametrizations can be written as functions
    that depend only on :math:`M` through :math:`\sigma_M`, the overdensity
    variance on spheres of the Lagrangian radius that corresponds to mass
    :math:`M`.
    """
    _mass_def_strict_always = False

    def _get_bsigma(
            self,
            cosmo: Cosmology,
            sigM: NDArray[float],
            a: Real
    ) -> NDArray[float]:
        r"""Compute :math:`b(\sigma_M)`.

        :meta public:

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        sigM : ndarray (nM,)
            Standard deviation in the overdensity field on the scale of
            halos of mass :math:`M`. Input will always be an array, and there
            is no need for the implemented function to convert it to one.
        a : int or float
            Scale factor.

        Returns
        -------
        b_sigma : ndarray (nM,)
            Values of :math:`b(\sigma_M)`. Output is expected to be an array
            and there is no need to squeeze extra dimensions.
        """

    def __call__(
            self,
            cosmo: Cosmology,
            M: Union[Real, NDArray[Real]],
            a: Real
    ) -> Union[Real, NDArray[Real]]:
        r"""Call the halo bias function. Calls :meth:`~HaloBias._get_bsigma`.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        M : array_like (nM,)
            Halo mass in units of :math:`\rm M_\odot`.
        a : int or float
            Scale factor.

        Returns
        -------
        halo_bias : array_like (nM,)
            Value(s) of the halo bias function at `M` and `a`.
        """
        M_use = np.atleast_1d(M)
        logM, sigM = self._get_logM_sigM(cosmo, M_use, a)
        b = self._get_bsigma(cosmo, sigM, a)
        if np.ndim(M) == 0:
            return b[0]
        return b

    @deprecated(new_api=__call__)
    @mass_def_api
    def get_halo_bias(self, cosmo, M, a):
        """Call the halo bias function.

        .. deprecated:: 2.8.0
            Use :meth:`~HaloBias.__call__`.
        """
        return self(cosmo, M, a)


class Concentration(HMIngredients):
    r"""Base class for halo mass-concentration relations.

    Halo mass-concentration relations are typically defined through the NFW
    profile,

    .. math::

        \rho(r) = \rho_{\rm c} \frac{\delta_c}{(r/r_s)(1 + r/r_s)^2},

    where :math:`r_s` is a scale radius. The concentration is then defined as

    .. math::

        c_\Delta \equiv \frac{r_\Delta}{r_s},

    where :math:`\Delta` is the density contrast.
    """
    _mass_def_strict_always: bool = True
    mass_def_strict: bool = True

    @warn_api
    def __init__(self, *, mass_def):
        super().__init__(mass_def=mass_def, mass_def_strict=True)

    def _concentration(
            self,
            cosmo: Cosmology,
            M: NDArray[float],
            a: Real
    ) -> NDArray[float]:
        r"""Compute :math:`c(M)`.

        :meta public:

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        M : ndarray (nM,)
            Halo mass in units of :math:`\rm M_\odot`.
            Input will always be an  array, and there is no need for the
            implemented function to convert it to one.
        a
            Scale factor.

        Returns
        -------
        concentration : ndarray (nM,)
            Values of :math:`c(M)`. Output is expected to be an array
            and there is no need to squeeze extra dimensions.
        """

    def __call__(
            self,
            cosmo: Cosmology,
            M: Union[Real, NDArray[Real]],
            a: Real
    ) -> Union[Real, NDArray[Real]]:
        r"""Call the concentration relation. Calls
        :meth:`~Concentration._concentration`.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        M : array_like (nM,)
            Halo mass in units of :math:`\rm M_\odot`.
        a
            Scale factor.

        Returns
        -------
        concentration : array_like (nM,)
            Value(s) of the concentration :math:`c(M)` at `M` and `a`.
        """
        M_use = np.atleast_1d(M)
        c = self._concentration(cosmo, M_use, a)
        if np.ndim(M) == 0:
            return c[0]
        return c

    @deprecated(new_api=__call__)
    @mass_def_api
    def get_concentration(self, cosmo, M, a):
        """Call the concentration relation.

        .. deprecated:: 2.8.0
            Use :meth:`~Concentration.__call__`.
        """
        return self(cosmo, M, a)


@functools.wraps(MassFunc.from_name)
@deprecated(new_api=MassFunc.from_name)
def mass_function_from_name(name):
    return MassFunc.from_name(name)


@functools.wraps(HaloBias.from_name)
@deprecated(new_api=HaloBias.from_name)
def halo_bias_from_name(name):
    return HaloBias.from_name(name)


@functools.wraps(Concentration.from_name)
@deprecated(new_api=Concentration.from_name)
def concentration_from_name(name):
    return Concentration.from_name(name)
