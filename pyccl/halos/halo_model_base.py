from .. import ccllib as lib
from ..pyutils import check
from ..parameters import physical_constants as const
from ..base import (CCLAutoRepr, CCLNamedClass,
                    warn_api, deprecated, deprecate_attr)
import numpy as np
import functools
from abc import abstractmethod


__all__ = ("HMIngredients",)


class HMIngredients(CCLAutoRepr, CCLNamedClass):
    """Base class for halo model ingredients."""
    __repr_attrs__ = __eq_attrs__ = ("mass_def", "mass_def_strict",)
    __getattr__ = deprecate_attr(pairs=[('mdef', 'mass_def')]
                                 )(super.__getattribute__)

    @warn_api
    def __init__(self, *, mass_def, mass_def_strict=True):
        # Check mass definition consistency.
        from .massdef import MassDef
        mass_def = MassDef.create_instance(mass_def)
        self.mass_def_strict = mass_def_strict
        self._check_mass_def(mass_def)
        self.mass_def = mass_def
        self._setup()

    @property
    @abstractmethod
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
        """Check if this class is defined for mass definition ``mass_def``."""

    def _setup(self) -> None:
        """ Use this function to initialize any internal attributes
        of this object. This function is called at the very end of the
        constructor call.
        """

    def _check_mass_def(self, mass_def) -> None:
        """ Return False if the input mass definition agrees with
        the definitions for which this mass function parametrization
        works. True otherwise. This function gets called at the
        start of the constructor call.

        Args:
            mass_def (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.
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
    """ This class enables the calculation of halo mass functions.
    We currently assume that all mass functions can be written as

    .. math::
        \\frac{dn}{d\\log_{10}M} = f(\\sigma_M)\\,\\frac{\\rho_M}{M}\\,
        \\frac{d\\log \\sigma_M}{d\\log_{10} M}

    where :math:`\\sigma_M^2` is the overdensity variance on spheres with a
    radius given by the Lagrangian radius for mass M.

    * Subclasses implementing analytical mass function parametrizations
      can be created by overriding the ``_get_fsigma`` method.

    * Subclasses may have particular implementations of
      ``_check_mass_def_strict`` to ensure consistency of the halo mass
      definition.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object that fixes
            the mass definition used by this mass function
            parametrization.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    _mass_def_strict_always = False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        """ Get the :math:`f(\\sigma_M)` function for this mass function
        object (see description of this class for details).

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            sigM (float or array_like): standard deviation in the
                overdensity field on the scale of this halo.
            a (float): scale factor.
            lnM (float or array_like): natural logarithm of the
                halo mass in units of M_sun (provided in addition
                to sigM for convenience in some mass function
                parametrizations).

        Returns:
            float or array_like: :math:`f(\\sigma_M)` function.
        """

    def __call__(self, cosmo, M, a):
        """ Returns the mass function for input parameters.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.

        Returns:
            float or array_like: mass function \
                :math:`dn/d\\log_{10}M` in units of Mpc^-3 (comoving).
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

    @deprecated(new_function=__call__)
    def get_mass_function(self, cosmo, M, a):
        return self(cosmo, M, a)


class HaloBias(HMIngredients):
    """ This class enables the calculation of halo bias functions.
    We currently assume that all halo bias functions can be written
    as functions that depend on M only through sigma_M (where
    sigma_M^2 is the overdensity variance on spheres with a
    radius given by the Lagrangian radius for mass M).
    All sub-classes implementing specific parametrizations
    can therefore be simply created by replacing this class'
    `_get_bsigma method`.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this halo bias
            parametrization.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    _mass_def_strict_always = False

    def _get_bsigma(self, cosmo, sigM, a):
        """ Get the halo bias as a function of sigmaM.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            sigM (float or array_like): standard deviation in the
                overdensity field on the scale of this halo.
            a (float): scale factor.

        Returns:
            float or array_like: f(sigma_M) function.
        """

    def __call__(self, cosmo, M, a):
        """ Returns the halo bias for input parameters.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.

        Returns:
            float or array_like: halo bias.
        """
        M_use = np.atleast_1d(M)
        logM, sigM = self._get_logM_sigM(cosmo, M_use, a)
        b = self._get_bsigma(cosmo, sigM, a)
        if np.ndim(M) == 0:
            return b[0]
        return b

    @deprecated(new_function=__call__)
    def get_halo_bias(self, cosmo, M, a):
        return self(cosmo, M, a)


class Concentration(HMIngredients):
    """ This class enables the calculation of halo concentrations.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass definition
            object that fixes the mass definition used by this c(M)
            parametrization.
    """
    _mass_def_strict_always = True

    @warn_api
    def __init__(self, *, mass_def):
        super().__init__(mass_def=mass_def, mass_def_strict=True)

    def _concentration(self, cosmo, M, a):
        """Implementation of the c(M) relation."""

    def __call__(self, cosmo, M, a):
        """ Returns the concentration for input parameters.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.

        Returns:
            float or array_like: concentration.
        """
        M_use = np.atleast_1d(M)
        c = self._concentration(cosmo, M_use, a)
        if np.ndim(M) == 0:
            return c[0]
        return c

    @deprecated(new_function=__call__)
    def get_concentration(self, cosmo, M, a):
        return self(cosmo, M, a)


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
