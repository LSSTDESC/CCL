from .. import ccllib as lib
from ..core import check
from ..parameters import physical_constants
from ..base import CCLAutoreprObject, warn_api, deprecated, deprecate_attr
import numpy as np
import functools
from abc import abstractmethod


__all__ = ("HMIngredients", "get_mass_function_and_halo_bias",)


class HMIngredients(CCLAutoreprObject):
    """Base class for halo model ingredients."""
    __repr_attrs__ = ("mass_def", "mass_def_strict",)
    __getattr__ = deprecate_attr(pairs=[('mdef', 'mass_def')]
                                 )(super.__getattribute__)

    @warn_api
    def __init__(self, *, mass_def, mass_def_strict=True):
        self.mass_def_strict = mass_def_strict
        # Check if mass definition was provided and check that it's sensible.
        if self._check_mass_def(mass_def):
            classname = self.__class__.name
            raise ValueError(
                f"{classname} is not defined for {mass_def.name}-based masses."
                " To disable this exception set `mass_def_strict=True`.")
        self.mass_def = mass_def
        self._setup()

    def _setup(self):
        """ Use this function to initialize any internal attributes
        of this object. This function is called at the very end of the
        constructor call.
        """

    @abstractmethod
    def _check_mass_def_strict(self, mass_def):
        """Check if this class is defined for mass definition ``mass_def``."""

    def _check_mass_def(self, mass_def):
        """ Return False if the input mass definition agrees with
        the definitions for which this mass function parametrization
        works. True otherwise. This function gets called at the
        start of the constructor call.

        Args:
            mass_def (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            bool: True if the mass definition is not compatible with \
                this mass function parametrization. False otherwise.
        """
        if self.mass_def_strict:
            return self._check_mass_def_strict(mass_def)
        return False

    def _get_consistent_mass(self, cosmo, M, a, mass_def_other):
        """ Transform a halo mass with a given mass definition into
        the corresponding mass definition that was used to initialize
        this object.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def_other (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            float or array_like: mass according to this object's \
                mass definition.
        """
        if mass_def_other is not None:
            M_use = mass_def_other.translate_mass(
                cosmo, M, a,
                mass_def_other=self.mass_def)
        else:
            M_use = M
        return M_use

    def _get_Delta_m(self, cosmo, a):
        """ For SO-based mass definitions, this returns the corresponding
        value of Delta for a rho_matter-based definition. This is useful
        mostly for the Tinker mass functions, which are defined for any
        SO mass in general, but explicitly only for Delta_matter.
        """
        delta = self.mass_def.get_Delta(cosmo, a)
        if self.mass_def.rho_type == 'matter':
            return delta
        else:
            om_this = cosmo.omega_x(a, self.mass_def.rho_type)
            om_matt = cosmo.omega_x(a, 'matter')
            return delta * om_this / om_matt

    def _get_logM_sigM(self, cosmo, M, a, mass_def, *, return_dlns=False):
        """Compute ``logM``, ``sigM``, and (optionally) ``dlns_dlogM``."""
        cosmo.compute_sigma()  # initialize sigma(M) splines if needed
        logM = np.log10(self._get_consistent_mass(cosmo, M, a, mass_def))

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

    @classmethod
    def from_name(cls, name):
        """Get particular model from name string.

        Args:
            name (string): the model name
        """
        models = {c.name: c for c in cls.__subclasses__()}
        return models[name]


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

    @abstractmethod
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

    @warn_api(pairs=[("mdef_other", "mass_def_other")])
    def get_mass_function(self, cosmo, M, a, *, mass_def_other=None):
        """ Returns the mass function for input parameters.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def_other (:class:`~pyccl.halos.massdef.MassDef`):
                the mass definition object that defines M.

        Returns:
            float or array_like: mass function \
                :math:`dn/d\\log_{10}M` in units of Mpc^-3 (comoving).
        """
        M_use = np.atleast_1d(M)
        logM, sigM, dlns_dlogM = self._get_logM_sigM(
            cosmo, M_use, a, mass_def_other, return_dlns=True)

        rho = (physical_constants.RHO_CRITICAL *
               cosmo['Omega_m'] * cosmo['h']**2)
        f = self._get_fsigma(cosmo, sigM, a, 2.302585092994046 * logM)
        mf = f * rho * dlns_dlogM / M_use
        if np.ndim(M) == 0:
            mf = mf[0]
        return mf


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

    @abstractmethod
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

    @warn_api(pairs=[("mdef_other", "mass_def_other")])
    def get_halo_bias(self, cosmo, M, a, *, mass_def_other=None):
        """ Returns the halo bias for input parameters.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def_other (:class:`~pyccl.halos.massdef.MassDef`):
                the mass definition object that defines M.

        Returns:
            float or array_like: halo bias.
        """
        M_use = np.atleast_1d(M)
        logM, sigM = self._get_logM_sigM(cosmo, M_use, a, mass_def_other)
        b = self._get_bsigma(cosmo, sigM, a)
        if np.ndim(M) == 0:
            b = b[0]
        return b


class Concentration(HMIngredients):
    """ This class enables the calculation of halo concentrations.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass definition
            object that fixes the mass definition used by this c(M)
            parametrization.
    """

    @warn_api
    def __init__(self, *, mass_def):
        super().__init__(mass_def=mass_def, mass_def_strict=True)

    @abstractmethod
    def _concentration(self, cosmo, M, a):
        """Implementation of the c(M) relation."""

    @warn_api(pairs=[("mdef_other", "mass_def_other")])
    def get_concentration(self, cosmo, M, a, *, mass_def_other=None):
        """ Returns the concentration for input parameters.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def_other (:class:`~pyccl.halos.massdef.MassDef`):
                the mass definition object that defines M.

        Returns:
            float or array_like: concentration.
        """
        M_use = np.atleast_1d(M)
        M_use = self._get_consistent_mass(cosmo, M_use, a, mass_def_other)
        c = self._concentration(cosmo, M_use, a)
        if np.ndim(M) == 0:
            c = c[0]
        return c


def get_mass_function_and_halo_bias(cosmo, hmc, M, a):
    """Helper to get mass function and halo bias in a single step."""
    M_use = np.atleast_1d(M)
    logM, sigM, dlns_dlogM = hmc.mass_function._get_logM_sigM(
        cosmo, M, a, hmc.mass_def, return_dlns=True)
    # mass function
    rho = (physical_constants.RHO_CRITICAL *
           cosmo['Omega_m'] * cosmo['h']**2)
    f = hmc.mass_function._get_fsigma(cosmo, sigM, a, 2.302585092994046 * logM)
    mf = f * rho * dlns_dlogM / M_use
    # halo bias
    b = hmc.halo_bias._get_bsigma(cosmo, sigM, a)
    if np.ndim(M) == 0:
        mf, b = mf[0], b[0]
    return mf, b


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
