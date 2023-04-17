from .. import ccllib as lib
from ..core import check
from ..background import omega_x
from ..base import CCLHalosObject, deprecated, warn_api
import numpy as np
import functools
from abc import abstractmethod


class HaloBias(CCLHalosObject):
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
    __repr_attrs__ = ("mass_def", "mass_def_strict",)

    @warn_api
    def __init__(self, *,
                 mass_def=None,
                 mass_def_strict=True):
        self.mass_def_strict = mass_def_strict
        if mass_def is not None:
            if self._check_mass_def(mass_def):
                raise ValueError("Halo bias " + self.name +
                                 " is not compatible with mass definition" +
                                 " Delta = %s, " % (mass_def.Delta) +
                                 " rho = " + mass_def.rho_type)
            self.mass_def = mass_def
        self._setup()

    def _setup(self):
        """ Use this function to initialize any internal attributes
        of this object. This function is called at the very end of the
        constructor call.
        """

    def _check_mass_def_strict(self, mass_def):
        return False

    def _check_mass_def(self, mass_def):
        """ Return False if the input mass definition agrees with
        the definitions for which this parametrization
        works. True otherwise. This function gets called at the
        start of the constructor call.

        Args:
            mass_def (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            bool: True if the mass definition is not compatible with
                this parametrization. False otherwise.
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
            float or array_like: mass according to this object's
            mass definition.
        """
        if mass_def_other is not None:
            M_use = mass_def_other.translate_mass(
                cosmo, M, a,
                mass_def_other=self.mass_def)
        else:
            M_use = M
        return np.log10(M_use)

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
            om_this = omega_x(cosmo, a, self.mass_def.rho_type)
            om_matt = omega_x(cosmo, a, 'matter')
            return delta * om_this / om_matt

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
        cosmo.compute_sigma()  # compute sigma if needed

        M_use = np.atleast_1d(M)
        logM = self._get_consistent_mass(cosmo, M_use, a, mass_def_other)

        # sigma(M)
        status = 0
        sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM, len(logM), status)
        check(status, cosmo=cosmo)

        b = self._get_bsigma(cosmo, sigM, a)
        if np.ndim(M) == 0:
            b = b[0]
        return b

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

    @classmethod
    def from_name(cls, name):
        """Returns halo bias subclass from name string

        Args:
            name (string): a halo bias name

        Returns:
            HaloBias subclass corresponding to the input name.
        """
        bias_functions = {c.name: c for c in cls.__subclasses__()}
        if name in bias_functions:
            return bias_functions[name]
        else:
            raise ValueError(
                f"Halo bias parametrization {name} not implemented.")


@functools.wraps(HaloBias.from_name)
@deprecated(new_function=HaloBias.from_name)
def halo_bias_from_name(name):
    return HaloBias.from_name(name)
