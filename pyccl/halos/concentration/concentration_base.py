from ...base import CCLAutoreprObject
import numpy as np
from abc import abstractmethod
import functools


__all__ = ("Concentration", "concentration_from_name",)


class Concentration(CCLAutoreprObject):
    """ This class enables the calculation of halo concentrations.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass definition
            object that fixes the mass definition used by this c(M)
            parametrization.
    """
    __repr_attrs__ = __eq_attrs__ = ("mdef",)

    def __init__(self, mass_def=None):
        if mass_def is not None:
            if self._check_mdef(mass_def):
                raise ValueError(
                    f"Mass definition {mass_def.Delta}-{mass_def.rho_type} "
                    f"is not compatible with c(M) {self.name} configuration.")
            self.mdef = mass_def
        else:
            self._default_mdef()
        self._setup()

    @abstractmethod
    def _default_mdef(self):
        """ Assigns a default mass definition for this object if
        none is passed at initialization.
        """

    def _setup(self):
        """ Use this function to initialize any internal attributes
        of this object. This function is called at the very end of the
        constructor call.
        """
        pass

    def _check_mdef(self, mdef):
        """ Return False if the input mass definition agrees with
        the definitions for which this concentration-mass relation
        works. True otherwise. This function gets called at the
        start of the constructor call.

        Args:
            mdef (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            bool: True if the mass definition is not compatible with
                this parametrization. False otherwise.
        """
        return False

    def _get_consistent_mass(self, cosmo, M, a, mdef_other):
        """ Transform a halo mass with a given mass definition into
        the corresponding mass definition that was used to initialize
        this object.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mdef_other (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            float or array_like: mass according to this object's
            mass definition.
        """
        if mdef_other is not None:
            M_use = mdef_other.translate_mass(cosmo, M, a, self.mdef)
        else:
            M_use = M
        return M_use

    @abstractmethod
    def _concentration(self, cosmo, M, a):
        """Implementation of the c(M) relation."""

    def get_concentration(self, cosmo, M, a, mdef_other=None):
        """ Returns the concentration for input parameters.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mdef_other (:class:`~pyccl.halos.massdef.MassDef`):
                the mass definition object that defines M.

        Returns:
            float or array_like: concentration.
        """
        M_use = self._get_consistent_mass(cosmo,
                                          np.atleast_1d(M),
                                          a, mdef_other)

        c = self._concentration(cosmo, M_use, a)
        if np.ndim(M) == 0:
            c = c[0]
        return c

    @classmethod
    def from_name(cls, name):
        """ Returns halo concentration subclass from name string

        Args:
            name (string): a concentration name

        Returns:
            Concentration subclass corresponding to the input name.
        """
        concentrations = {c.name: c for c in cls.__subclasses__()}
        if name in concentrations:
            return concentrations[name]
        else:
            raise ValueError(f"Concentration {name} not implemented.")


@functools.wraps(Concentration.from_name)
def concentration_from_name(name):
    return Concentration.from_name(name)
