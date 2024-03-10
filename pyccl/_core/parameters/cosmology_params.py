__all__ = ("CosmologyParams",)

from ... import lib
from . import CCLParameters


class CosmologyParams(CCLParameters, factory=lib.parameters):
    """Instances of this class hold cosmological parameters."""

    def __getattribute__(self, key):
        if key == "m_nu":
            N_nu_mass = self._instance.N_nu_mass
            nu_masses = lib.parameters_get_nu_masses(self._instance, N_nu_mass)
            return nu_masses.tolist()
        return super().__getattribute__(key)

    def __setattr__(self, key, value):
        if key == "m_nu":
            self._instance.N_nu_mass = len(value)
            return lib.parameters_m_nu_set_custom(self._instance, value)
        super().__setattr__(key, value)
