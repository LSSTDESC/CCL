from ... import ccllib as lib
from .parameters_base import Parameters


__all__ = ("CosmologyParams",)


class CosmologyParams(Parameters, factory=lib.parameters):
    """Instances of this class hold cosmological parameters."""
    mgrowth: list = []

    def __getattribute__(self, key):
        if key == "m_nu":
            N_nu_mass = self.N_nu_mass
            nu_masses = lib.parameters_get_nu_masses(self._instance, N_nu_mass)
            return nu_masses.tolist()
        return super().__getattribute__(key)

    def __setattr__(self, key, value):
        if key == "m_nu":
            lib.parameters_m_nu_set_custom(self._instance, value)
            return object.__setattr__(self, "m_nu", self._instance.m_nu)
        if key == "mgrowth":
            return lib.parameters_mgrowth_set_custom(self._instance, *value)
        super().__setattr__(key, value)
