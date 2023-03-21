from ...base import warn_api
from ..massdef import MassDef
from .concentration_base import Concentration
import numpy as np


__all__ = ("ConcentrationConstant",)


class ConcentrationConstant(Concentration):
    """ Constant contentration-mass relation.

    Args:
        c (float): constant concentration value.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization. In this case it's arbitrary.
    """
    __repr_attrs__ = ("mass_def", "c",)
    name = 'Constant'

    @warn_api(pairs=[("mdef", "mass_def")])
    def __init__(self, c=1, *, mass_def=None):
        self.c = c
        super(ConcentrationConstant, self).__init__(mass_def=mass_def)

    def _default_mass_def(self):
        self.mass_def = MassDef(200, 'critical')

    def _check_mass_def(self, mass_def):
        return False

    def _concentration(self, cosmo, M, a):
        if np.ndim(M) == 0:
            return self.c
        else:
            return self.c * np.ones(M.size)
