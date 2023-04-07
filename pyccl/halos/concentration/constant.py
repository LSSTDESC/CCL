from ..massdef import MassDef
from .concentration_base import Concentration
import numpy as np


__all__ = ("ConcentrationConstant",)


class ConcentrationConstant(Concentration):
    """ Constant contentration-mass relation.

    Args:
        c (float): constant concentration value.
        mdef (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization. In this case it's arbitrary.
    """
    __repr_attrs__ = __eq_attrs__ = ("mdef", "c",)
    name = 'Constant'

    def __init__(self, c=1, mdef=None):
        self.c = c
        super(ConcentrationConstant, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = MassDef(200, 'critical')

    def _check_mdef(self, mdef):
        return False

    def _concentration(self, cosmo, M, a):
        if np.ndim(M) == 0:
            return self.c
        else:
            return self.c * np.ones(M.size)
