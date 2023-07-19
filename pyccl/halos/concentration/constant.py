__all__ = ("ConcentrationConstant",)

import numpy as np

from . import Concentration


class ConcentrationConstant(Concentration):
    """Constant contentration-mass relation.

    Args:
        c (:obj:`float`): constant concentration value.
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`): a mass
            definition (arbitrary in this case).
    """
    __repr_attrs__ = __eq_attrs__ = ("mass_def", "c",)
    name = 'Constant'

    def __init__(self, c=1, *, mass_def="200c"):
        self.c = c
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return False

    def _concentration(self, cosmo, M, a):
        return np.full_like(M, self.c)[()]
