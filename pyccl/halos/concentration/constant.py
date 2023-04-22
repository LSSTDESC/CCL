__all__ = ("ConcentrationConstant",)

import numpy as np

from ... import warn_api
from . import Concentration


class ConcentrationConstant(Concentration):
    """Constant contentration-mass relation.

    Parameters
    ---------
    c : float, optional
        Value of the constant concentration.
        The default is :math:`1.0`.
    mass_def : :class:`~pyccl.halos.massdef.MassDef` or str, optional
        The mass definition for this :math:`c(M)` parametrization is arbitrary
        and is not used for any calculations.
    """
    __repr_attrs__ = __eq_attrs__ = ("mass_def", "c",)
    name = 'Constant'

    @warn_api(pairs=[("mdef", "mass_def")])
    def __init__(self, c=1, *, mass_def="200c"):
        self.c = c
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return False

    def _concentration(self, cosmo, M, a):
        return np.full_like(M, self.c)[()]
