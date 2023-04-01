from ...base import warn_api
from ..massdef import MassDef
from ..halo_model_base import Concentration
import numpy as np


__all__ = ("ConcentrationConstant",)


class ConcentrationConstant(Concentration):
    """Constant contentration-mass relation.

    .. note::
        The mass definition for this concentration is arbitrary, and is
        internally set to ``None``.

    Parameters
    ---------
    c : float
        Value of the constant concentration.
    """
    __repr_attrs__ = ("mass_def", "c",)
    name = 'Constant'

    @warn_api(pairs=[("mdef", "mass_def")])
    def __init__(self, c=1, *, mass_def=MassDef(200, 'critical')):
        self.c = c
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return False

    def _concentration(self, cosmo, M, a):
        return np.full_like(M, self.c)
