__all__ = ("MassFuncAngulo12",)

import numpy as np

from ... import warn_api
from . import MassFunc


class MassFuncAngulo12(MassFunc):
    """ Implements mass function described in arXiv:1203.3216.
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or str):
            a mass definition object, or a name string.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Angulo12'

    @warn_api
    def __init__(self, *,
                 mass_def="fof",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.A = 0.201
        self.a = 2.08
        self.b = 1.7
        self.c = 1.172

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * ((self.a / sigM)**self.b + 1.) * (
            np.exp(-self.c / sigM**2))
