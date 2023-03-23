from ...base import warn_api
from ..massdef import MassDef
from ..halo_model_base import MassFunc
import numpy as np


__all__ = ("MassFuncAngulo12",)


class MassFuncAngulo12(MassFunc):
    """ Implements mass function described in arXiv:1203.3216.
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Angulo12'

    @warn_api
    def __init__(self, *,
                 mass_def=MassDef('fof', 'matter'),
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _setup(self):
        self.A = 0.201
        self.a = 2.08
        self.b = 1.7
        self.c = 1.172

    def _check_mass_def_strict(self, mass_def):
        if mass_def.Delta != 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * ((self.a / sigM)**self.b + 1.) * \
            np.exp(-self.c / sigM**2)
