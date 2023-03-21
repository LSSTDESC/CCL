from ..massdef import MassDef
from .hmfunc_base import MassFunc
import numpy as np


__all__ = ("MassFuncAngulo12",)


class MassFuncAngulo12(MassFunc):
    """ Implements mass function described in arXiv:1203.3216.
    This parametrization is only valid for 'fof' masses.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Angulo12'

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True):
        super(MassFuncAngulo12, self).__init__(cosmo,
                                               mass_def,
                                               mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef('fof', 'matter')

    def _setup(self, cosmo):
        self.A = 0.201
        self.a = 2.08
        self.b = 1.7
        self.c = 1.172

    def _check_mdef_strict(self, mdef):
        if mdef.Delta != 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * ((self.a / sigM)**self.b + 1.) * \
            np.exp(-self.c / sigM**2)
