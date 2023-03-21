from ..massdef import MassDef
from .hmfunc_base import MassFunc
import numpy as np


__all__ = ("MassFuncPress74",)


class MassFuncPress74(MassFunc):
    """ Implements mass function described in 1974ApJ...187..425P.
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
    name = 'Press74'

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True):
        super(MassFuncPress74, self).__init__(cosmo,
                                              mass_def,
                                              mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef('fof', 'matter')

    def _setup(self, cosmo):
        self.norm = np.sqrt(2/np.pi)

    def _check_mdef_strict(self, mdef):
        if mdef.Delta != 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        delta_c = 1.68647

        nu = delta_c/sigM
        return self.norm * nu * np.exp(-0.5 * nu**2)
