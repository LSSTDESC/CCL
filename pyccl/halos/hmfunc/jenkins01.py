from ..massdef import MassDef
from .hmfunc_base import MassFunc
import numpy as np


__all__ = ("MassFuncJenkins01",)


class MassFuncJenkins01(MassFunc):
    """ Implements mass function described in astro-ph/0005260.
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
    name = 'Jenkins01'

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True):
        super(MassFuncJenkins01, self).__init__(cosmo,
                                                mass_def=mass_def,
                                                mass_def_strict=True)

    def _default_mdef(self):
        self.mdef = MassDef('fof', 'matter')

    def _setup(self, cosmo):
        self.A = 0.315
        self.b = 0.61
        self.q = 3.8

    def _check_mdef_strict(self, mdef):
        if mdef.Delta != 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * np.exp(-np.fabs(-np.log(sigM) + self.b)**self.q)
