from ...base import warn_api
from ..massdef import MassDef
from ..halo_model_base import MassFunc
import numpy as np


__all__ = ("MassFuncPress74",)


class MassFuncPress74(MassFunc):
    """ Implements mass function described in 1974ApJ...187..425P.
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            The default is 'fof'.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Press74'

    @warn_api
    def __init__(self, *,
                 mass_def=MassDef('fof', 'matter'),
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _setup(self):
        self.norm = np.sqrt(2/np.pi)

    def _check_mass_def_strict(self, mass_def):
        if mass_def.Delta != 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        delta_c = 1.68647

        nu = delta_c/sigM
        return self.norm * nu * np.exp(-0.5 * nu**2)
