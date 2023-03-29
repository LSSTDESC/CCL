from ...base import warn_api
from ..massdef import MassDef
from ..halo_model_base import MassFunc
import numpy as np


__all__ = ("MassFuncJenkins01",)


class MassFuncJenkins01(MassFunc):
    """ Implements mass function described in astro-ph/0005260.
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            The default is 'fof'.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Jenkins01'

    @warn_api
    def __init__(self, *,
                 mass_def=MassDef('fof', 'matter'),
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.A = 0.315
        self.b = 0.61
        self.q = 3.8

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * np.exp(-np.abs(-np.log(sigM) + self.b)**self.q)
