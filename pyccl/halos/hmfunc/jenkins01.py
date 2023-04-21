__all__ = ("MassFuncJenkins01",)

import numpy as np

from ... import warn_api
from . import MassFunc


class MassFuncJenkins01(MassFunc):
    """ Implements mass function described in astro-ph/0005260.
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or str):
            a mass definition object, or a name string.
            This parametrization accepts FoF masses only.
            The default is 'fof'.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Jenkins01'

    @warn_api
    def __init__(self, *,
                 mass_def="fof",
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
