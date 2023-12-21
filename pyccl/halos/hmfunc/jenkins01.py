__all__ = ("MassFuncJenkins01",)

import numpy as np

from . import MassFunc


class MassFuncJenkins01(MassFunc):
    """Implements the mass function of `Jenkins et al. 2001
    <https://arxiv.org/abs/astro-ph/0005260>`_. This parametrization
    is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
            definition will be ignored.
    """
    name = 'Jenkins01'

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
