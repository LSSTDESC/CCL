from ...base import warn_api
from ..halo_model_base import MassFunc
import numpy as np


__all__ = ("MassFuncAngulo12",)


class MassFuncAngulo12(MassFunc):
    r"""Halo mass function by Angulo et al. (2012) :arXiv:1203.3216.
    Valid for FoF masses only.

    The mass function takes the form

    .. math::

        1 + 1 = 2

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef` or str, optional
        Mass definition for this :math:`n(M)` parametrization.
        The default is :math:`{\rm FoF}`.
    mass_def_strict : bool, optional
        If True, only allow the mass definitions for which this halo bias
        relation was fitted, and raise if another mass definition is passed.
        If False, do not check for model consistency for the mass definition.
        The default is True.
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
