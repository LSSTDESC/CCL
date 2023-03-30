from ...base import warn_api
from ..massdef import MassDef200m
from ..halo_model_base import HaloBias, TinkerFunction
import numpy as np


__all__ = ("HaloBiasTinker10",)


class HaloBiasTinker10(HaloBias, TinkerFunction):
    """ Implements halo bias described in arXiv:1001.3162.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts SO masses with
            200 < Delta < 3200 with respect to the matter density.
            If `None`, Delta = 200 (matter) will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = "Tinker10"

    @warn_api
    def __init__(self, *,
                 mass_def=MassDef200m(),
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta == "fof"

    def _setup(self):
        self.B = 0.183
        self.b = 1.5
        self.c = 2.4
        self.dc = 1.68647

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc / sigM
        ld = np.log10(self._get_Delta_m(cosmo, a))
        xp = np.exp(-(4./ld)**4.)
        A = 1.0 + 0.24 * ld * xp
        C = 0.019 + 0.107 * ld + 0.19*xp
        aa = 0.44 * ld - 0.88
        nupa = nu**aa
        return 1 - A * nupa / (nupa + self.dc**aa) + (
            self.B * nu**self.b + C * nu**self.c)
