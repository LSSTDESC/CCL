from ...base import warn_api
from ..massdef import MassDef
from ..halo_model_base import HaloBias


__all__ = ("HaloBiasBhattacharya11",)


class HaloBiasBhattacharya11(HaloBias):
    """ Implements halo bias described in arXiv:1005.2239.
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = "Bhattacharya11"

    @warn_api
    def __init__(self, *,
                 mass_def=MassDef('fof', 'matter'),
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _setup(self):
        self.a = 0.788
        self.az = 0.01
        self.p = 0.807
        self.q = 1.795
        self.dc = 1.68647

    def _check_mass_def_strict(self, mass_def):
        if mass_def.Delta != 'fof':
            return True
        return False

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc / sigM
        a = self.a * a**self.az
        anu2 = a * nu**2
        return 1. + (anu2 - self.q + 2*self.p / (1 + anu2**self.p)) / self.dc
