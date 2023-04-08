from ...base import warn_api
from ...base.parameters import physical_constants as const
from ..halo_model_base import HaloBias


__all__ = ("HaloBiasBhattacharya11",)


class HaloBiasBhattacharya11(HaloBias):
    """ Implements halo bias described in arXiv:1005.2239.
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or str):
            a mass definition object, or a name string.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = "Bhattacharya11"

    @warn_api
    def __init__(self, *,
                 mass_def="fof",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.Delta != "fof"

    def _setup(self):
        self.a = 0.788
        self.az = 0.01
        self.p = 0.807
        self.q = 1.795
        self.dc = const.DELTA_C

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc / sigM
        a = self.a * a**self.az
        anu2 = a * nu**2
        return 1 + (anu2 - self.q + 2*self.p / (1 + anu2**self.p)) / self.dc
