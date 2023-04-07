from ..massdef import MassDef
from .hbias_base import HaloBias


__all__ = ("HaloBiasBhattacharya11",)


class HaloBiasBhattacharya11(HaloBias):
    """ Implements halo bias described in arXiv:1005.2239.
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
    name = "Bhattacharya11"

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True):
        super(HaloBiasBhattacharya11, self).__init__(cosmo,
                                                     mass_def,
                                                     mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef('fof', 'matter')

    def _setup(self, cosmo):
        self.a = 0.788
        self.az = 0.01
        self.p = 0.807
        self.q = 1.795
        self.dc = 1.68647

    def _check_mdef_strict(self, mdef):
        if mdef.Delta != 'fof':
            return True
        return False

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc / sigM
        a = self.a * a**self.az
        anu2 = a * nu**2
        return 1. + (anu2 - self.q + 2*self.p / (1 + anu2**self.p)) / self.dc
