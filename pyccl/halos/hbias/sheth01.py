from ..massdef import MassDef
from .hbias_base import HaloBias


__all__ = ("HaloBiasSheth01",)


class HaloBiasSheth01(HaloBias):
    """ Implements halo bias described in arXiv:astro-ph/9907024.
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
    name = "Sheth01"

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True):
        super(HaloBiasSheth01, self).__init__(cosmo,
                                              mass_def,
                                              mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef('fof', 'matter')

    def _setup(self, cosmo):
        self.a = 0.707
        self.sqrta = 0.84083292038
        self.b = 0.5
        self.c = 0.6
        self.dc = 1.68647

    def _check_mdef_strict(self, mdef):
        if mdef.Delta != 'fof':
            return True
        return False

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc/sigM
        anu2 = self.a * nu**2
        anu2c = anu2**self.c
        t1 = self.b * (1.0 - self.c) * (1.0 - 0.5 * self.c)
        return 1. + (self.sqrta * anu2 * (1 + self.b / anu2c) -
                     anu2c / (anu2c + t1)) / (self.sqrta * self.dc)
