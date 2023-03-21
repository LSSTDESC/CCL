from ... import ccllib as lib
from ...core import check
from ..massdef import MassDef
from .hbias_base import HaloBias


__all__ = ("HaloBiasSheth99",)


class HaloBiasSheth99(HaloBias):
    """ Implements halo bias described in 1999MNRAS.308..119S
    This parametrization is only valid for 'fof' masses.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
        use_delta_c_fit (bool): if True, use delta_crit given by
            the fit of Nakamura & Suto 1997. Otherwise use
            delta_crit = 1.68647.
    """
    __repr_attrs__ = ("mdef", "mass_def_strict", "use_delta_c_fit",)
    name = "Sheth99"

    def __init__(self, cosmo, mass_def=None,
                 mass_def_strict=True,
                 use_delta_c_fit=False):
        self.use_delta_c_fit = use_delta_c_fit
        super(HaloBiasSheth99, self).__init__(cosmo,
                                              mass_def,
                                              mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef('fof', 'matter')

    def _setup(self, cosmo):
        self.p = 0.3
        self.a = 0.707

    def _check_mdef_strict(self, mdef):
        if self.mass_def_strict:
            if mdef.Delta != 'fof':
                return True
        return False

    def _get_bsigma(self, cosmo, sigM, a):
        if self.use_delta_c_fit:
            status = 0
            delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
            check(status, cosmo=cosmo)
        else:
            delta_c = 1.68647

        nu = delta_c / sigM
        anu2 = self.a * nu**2
        return 1. + (anu2 - 1. + 2. * self.p / (1. + anu2**self.p))/delta_c
