from ... import ccllib as lib
from ..massdef import MassDef
from .concentration_base import Concentration


__all__ = ("ConcentrationBhattacharya13",)


class ConcentrationBhattacharya13(Concentration):
    """ Concentration-mass relation by Bhattacharya et al. 2013
    (arXiv:1112.5479). This parametrization is only valid for
    S.O. masses with Delta = Delta_vir, 200-matter and 200-critical.
    By default it will be initialized for Delta = 200-critical.

    Args:
        mdef (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'Bhattacharya13'

    def __init__(self, mdef=None):
        super(ConcentrationBhattacharya13, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = MassDef(200, 'critical')

    def _check_mdef(self, mdef):
        if mdef.Delta != 'vir':
            if isinstance(mdef.Delta, str):
                return True
            elif int(mdef.Delta) != 200:
                return True
        return False

    def _setup(self):
        if self.mdef.Delta == 'vir':
            self.A = 7.7
            self.B = 0.9
            self.C = -0.29
        else:  # Now Delta has to be 200
            if self.mdef.rho_type == 'matter':
                self.A = 9.0
                self.B = 1.15
                self.C = -0.29
            else:  # Now rho_type has to be critical
                self.A = 5.9
                self.B = 0.54
                self.C = -0.35

    def _concentration(self, cosmo, M, a):
        gz = cosmo.growth_factor(a)
        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
        sig = cosmo.sigmaM(M, a)
        nu = delta_c / sig
        return self.A * gz**self.B * nu**self.C
