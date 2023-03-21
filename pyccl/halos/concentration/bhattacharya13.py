from ... import ccllib as lib
from ...base import warn_api
from ..massdef import MassDef
from .concentration_base import Concentration


__all__ = ("ConcentrationBhattacharya13",)


class ConcentrationBhattacharya13(Concentration):
    """ Concentration-mass relation by Bhattacharya et al. 2013
    (arXiv:1112.5479). This parametrization is only valid for
    S.O. masses with Delta = Delta_vir, 200-matter and 200-critical.
    By default it will be initialized for Delta = 200-critical.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'Bhattacharya13'

    @warn_api(pairs=[("mdef", "mass_def")])
    def __init__(self, *, mass_def=None):
        super(ConcentrationBhattacharya13, self).__init__(mass_def=mass_def)

    def _default_mass_def(self):
        self.mass_def = MassDef(200, 'critical')

    def _check_mass_def(self, mass_def):
        if mass_def.Delta != 'vir':
            if isinstance(mass_def.Delta, str):
                return True
            elif int(mass_def.Delta) != 200:
                return True
        return False

    def _setup(self):
        if self.mass_def.Delta == 'vir':
            self.A = 7.7
            self.B = 0.9
            self.C = -0.29
        else:  # Now Delta has to be 200
            if self.mass_def.rho_type == 'matter':
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
