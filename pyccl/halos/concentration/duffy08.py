from ..massdef import MassDef
from .concentration_base import Concentration


class ConcentrationDuffy08(Concentration):
    """ Concentration-mass relation by Duffy et al. 2008
    (arXiv:0804.2486). This parametrization is only valid for
    S.O. masses with Delta = Delta_vir, 200-matter and 200-critical.
    By default it will be initialized for Delta = 200-critical.

    Args:
        mdef (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'Duffy08'

    def __init__(self, mdef=None):
        super(ConcentrationDuffy08, self).__init__(mdef)

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
            self.A = 7.85
            self.B = -0.081
            self.C = -0.71
        else:  # Now Delta has to be 200
            if self.mdef.rho_type == 'matter':
                self.A = 10.14
                self.B = -0.081
                self.C = -1.01
            else:  # Now rho_type has to be critical
                self.A = 5.71
                self.B = -0.084
                self.C = -0.47

    def _concentration(self, cosmo, M, a):
        M_pivot_inv = cosmo.cosmo.params.h * 5E-13
        return self.A * (M * M_pivot_inv)**self.B * a**(-self.C)
