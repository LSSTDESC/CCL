from ..massdef import MassDef
from .concentration_base import Concentration


__all__ = ("ConcentrationKlypin11",)


class ConcentrationKlypin11(Concentration):
    """ Concentration-mass relation by Klypin et al. 2011
    (arXiv:1002.3660). This parametrization is only valid for
    S.O. masses with Delta = Delta_vir.

    Args:
        mdef (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'Klypin11'

    def __init__(self, mdef=None):
        super(ConcentrationKlypin11, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = MassDef('vir', 'critical')

    def _check_mdef(self, mdef):
        if mdef.Delta != 'vir':
            return True
        return False

    def _concentration(self, cosmo, M, a):
        M_pivot_inv = cosmo.cosmo.params.h * 1E-12
        return 9.6 * (M * M_pivot_inv)**-0.075
