from ...base import warn_api
from ..massdef import MassDef
from .concentration_base import Concentration


__all__ = ("ConcentrationKlypin11",)


class ConcentrationKlypin11(Concentration):
    """ Concentration-mass relation by Klypin et al. 2011
    (arXiv:1002.3660). This parametrization is only valid for
    S.O. masses with Delta = Delta_vir.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'Klypin11'

    @warn_api(pairs=[("mdef", "mass_def")])
    def __init__(self, *, mass_def=None):
        super(ConcentrationKlypin11, self).__init__(mass_def=mass_def)

    def _default_mass_def(self):
        self.mass_def = MassDef('vir', 'critical')

    def _check_mass_def(self, mass_def):
        if mass_def.Delta != 'vir':
            return True
        return False

    def _concentration(self, cosmo, M, a):
        M_pivot_inv = cosmo.cosmo.params.h * 1E-12
        return 9.6 * (M * M_pivot_inv)**-0.075
