from ... import ccllib as lib
from ...base import warn_api
from ..massdef import MassDef
from ..halo_model_base import Concentration


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
    def __init__(self, *, mass_def=MassDef(200, 'critical')):
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name not in ["vir", "200m", "200c"]

    def _setup(self):
        vals = {("vir", "critical"): (7.7, 0.9, -0.29),
                (200, "matter"): (9.0, 1.15, -0.29),
                (200, "critical"): (5.9, 0.54, -0.35)}

        key = (self.mass_def.Delta, self.mass_def.rho_type)
        self.A, self.B, self.C = vals[key]

    def _concentration(self, cosmo, M, a):
        gz = cosmo.growth_factor(a)
        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
        sig = cosmo.sigmaM(M, a)
        nu = delta_c / sig
        return self.A * gz**self.B * nu**self.C
