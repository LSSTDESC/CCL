from ...base import warn_api
from ..massdef import MassDef
from ..halo_model_base import Concentration


__all__ = ("ConcentrationDuffy08",)


class ConcentrationDuffy08(Concentration):
    """ Concentration-mass relation by Duffy et al. 2008
    (arXiv:0804.2486). This parametrization is only valid for
    S.O. masses with Delta = Delta_vir, 200-matter and 200-critical.
    By default it will be initialized for Delta = 200-critical.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'Duffy08'

    @warn_api(pairs=[("mdef", "mass_def")])
    def __init__(self, *, mass_def=MassDef(200, 'critical')):
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name not in ["vir", "200m", "200c"]

    def _setup(self):
        vals = {("vir", "critical"): (7.85, -0.081, -0.71),
                (200, "matter"): (10.14, -0.081, -1.01),
                (200, "critical"): (5.71, -0.084, -0.47)}

        key = (self.mass_def.Delta, self.mass_def.rho_type)
        self.A, self.B, self.C = vals[key]

    def _concentration(self, cosmo, M, a):
        M_pivot_inv = cosmo["h"] * 5E-13
        return self.A * (M * M_pivot_inv)**self.B * a**(-self.C)
