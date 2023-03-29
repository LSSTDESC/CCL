from ...base import warn_api
from ..massdef import MassDef
from ..halo_model_base import Concentration
import numpy as np


__all__ = ("ConcentrationDiemer15",)


class ConcentrationDiemer15(Concentration):
    """ Concentration-mass relation by Diemer & Kravtsov 2015
    (arXiv:1407.4730). This parametrization is only valid for
    S.O. masses with Delta = 200-critical.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'Diemer15'

    @warn_api(pairs=[("mdef", "mass_def")])
    def __init__(self, *, mass_def=MassDef(200, 'critical')):
        super().__init__(mass_def=mass_def)

    def _setup(self):
        self.kappa = 1.0
        self.phi_0 = 6.58
        self.phi_1 = 1.27
        self.eta_0 = 7.28
        self.eta_1 = 1.56
        self.alpha = 1.08
        self.beta = 1.77

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name != "200c"

    def _concentration(self, cosmo, M, a):
        # Compute power spectrum slope
        R = cosmo.mass2radius_lagrangian(M)
        k_R = 2.0 * np.pi / R * self.kappa

        cosmo.compute_linear_power()
        pk = cosmo.get_linear_power()
        n = pk(k_R, a, derivative=True)

        sig = cosmo.sigmaM(M, a)
        delta_c = 1.68647
        nu = delta_c / sig

        floor = self.phi_0 + n * self.phi_1
        nu0 = self.eta_0 + n * self.eta_1
        c = 0.5 * floor * ((nu0 / nu)**self.alpha +
                           (nu / nu0)**self.beta)
        return c
