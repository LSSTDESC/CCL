__all__ = ("MassFuncWatson13",)

import numpy as np

from . import MassFunc


class MassFuncWatson13(MassFunc):
    """Implements the mass function of `Watson et al. 2013
    <https://arxiv.org/abs/1212.0095>`_. This parametrization accepts
    `fof` and any S.O. masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
            definition will be ignored.
    """
    name = 'Watson13'

    def __init__(self, *,
                 mass_def="200m",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name == "vir"

    def _get_fsigma_fof(self, cosmo, sigM, a, lnM):
        pA = 0.282
        pa = 2.163
        pb = 1.406
        pc = 1.210
        return pA * ((pb / sigM)**pa + 1.) * np.exp(-pc / sigM**2)

    def _get_fsigma_SO(self, cosmo, sigM, a, lnM):
        om = cosmo.omega_x(a, "matter")
        Delta_178 = self.mass_def.Delta / 178

        if a == 1:
            pA = 0.194
            pa = 1.805
            pb = 2.267
            pc = 1.287
        elif a < 1/(1+6):
            pA = 0.563
            pa = 3.810
            pb = 0.874
            pc = 1.453
        else:
            pA = om * (1.097 * a**3.216 + 0.074)
            pa = om * (5.907 * a**3.058 + 2.349)
            pb = om * (3.136 * a**3.599 + 2.344)
            pc = 1.318

        f_178 = pA * ((pb / sigM)**pa + 1.) * np.exp(-pc / sigM**2)
        C = np.exp(0.023 * (Delta_178 - 1.0))
        d = -0.456 * om - 0.139
        Gamma = (C * Delta_178**d *
                 np.exp(0.072 * (1.0 - Delta_178) / sigM**2.130))
        return f_178 * Gamma

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        if self.mass_def.name == 'fof':
            return self._get_fsigma_fof(cosmo, sigM, a, lnM)
        return self._get_fsigma_SO(cosmo, sigM, a, lnM)
