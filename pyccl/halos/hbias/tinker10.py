__all__ = ("HaloBiasTinker10",)

import numpy as np

from . import HaloBias, get_delta_c


class HaloBiasTinker10(HaloBias):
    """Implements halo bias as described in `Tinker et al. 2010
    <https://arxiv.org/abs/1001.3162>`_. This parametrization accepts S.O.
    masses with :math:`200 < \\Delta < 3200`, defined with respect to the
    matter density. This can be automatically translated to S.O. masses
    defined with respect to the critical density.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
            definition will be ignored.
    """
    name = "Tinker10"

    def __init__(self, *,
                 mass_def="200m",
                 mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _check_mass_def_strict(self, mass_def):
        # True for FoF since Tinker10 is not defined for this mass def.
        return mass_def.Delta == "fof"

    def _setup(self):
        self.B = 0.183
        self.b = 1.5
        self.c = 2.4
        self.dc = get_delta_c(None, None, kind='EdS')

    def _get_bsigma(self, cosmo, sigM, a):
        nu = self.dc / sigM
        ld = np.log10(self.mass_def._get_Delta_m(cosmo, a))
        xp = np.exp(-(4./ld)**4.)
        A = 1.0 + 0.24 * ld * xp
        C = 0.019 + 0.107 * ld + 0.19*xp
        aa = 0.44 * ld - 0.88
        nupa = nu**aa
        return 1 - A * nupa / (nupa + self.dc**aa) + (
            self.B * nu**self.b + C * nu**self.c)
