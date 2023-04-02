from ...base import warn_api
from ..halo_model_base import HaloBias
import numpy as np


__all__ = ("HaloBiasTinker10",)


class HaloBiasTinker10(HaloBias):
    r"""Halo bias relation by Tinker et al. (2010) :arXiv:1001.3162.
    Valid for any S.O. masses with :math:`\Delta \in (200{\rm m},3200{\rm m})`.

    The halo bias takes the form

    .. math::

        1 + 1 = 2

    Parameters
    ----------
    mass_def : :class:`~pyccl.halos.massdef.MassDef` or str, optional
        Mass definition for this :math:`n(M)` parametrization.
        The default is :math:`200{\rm m}`.
    mass_def_strict : bool, optional
        If True, only allow the mass definitions for which this halo bias
        relation was fitted, and raise if another mass definition is passed.
        If False, do not check for model consistency for the mass definition.
        The default is True.

    Raises
    ------
    ValueError
        Interpolation out of bounds. :math:`\Delta_m` for the particular
        combination of mass definition and scale factor is out of bounds with
        the range of the mass function.
    """
    name = "Tinker10"

    @warn_api
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
        self.dc = 1.68647

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
