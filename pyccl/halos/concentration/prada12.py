__all__ = ("ConcentrationPrada12",)

import numpy as np

from . import Concentration


class ConcentrationPrada12(Concentration):
    """Concentration-mass relation by `Prada et al. 2012
    <https://arXiv.org/abs/1104.5130>`_. This parametrization is only valid for
    S.O. masses with :math:`\\Delta = 200` times the critical density.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`): a mass
            definition object ora name string.
    """
    name = 'Prada12'

    def __init__(self, *, mass_def="200c"):
        super().__init__(mass_def=mass_def)

    def _check_mass_def_strict(self, mass_def):
        return mass_def.name != "200c"

    def _setup(self):
        self.c0 = 3.681
        self.c1 = 5.033
        self.al = 6.948
        self.x0 = 0.424
        self.i0 = 1.047
        self.i1 = 1.646
        self.be = 7.386
        self.x1 = 0.526
        self.cnorm = 1. / self._cmin(1.393)
        self.inorm = 1. / self._imin(1.393)

    def _form(self, x, x0, v0, v1, v2):
        # form factor for `cmin` and `imin`
        return v0 + (v1 - v0) * (np.arctan(v2 * (x - x0)) / np.pi + 0.5)

    def _cmin(self, x):
        return self._form(x, x0=self.x0, v0=self.c0, v1=self.c1, v2=self.al)

    def _imin(self, x):
        return self._form(x, x0=self.x1, v0=self.i0, v1=self.i1, v2=self.be)

    def _concentration(self, cosmo, M, a):
        sig = cosmo.sigmaM(M, a)
        x = a * (cosmo["Omega_l"] / cosmo["Omega_m"])**(1. / 3.)
        B0 = self._cmin(x) * self.cnorm
        B1 = self._imin(x) * self.inorm
        sig_p = B1 * sig
        Cc = 2.881 * ((sig_p / 1.257)**1.022 + 1) * np.exp(0.060 / sig_p**2)
        return B0 * Cc
