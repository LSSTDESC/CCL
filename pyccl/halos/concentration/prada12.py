from ...base import warn_api
from ..halo_model_base import Concentration
import numpy as np


__all__ = ("ConcentrationPrada12",)


class ConcentrationPrada12(Concentration):
    r"""Concentration-mass relation by Prada et al. (2012)
    :arXiv:1104.5130. Valid only for S.O. masses with :math:`\Delta = 200c`.

    The concentration takes the form

    .. math::

        c(M, z) &= B_0(x) \, \mathcal{C}(\sigma'), \\
        \sigma' &= B_1(x) \, \sigma(M, x), \\
        \mathcal{C}(\sigma') &= A \left[
            \left( \frac{\sigma'}{b} \right)^c + 1 \right]
            \exp \left( \frac{d}{\sigma'^2} \right),

    where :math:`(A,b,c,d) = (2.881, 1.257, 1.022, 0.060)`. The approximations
    for :math:`B_0(x)` and :math:`B_1(x)` are

     .. math::

         B_0(x) &= \frac{c_{\min}(x)}{c_{\min}(1.393)}, \\
         B_1(x) &= \frac{\sigma_{\min}^{-1}(x)}{\sigma_{\min}^{-1}(1.393)},

    where :math:`c_{\min}` and :math:`\sigma_{\min}^{-1}` define the minimum
    of the halo concentrations and the value of :math:`\sigma` at the minimum:

    .. math::

        c_{\min}(x) &= c_0 + (c_1 - c_0) \left[ \frac{1}{\pi}
        \arctan \left[ \alpha (x - x_0) \right] + \frac{1}{2} \right] \\
        \sigma_{\min}^{-1}(x) &= \sigma_0^{-1}
        + (\sigma_1^{-1} - \sigma_0^{-1}) \left[ \frac{1}{\pi}
        \arctan \left[ \beta (x - x_1) \right] + \frac{1}{2} \right],

    where :math:`(c_0, c_1, \alpha, x_0) = (3.681, 5.033, 6.948, 0.424)`
    and :math:`(\sigma_0^{-1}, \sigma_1^{-1}, \beta, x_1)
    = (1.047, 1.646, 7.386, 0.526)`.

    .. note::

        The mass definition for this concentration is fixed to :math:`200c`.

    Parameters
    ---------
    mass_def : :class:`~pyccl.halos.massdef.MassDef` or str, optional
        Mass definition for this :math:`c(M)` parametrization.
        The default is :math:`\Delta=200c`.
    """
    name = 'Prada12'

    @warn_api(pairs=[("mdef", "mass_def")])
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
