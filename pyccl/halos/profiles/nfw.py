__all__ = ("HaloProfileNFW",)

import numpy as np
from scipy.special import sici

from . import HaloProfileMatter


class HaloProfileNFW(HaloProfileMatter):
    """`Navarro-Frenk-White
    <https://arxiv.org/abs/astro-ph/9508025>`_ profile.

    .. math::
       \\rho(r) = \\frac{\\rho_0}
       {\\frac{r}{r_s}\\left(1+\\frac{r}{r_s}\\right)^2}

    where :math:`r_s` is related to the comoving spherical overdensity
    halo radius :math:`r_\\Delta(M)` through the concentration
    parameter :math:`c(M)` as

    .. math::
       r_\\Delta(M) = c(M)\\,r_s

    and the normalization :math:`\\rho_0` is

    .. math::
       \\rho_0 = \\frac{M}{4\\pi\\,r_s^3\\,[\\log(1+c) - c/(1+c)]}

    By default, this profile is truncated at :math:`r = r_\\Delta(M)`.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        concentration (:obj:`~pyccl.halos.halo_model_base.Concentration`):
            concentration-mass relation to use with this profile.
        fourier_analytic (:obj:`bool`): set to ``True`` if you want to compute
            the Fourier profile analytically (and not through FFTLog).
        projected_analytic (:obj:`bool`): set to ``True`` if you want to
            compute the 2D projected profile analytically (and not
            through FFTLog).
        cumul2d_analytic (:obj:`bool`): set to ``True`` if you want to
            compute the 2D cumulative surface density analytically
            (and not through FFTLog).
        truncated (:obj:`bool`): set to ``True`` if the profile should be
            truncated at :math:`r = r_\\Delta`.
    """
    __repr_attrs__ = __eq_attrs__ = (
        "fourier_analytic", "projected_analytic", "cumul2d_analytic",
        "truncated", "mass_def", "concentration", "precision_fftlog",)

    def __init__(self, *, mass_def, concentration,
                 fourier_analytic=True,
                 projected_analytic=False,
                 cumul2d_analytic=False,
                 truncated=True):
        self.truncated = truncated
        self.fourier_analytic = fourier_analytic
        self.projected_analytic = projected_analytic
        self.cumul2d_analytic = cumul2d_analytic
        if fourier_analytic:
            self._fourier = self._fourier_analytic
        if projected_analytic:
            if truncated:
                raise ValueError("Analytic projected profile not supported "
                                 "for truncated NFW. Set `truncated` or "
                                 "`projected_analytic` to `False`.")
            self._projected = self._projected_analytic
        if cumul2d_analytic:
            if truncated:
                raise ValueError("Analytic cumuative 2d profile not supported "
                                 "for truncated NFW. Set `truncated` or "
                                 "`cumul2d_analytic` to `False`.")
            self._cumul2d = self._cumul2d_analytic
        self._omln2 = 1 - np.log(2)
        super().__init__(mass_def=mass_def, concentration=concentration)
        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-2,
                                     n_per_decade=1000,
                                     plaw_fourier=-2.)

    def _norm(self, M, Rs, c):
        # NFW normalization from mass, radius and concentration
        return M / (4 * np.pi * Rs**3 * (np.log(1+c) - c/(1+c)))

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M

        x = r_use[None, :] / R_s[:, None]
        prof = 1./(x * (1 + x)**2)
        if self.truncated:
            prof[r_use[None, :] > R_M[:, None]] = 0

        norm = self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fx_projected(self, x):

        def f1(xx):
            x2m1 = xx * xx - 1
            sqx2m1 = np.sqrt(-x2m1)
            return 1 / x2m1 + np.arcsinh(sqx2m1 / xx) / (-x2m1)**1.5

        def f2(xx):
            x2m1 = xx * xx - 1
            sqx2m1 = np.sqrt(x2m1)
            return 1 / x2m1 - np.arcsin(sqx2m1 / xx) / x2m1**1.5

        xf = x.flatten()
        return np.piecewise(xf,
                            [xf < 1, xf > 1],
                            [f1, f2, 1./3.]).reshape(x.shape)

    def _projected_analytic(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M

        x = r_use[None, :] / R_s[:, None]
        prof = self._fx_projected(x)
        norm = 2 * R_s * self._norm(M_use, R_s, c_M)
        prof *= norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fx_cumul2d(self, x):

        def f1(xx):
            sqx2m1 = np.sqrt(-(xx * xx - 1))
            return np.log(0.5 * xx) + np.arcsinh(sqx2m1 / xx) / sqx2m1

        def f2(xx):
            sqx2m1 = np.sqrt(xx * xx - 1)
            return np.log(0.5 * xx) + np.arcsin(sqx2m1 / xx) / sqx2m1

        xf = x.flatten()
        f = np.piecewise(xf,
                         [xf < 1, xf > 1],
                         [f1, f2, self._omln2]).reshape(x.shape)
        return 2 * f / x**2

    def _cumul2d_analytic(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M

        x = r_use[None, :] / R_s[:, None]
        prof = self._fx_cumul2d(x)
        norm = 2 * R_s * self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_analytic(self, cosmo, k, M, a):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M

        x = k_use[None, :] * R_s[:, None]
        Si2, Ci2 = sici(x)
        P1 = M_use / (np.log(1 + c_M) - c_M / (1 + c_M))
        if self.truncated:
            Si1, Ci1 = sici((1 + c_M[:, None]) * x)
            P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
            P3 = np.sin(c_M[:, None] * x) / ((1 + c_M[:, None]) * x)
            prof = P1[:, None] * (P2 - P3)
        else:
            P2 = np.sin(x) * (0.5 * np.pi - Si2) - np.cos(x) * Ci2
            prof = P1[:, None] * P2

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
