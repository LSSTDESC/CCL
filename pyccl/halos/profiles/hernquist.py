from ...base import warn_api
from ..concentration import Concentration
from .profile_base import HaloProfile
import numpy as np
from scipy.special import sici


__all__ = ("HaloProfileHernquist",)


class HaloProfileHernquist(HaloProfile):
    """ Hernquist (1990ApJ...356..359H).

    .. math::
       \\rho(r) = \\frac{\\rho_0}
       {\\frac{r}{r_s}\\left(1+\\frac{r}{r_s}\\right)^3}

    where :math:`r_s` is related to the spherical overdensity
    halo radius :math:`R_\\Delta(M)` through the concentration
    parameter :math:`c(M)` as

    .. math::
       R_\\Delta(M) = c(M)\\,r_s

    and the normalization :math:`\\rho_0` is the mean density
    within the :math:`R_\\Delta(M)` of the halo.

    By default, this profile is truncated at :math:`r = R_\\Delta(M)`.

    Args:
        c_m_relation (:obj:`Concentration`): concentration-mass
            relation to use with this profile.
        fourier_analytic (bool): set to `True` if you want to compute
            the Fourier profile analytically (and not through FFTLog).
            Default: `False`.
        projected_analytic (bool): set to `True` if you want to
            compute the 2D projected profile analytically (and not
            through FFTLog). Default: `False`.
        cumul2d_analytic (bool): set to `True` if you want to
            compute the 2D cumulative surface density analytically
            (and not through FFTLog). Default: `False`.
        truncated (bool): set to `True` if the profile should be
            truncated at :math:`r = R_\\Delta` (i.e. zero at larger
            radii.
    """
    __repr_attrs__ = ("c_m_relation", "fourier_analytic", "projected_analytic",
                      "cumul2d_analytic", "truncated", "precision_fftlog",)
    name = 'Hernquist'

    @warn_api(pairs=[("c_M_relation", "c_m_relation")])
    def __init__(self, *, c_m_relation,
                 truncated=True,
                 fourier_analytic=False,
                 projected_analytic=False,
                 cumul2d_analytic=False):
        if not isinstance(c_m_relation, Concentration):
            raise TypeError("c_m_relation must be of type `Concentration`")

        self.c_m_relation = c_m_relation
        self.truncated = truncated
        self.fourier_analytic = fourier_analytic
        self.projected_analytic = projected_analytic
        self.cumul2d_analytic = cumul2d_analytic
        if fourier_analytic:
            self._fourier = self._fourier_analytic
        if projected_analytic:
            if truncated:
                raise ValueError("Analytic projected profile not supported "
                                 "for truncated Hernquist. Set `truncated` or "
                                 "`projected_analytic` to `False`.")
            self._projected = self._projected_analytic
        if cumul2d_analytic:
            if truncated:
                raise ValueError("Analytic cumuative 2d profile not supported "
                                 "for truncated Hernquist. Set `truncated` or "
                                 "`cumul2d_analytic` to `False`.")
            self._cumul2d = self._cumul2d_analytic
        super(HaloProfileHernquist, self).__init__()
        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-4,
                                     n_per_decade=1000,
                                     plaw_fourier=-2.)

    def _get_c_m_relation(self, cosmo, M, a, mass_def=None):
        return self.c_m_relation.get_concentration(cosmo, M, a,
                                                   mass_def_other=mass_def)

    def _norm(self, M, Rs, c):
        # Hernquist normalization from mass, radius and concentration
        return M / (2 * np.pi * Rs**3 * (c / (1 + c))**2)

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_c_m_relation(cosmo, M_use, a, mass_def=mass_def)
        R_s = R_M / c_M

        norm = self._norm(M_use, R_s, c_M)

        x = r_use[None, :] / R_s[:, None]
        prof = norm[:, None] / (x * (1 + x)**3)
        if self.truncated:
            prof[r_use[None, :] > R_M[:, None]] = 0

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fx_projected(self, x):

        def f1(xx):
            x2m1 = xx * xx - 1
            return (-3 / 2 / x2m1**2
                    + (x2m1+3) * np.arccosh(1 / xx) / 2 / np.fabs(x2m1)**2.5)

        def f2(xx):
            x2m1 = xx * xx - 1
            return (-3 / 2 / x2m1**2
                    + (x2m1+3) * np.arccos(1 / xx) / 2 / np.fabs(x2m1)**2.5)

        xf = x.flatten()
        return np.piecewise(xf,
                            [xf < 1, xf > 1],
                            [f1, f2, 2./15.]).reshape(x.shape)

    def _projected_analytic(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_c_m_relation(cosmo, M_use, a, mass_def=mass_def)
        R_s = R_M / c_M

        x = r_use[None, :] / R_s[:, None]
        prof = self._fx_projected(x)
        norm = 2 * R_s * self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fx_cumul2d(self, x):

        def f1(xx):
            x2m1 = xx * xx - 1
            return (1 + 1 / x2m1
                    + (x2m1 + 1) * np.arccosh(1 / xx) / np.fabs(x2m1)**1.5)

        def f2(xx):
            x2m1 = xx * xx - 1
            return (1 + 1 / x2m1
                    - (x2m1 + 1) * np.arccos(1 / xx) / np.fabs(x2m1)**1.5)

        xf = x.flatten()
        f = np.piecewise(xf,
                         [xf < 1, xf > 1],
                         [f1, f2, 1./3.]).reshape(x.shape)

        return f / x**2

    def _cumul2d_analytic(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_c_m_relation(cosmo, M_use, a, mass_def=mass_def)
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

    def _fourier_analytic(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_c_m_relation(cosmo, M_use, a, mass_def=mass_def)
        R_s = R_M / c_M

        x = k_use[None, :] * R_s[:, None]
        Si2, Ci2 = sici(x)
        P1 = M / ((c_M / (c_M + 1))**2 / 2)
        c_Mp1 = c_M[:, None] + 1
        if self.truncated:
            Si1, Ci1 = sici(c_Mp1 * x)
            P2 = x * np.sin(x) * (Ci1 - Ci2) - x * np.cos(x) * (Si1 - Si2)
            P3 = (-1 + np.sin(c_M[:, None] * x) / (c_Mp1**2 * x)
                  + c_Mp1 * np.cos(c_M[:, None] * x) / (c_Mp1**2))
            prof = P1[:, None] * (P2 - P3) / 2
        else:
            P2 = (-x * (2 * np.sin(x) * Ci2 + np.pi * np.cos(x))
                  + 2 * x * np.cos(x) * Si2 + 2) / 4
            prof = P1[:, None] * P2

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
