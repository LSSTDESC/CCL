from .. import ccllib as lib
from ..core import check
from ..power import sigmaM
from ..pyutils import resample_array
from .concentration import Concentration
from .massdef import MassDef
import numpy as np
from scipy.special import sici


class HaloProfile(object):
    name = 'default'

    def __init__(self):
        self.precision_fftlog = {'padding_lo_fftlog': 0.1,
                                 'padding_lo_extra': 0.1,
                                 'padding_hi_fftlog': 10.,
                                 'padding_hi_extra': 10.,
                                 'large_padding_2D': False,
                                 'n_per_decade': 1000,
                                 'extrapol': 'linx_liny'}

    def update_precision_fftlog(self, **kwargs):
        self.precision_fftlog.update(kwargs)

    def _get_plaw_fourier(self, cosmo, M, a, mass_def):
        return -1.5

    def _get_plaw_projected(self, cosmo, M, a, mass_def):
        return -1.

    def real(self, cosmo, r, M, a, mass_def=None):
        if getattr(self, '_real', None):
            f_r = self._real(cosmo, r, M, a, mass_def)
        elif getattr(self, '_fourier', None):
            f_r = self._fftlog_wrap(cosmo, r, M, a, mass_def,
                                    fourier_out=False)
        else:
            raise NotImplementedError("Profiles must have at least "
                                      " either a _real or a "
                                      " _fourier method.")
        return f_r

    def fourier(self, cosmo, k, M, a, mass_def=None):
        if getattr(self, '_fourier', None):
            f_k = self._fourier(cosmo, k, M, a, mass_def)
        elif getattr(self, '_real', None):
            f_k = self._fftlog_wrap(cosmo, k, M, a, mass_def,
                                    fourier_out=True)
        else:
            raise NotImplementedError("Profiles must have at least "
                                      " either a _real or a "
                                      " _fourier method.")
        return f_k

    def projected(self, cosmo, r_t, M, a, mass_def=None):
        if getattr(self, '_projected', None):
            s_r_t = self._projected(cosmo, r_t, M, a, mass_def)
        else:
            s_r_t = self._projected_fftlog_wrap(cosmo, r_t, M,
                                                a, mass_def)
        return s_r_t

    def cumul2d(self, cosmo, r_t, M, a, mass_def=None):
        if getattr(self, '_cumul2d', None):
            s_r_t = self._cumul2d(cosmo, r_t, M, a, mass_def)
        else:
            s_r_t = self._cumul2d_fftlog_wrap(cosmo, r_t, M,
                                              a, mass_def)
        return s_r_t

    def _cumul2d_fftlog_wrap(self, cosmo, r_t, M, a, mass_def):
        r_t_use = np.atleast_1d(r_t)
        M_use = np.atleast_1d(M)
        lr_t_use = np.log(r_t_use)
        r_t_min = self.precision_fftlog['padding_lo_fftlog'] * np.amin(r_t_use)
        r_t_max = self.precision_fftlog['padding_hi_fftlog'] * np.amax(r_t_use)
        n_r_t = (int(np.log10(r_t_max / r_t_min)) *
                 self.precision_fftlog['n_per_decade'])

        k_arr = np.geomspace(r_t_min, r_t_max, n_r_t)
        sig_r_t_out = np.zeros([M_use.size, r_t_use.size])
        for im, mass in enumerate(M_use):
            if getattr(self, '_fourier', None):
                p_fourier = self._fourier(cosmo, k_arr, mass,
                                          a, mass_def)
            else:
                lpad = self.precision_fftlog['large_padding_2D']
                p_fourier = self._fftlog_wrap(cosmo,
                                              k_arr,
                                              mass, a,
                                              mass_def,
                                              fourier_out=True,
                                              large_padding=lpad)
            p_fourier *= 2 / k_arr

            status = 0
            plaw_index = self._get_plaw_projected(cosmo, mass,
                                                  a, mass_def) - 1
            result, status = lib.fftlog_transform(k_arr, p_fourier,
                                                  2, 1, plaw_index,
                                                  2 * k_arr.size, status)
            check(status)
            r_t_arr, sig_r_t_arr = result.reshape([2, k_arr.size])
            sig_r_t_arr /= r_t_arr

            sig_r_t = resample_array(np.log(r_t_arr), sig_r_t_arr,
                                     lr_t_use,
                                     self.precision_fftlog['extrapol'],
                                     self.precision_fftlog['extrapol'],
                                     0, 0)
            sig_r_t_out[im, :] = sig_r_t
        sig_r_t_out = sig_r_t_out.T

        if np.ndim(M) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=-1)
        if np.ndim(r_t) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=0)
        return sig_r_t_out

    def _projected_fftlog_wrap(self, cosmo, r_t, M, a, mass_def):
        r_t_use = np.atleast_1d(r_t)
        M_use = np.atleast_1d(M)
        lr_t_use = np.log(r_t_use)
        r_t_min = self.precision_fftlog['padding_lo_fftlog'] * np.amin(r_t_use)
        r_t_max = self.precision_fftlog['padding_hi_fftlog'] * np.amax(r_t_use)
        n_r_t = (int(np.log10(r_t_max / r_t_min)) *
                 self.precision_fftlog['n_per_decade'])

        k_arr = np.geomspace(r_t_min, r_t_max, n_r_t)
        sig_r_t_out = np.zeros([M_use.size, r_t_use.size])
        for im, mass in enumerate(M_use):
            if getattr(self, '_fourier', None):
                p_fourier = self._fourier(cosmo, k_arr, mass,
                                          a, mass_def)
            else:
                lpad = self.precision_fftlog['large_padding_2D']
                p_fourier = self._fftlog_wrap(cosmo,
                                              k_arr,
                                              mass, a,
                                              mass_def,
                                              fourier_out=True,
                                              large_padding=lpad)

            status = 0
            plaw_index = self._get_plaw_projected(cosmo, mass,
                                                  a, mass_def)
            result, status = lib.fftlog_transform(k_arr, p_fourier,
                                                  2, 0, plaw_index,
                                                  2 * k_arr.size, status)
            check(status)
            r_t_arr, sig_r_t_arr = result.reshape([2, k_arr.size])

            sig_r_t = resample_array(np.log(r_t_arr), sig_r_t_arr,
                                     lr_t_use,
                                     self.precision_fftlog['extrapol'],
                                     self.precision_fftlog['extrapol'],
                                     0, 0)
            sig_r_t_out[im, :] = sig_r_t
        sig_r_t_out = sig_r_t_out.T

        if np.ndim(M) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=-1)
        if np.ndim(r_t) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=0)
        return sig_r_t_out

    def _fftlog_wrap(self, cosmo, k, M, a, mass_def,
                     fourier_out=False,
                     large_padding=True):
        if fourier_out:
            p_func = self._real
        else:
            p_func = self._fourier
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)
        lk_use = np.log(k_use)

        if large_padding:
            k_min = self.precision_fftlog['padding_lo_fftlog'] * np.amin(k_use)
            k_max = self.precision_fftlog['padding_hi_fftlog'] * np.amax(k_use)
        else:
            k_min = self.precision_fftlog['padding_lo_extra'] * np.amin(k_use)
            k_max = self.precision_fftlog['padding_hi_extra'] * np.amax(k_use)

        n_k = (int(np.log10(k_max / k_min)) *
               self.precision_fftlog['n_per_decade'])
        r_arr = np.geomspace(k_min, k_max, n_k)

        p_k_out = np.zeros([M_use.size, k_use.size])
        for im, mass in enumerate(M_use):
            # Compute real profile values
            p_real = p_func(cosmo, r_arr, mass, a, mass_def)

            # Compute Fourier profile through fftlog
            status = 0
            # TODO: we could probably benefit from precomputing all
            #       the FFTLog Gamma functions only once.
            plaw_index = self._get_plaw_fourier(cosmo, mass,
                                                a, mass_def)
            result, status = lib.fftlog_transform(r_arr, p_real,
                                                  3, 0, plaw_index,
                                                  2 * r_arr.size, status)
            check(status)
            k_arr, p_k_arr = result.reshape([2, r_arr.size])

            # Resample into input k values
            p_fourier = resample_array(np.log(k_arr), p_k_arr, lk_use,
                                       self.precision_fftlog['extrapol'],
                                       self.precision_fftlog['extrapol'],
                                       0, 0)
            p_k_out[im, :] = p_fourier

        if fourier_out:
            p_k_out *= (2 * np.pi)**3

        p_k_out = p_k_out.T
        if np.ndim(M) == 0:
            p_k_out = np.squeeze(p_k_out, axis=-1)
        if np.ndim(k) == 0:
            p_k_out = np.squeeze(p_k_out, axis=0)
        return p_k_out


class HaloProfileGaussian(HaloProfile):
    def __init__(self, r_scale, rho0):
        self.rho_0 = rho0
        self.r_s = r_scale
        super(HaloProfileGaussian, self).__init__()
        self.update_precision_fftlog(padding_lo_fftlog=0.01,
                                     padding_hi_fftlog=100.,
                                     n_per_decade=10000)

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Compute scale
        rs = self.r_s(cosmo, M_use, a, mass_def)
        # Compute normalization
        rho0 = self.rho_0(cosmo, M_use, a, mass_def)
        # Form factor
        prof = np.exp(-(r_use[:, None] / rs[None, :])**2)
        prof = prof * rho0[None, :]

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfilePowerLaw(HaloProfile):
    def __init__(self, r_scale, tilt):
        self.r_s = r_scale
        self.tilt = tilt
        super(HaloProfilePowerLaw, self).__init__()

    def _get_plaw_fourier(self, cosmo, M, a, mass_def):
        return self.tilt(cosmo, M, a, mass_def)

    def _get_plaw_projected(self, cosmo, M, a, mass_def):
        return -3 - self.tilt(cosmo, M, a, mass_def)

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Compute scale
        rs = self.r_s(cosmo, M_use, a, mass_def)
        tilt = self.tilt(cosmo, M_use, a, mass_def)
        # Form factor
        prof = (r_use[:, None] / rs[None, :])**tilt[None, :]

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileNFW(HaloProfile):
    def __init__(self, c_M_relation, fourier_analytic=False,
                 projected_analytic=False,
                 cumul2d_analytic=False,
                 truncated=True):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.cM = c_M_relation
        self.truncated = truncated
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
        super(HaloProfileNFW, self).__init__()
        self.update_precision_fftlog(padding_hi_fftlog=2E4,
                                     padding_lo_fftlog=1E-2)

    def _get_cM(self, cosmo, M, a, mdef=None):
        return self.cM.get_concentration(cosmo, M, a, mdef_other=mdef)

    def _norm(self, M, Rs, c):
        # NFW normalization from mass, radius and concentration
        return M / (4 * np.pi * Rs**3 * (np.log(1+c) - c/(1+c)))

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        x = r_use[:, None] / R_s[None, :]
        prof = 1./(x * (1 + x)**2)
        if self.truncated:
            prof[r_use[:, None] > R_M[None, :]] = 0

        norm = self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[None, :]

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fx_projected(self, x):

        def f1(xx):
            x2m1 = xx * xx - 1
            return 1 / x2m1 + np.arccosh(1 / xx) / np.fabs(x2m1)**1.5

        def f2(xx):
            x2m1 = xx * xx - 1
            return 1 / x2m1 - np.arccos(1 / xx) / np.fabs(x2m1)**1.5

        xf = x.flatten()
        return np.piecewise(xf,
                            [xf < 1, xf > 1],
                            [f1, f2, 1./3.]).reshape(x.shape)

    def _projected_analytic(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        x = r_use[:, None] / R_s[None, :]
        prof = self._fx_projected(x)
        norm = 2 * R_s * self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[None, :]

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fx_cumul2d(self, x):

        def f1(xx):
            sqx2m1 = np.sqrt(np.fabs(xx * xx - 1))
            return np.log(0.5 * xx) + np.arccosh(1 / xx) / sqx2m1

        def f2(xx):
            sqx2m1 = np.sqrt(np.fabs(xx * xx - 1))
            return np.log(0.5 * xx) + np.arccos(1 / xx) / sqx2m1

        xf = x.flatten()
        omln2 = 0.3068528194400547  # 1-Log[2]
        f = np.piecewise(xf,
                         [xf < 1, xf > 1],
                         [f1, f2, omln2]).reshape(x.shape)
        return 2 * f / x**2

    def _cumul2d_analytic(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        x = r_use[:, None] / R_s[None, :]
        prof = self._fx_cumul2d(x)
        norm = 2 * R_s * self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[None, :]

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_analytic(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        x = k_use[:, None] * R_s[None, :]
        Si2, Ci2 = sici(x)
        P1 = M / (np.log(1 + c_M) - c_M / (1 + c_M))
        if self.truncated:
            Si1, Ci1 = sici((1 + c_M[None, :]) * x)
            P2 = np.sin(x) * (Si1 - Si2) + np.cos(x) * (Ci1 - Ci2)
            P3 = np.sin(c_M[None, :] * x) / ((1 + c_M[None, :]) * x)
            prof = P1[None, :] * (P2 - P3)
        else:
            P2 = np.sin(x) * (0.5 * np.pi - Si2) - np.cos(x) * Ci2
            prof = P1[None, :] * P2

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileEinasto(HaloProfile):
    def __init__(self, c_M_relation, truncated=True):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.cM = c_M_relation
        self.truncated = truncated
        super(HaloProfileEinasto, self).__init__()
        self.update_precision_fftlog(padding_hi_fftlog=2E4,
                                     padding_lo_fftlog=1E-2)

    def _get_cM(self, cosmo, M, a, mdef=None):
        return self.cM.get_concentration(cosmo, M, a, mdef_other=mdef)

    def _get_alpha(self, cosmo, M, a, mdef):
        mdef_vir = MassDef('vir', 'matter')
        Mvir = mdef.translate_mass(cosmo, M, a, mdef_vir)
        sM = sigmaM(cosmo, Mvir, a)
        nu = 1.686 / sM
        alpha = 0.155 + 0.0095 * nu * nu
        return alpha

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        alpha = self._get_alpha(cosmo, M_use, a, mass_def)

        status = 0
        norm, status = lib.einasto_norm(R_s, R_M, alpha, M_use.size, status)
        check(status)
        norm = M_use / norm

        x = r_use[:, None] / R_s[None, :]
        prof = norm[None, :] * np.exp(-2. * (x**alpha[None, :] - 1) /
                                      alpha[None, :])
        if self.truncated:
            prof[r_use[:, None] > R_M[None, :]] = 0

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class HaloProfileHernquist(HaloProfile):
    def __init__(self, c_M_relation, truncated=True):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.cM = c_M_relation
        self.truncated = truncated
        super(HaloProfileHernquist, self).__init__()
        self.update_precision_fftlog(padding_hi_fftlog=2E4,
                                     padding_lo_fftlog=1E-2)

    def _get_cM(self, cosmo, M, a, mdef=None):
        return self.cM.get_concentration(cosmo, M, a, mdef_other=mdef)

    def _real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M_use, a, mdef=mass_def)
        R_s = R_M / c_M

        status = 0
        norm, status = lib.hernquist_norm(R_s, R_M, M_use.size, status)
        check(status)
        norm = M_use / norm

        x = r_use[:, None] / R_s[None, :]
        prof = norm[None, :] / (x * (1 + x)**3)
        if self.truncated:
            prof[r_use[:, None] > R_M[None, :]] = 0

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
