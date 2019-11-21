from .. import ccllib as lib
from ..core import check
from ..pyutils import resample_array
from .concentration import Concentration
import numpy as np


class HaloProfile(object):
    name = 'default'

    def __init__(self):
        self.precision_fftlog = {'fac_lo': 0.1,
                                 'fac_hi': 10.,
                                 'n_per_decade': 1000,
                                 'extrapol': 'linx_liny',
                                 'epsilon': 0}

    def update_precision_fftlog(self, **kwargs):
        self.precision_fftlog.update(kwargs)

    def profile_real(self, cosmo, r, M, a, mass_def=None):
        if getattr(self, '_profile_real', None):
            f_r = self._profile_real(cosmo, r, M, a, mass_def)
        elif getattr(self, '_profile_fourier', None):
            f_r = _profile_fourier_to_real(self._profile_fourier,  # noqa
                                           cosmo, r, M, a, mass_def)
        else:
            raise NotImplementedError("Profiles must have at least "
                                      " either a _profile_real or a "
                                      " _profile_fourier method.")
        return f_r

    def profile_fourier(self, cosmo, k, M, a, mass_def=None):
        if getattr(self, '_profile_fourier', None):
            f_k = self._profile_fourier(cosmo, k, M, a, mass_def)
        elif getattr(self, '_profile_real', None):
            f_k = self._profile_real_to_fourier(cosmo, k, M, a, mass_def)
        else:
            raise NotImplementedError("Profiles must have at least "
                                      " either a _profile_real or a "
                                      " _profile_fourier method.")
        return f_k

    def _profile_real_to_fourier(self, cosmo, k, M, a, mass_def):
        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)
        lk_use = np.log(k_use)

        k_min = self.precision_fftlog['fac_lo'] * np.amin(k_use)
        k_max = self.precision_fftlog['fac_hi'] * np.amax(k_use)
        n_k = (int(np.log10(k_max / k_min)) *
               self.precision_fftlog['n_per_decade'])
        twopicubed = (2 * np.pi)**3
        r_arr = np.geomspace(k_min, k_max, n_k)

        p_k_out = np.zeros([M_use.size, k_use.size])
        for im, mass in enumerate(M_use):
            # Compute real profile values
            p_real = self._profile_real(cosmo, r_arr, mass, a, mass_def)

            # Compute Fourier profile through fftlog
            status = 0
            # TODO: we could probably benefit from precomputing all
            #       the FFTLog Gamma functions only once.
            epsilon = self.precision_fftlog['epsilon']
            result, status = lib.fftlog_transform(r_arr, p_real,
                                                  3, 0, epsilon,
                                                  2 * r_arr.size, status)
            check(status)
            k_arr, p_k_arr = result.reshape([2, r_arr.size])

            # Resample into input k values
            p_fourier = resample_array(np.log(k_arr), p_k_arr, lk_use,
                                       self.precision_fftlog['extrapol'],
                                       self.precision_fftlog['extrapol'],
                                       0, 0)
            p_k_out[im, :] = p_fourier * twopicubed

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

    def _profile_real(self, cosmo, r, M, a, mass_def):
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

    def _profile_real(self, cosmo, r, M, a, mass_def):
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
    def __init__(self, c_M_relation):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.cM = c_M_relation
        super(HaloProfileNFW, self).__init__()

    def _get_cM(self, cosmo, M, a, mdef=None):
        return self.cM.get_concentration(cosmo, M, a, mdef_other=mdef)

    def _norm(self, M, Rs, c):
        # NFW normalization from mass, radius and concentration
        return M / (4 * np.pi * Rs**3 * (np.log(1+c) - c/(1+c)))

    def _profile_real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self._get_cM(cosmo, M, a, mdef=mass_def)
        R_s = R_M / c_M

        x = r_use[:, None] / R_s[None, :]
        prof = 1./(x * (1 + x)**2)
        prof[r_use[:, None] > R_M[None, :]] = 0

        norm = self._norm(M_use, R_s, c_M)
        prof = prof[:, :] * norm[None, :]

        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
