from .concentration import Concentration
import numpy as np


class HaloProfile(object):
    name = 'default'

    def __init__(self):
        pass

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

    def profile_fourier(self, cosmo, r, M, a, mass_def=None):
        if getattr(self, '_profile_fourier', None):
            f_k = self._profile_fourier(cosmo, r, M, a, mass_def)
        elif getattr(self, '_profile_real', None):
            f_k = _profile_real_to_fourier(self._profile_real,  # noqa
                                           cosmo, r, M, a, mass_def)
        else:
            raise NotImplementedError("Profiles must have at least "
                                      " either a _profile_real or a "
                                      " _profile_fourier method.")
        return f_k


class HaloProfileNFW(object):
    def __init__(self, c_M_relation):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.cM = c_M_relation

    def _get_cM(self, cosmo, M, a, mdef=None):
        return self.cM.get_concentration(cosmo, M, a, mdef_other=mdef)

    def _norm(self, M, Rs, c):
        # NFW normalization from mass, radius and concentration
        return M / (4 * np.pi * Rs**3 * (np.log(1+c) - c/(1+c)))

    def _profile_real(self, cosmo, r, M, a, mass_def):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        R_M = mass_def.get_radius(cosmo, M_use, a)
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
