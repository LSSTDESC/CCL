from .hmfunc import MassFunc
from .hbias import HaloBias
from .profiles import HaloProfile
from ..pk2d import Pk2D
from ..power import linear_matter_power, nonlin_matter_power
import numpy as np


class ProfileCovar(object):
    def __init__(self):
        pass

    def fourier_covar(self, prof, cosmo, k, M, a,
                      prof_2=None, mass_def=None):
        if not isinstance(prof, HaloProfile):
            raise TypeError("prof must be of type `HaloProfile`")
        uk1 = prof.fourier(cosmo, k, M, a, mass_def=mass_def)

        if prof_2 is None:
            uk2 = uk1
        else:
            if not isinstance(prof_2, HaloProfile):
                raise TypeError("prof_2 must be of type "
                                "`HaloProfile` or `None`")

            uk2 = prof_2.fourier(cosmo, k, M, a, mass_def=mass_def)

        return uk1 * uk2


class HMCalculator(object):
    def __init__(self, **kwargs):
        self.precision = {'l10M_min': 8.,
                          'l10M_max': 16.,
                          'nl10M': 128,
                          'integration_method_M': 'Simpson',
                          'k_min': 1E-5}
        self.precision.update(kwargs)
        self.lmass = np.linspace(self.precision['l10M_min'],
                                 self.precision['l10M_max'],
                                 self.precision['nl10M'])
        self.mass = 10.**self.lmass

        if self.precision['integration_method_M'] == 'Simpson':
            from scipy.integrate import simps
            self.integrator = simps
        else:
            raise NotImplementedError("Only \'Simpson\' supported "
                                      "as integration method")

    def _hmf(self, hmf, cosmo, a, mdef=None):
        return hmf.get_mass_function(cosmo, self.mass,
                                     a, mdef_other=mdef)

    def _hbf(self, hbf, cosmo, a, mdef=None):
        return hbf.get_halo_bias(cosmo, self.mass,
                                 a, mdef_other=mdef)

    def _u_k_from_arrays(self, hmf_a, uk_a):
        # TODO: M=0 limit
        return self.integrator(hmf_a[..., :] * uk_a,
                               self.lmass)

    def _b_k_from_arrays(self, hmf_a, hbf_a, uk_a):
        return self.integrator((hmf_a * hbf_a)[..., :] * uk_a,
                               self.lmass)

    def _check_massfunc(self, massfunc):
        if not isinstance(massfunc, MassFunc):
            raise TypeError("massfunc must be of type `MassFunc`")

    def _check_hbias(self, hbias):
        if not isinstance(hbias, HaloBias):
            raise TypeError("hbias must be of type `HaloBias`")

    def _check_prof(self, prof):
        if not isinstance(prof, HaloProfile):
            raise TypeError("prof must be of type `HaloProfile`")

    def mean_profile(self, cosmo, k, a, massfunc, prof,
                     normprof=False, mdef=None):
        a_use = np.atleast_1d(a)
        k_use = np.atleast_1d(k)

        # Check inputs
        self._check_massfunc(massfunc)
        self._check_prof(prof)

        na = len(a_use)
        nk = len(k_use)
        out = np.zeros([na, nk])
        for ia, aa in enumerate(a_use):
            mf = self._hmf(massfunc, cosmo, aa, mdef=mdef)
            uk = prof.fourier(cosmo, k_use, self.mass, aa,
                              mass_def=mdef)
            if normprof:
                uk0 = prof.fourier(cosmo,
                                   self.precision['k_min'],
                                   self.mass, aa,
                                   mass_def=mdef)
                norm = 1. / self._u_k_from_arrays(mf, uk0)
            else:
                norm = 1.
            out[ia, :] = self._u_k_from_arrays(mf, uk) * norm

        if np.ndim(a) == 0:
            out = np.squeeze(out, axis=0)
        if np.ndim(k) == 0:
            out = np.squeeze(out, axis=-1)
        return out

    def bias(self, cosmo, k, a, massfunc, hbias, prof,
             normprof=False, mdef=None):
        a_use = np.atleast_1d(a)
        k_use = np.atleast_1d(k)

        # Check inputs
        self._check_massfunc(massfunc)
        self._check_hbias(hbias)
        self._check_prof(prof)

        na = len(a_use)
        nk = len(k_use)
        out = np.zeros([na, nk])
        for ia, aa in enumerate(a_use):
            mf = self._hmf(massfunc, cosmo, aa, mdef=mdef)
            bf = self._hbf(hbias, cosmo, aa, mdef=mdef)
            uk = prof.fourier(cosmo, k_use, self.mass, aa,
                              mass_def=mdef)
            if normprof:
                uk0 = prof.fourier(cosmo,
                                   self.precision['k_min'],
                                   self.mass, aa,
                                   mass_def=mdef)
                norm = 1. / self._u_k_from_arrays(mf, uk0)
            else:
                norm = 1.
            out[ia, :] = self._b_k_from_arrays(mf, bf, uk) * norm

        if np.ndim(a) == 0:
            out = np.squeeze(out, axis=0)
        if np.ndim(k) == 0:
            out = np.squeeze(out, axis=-1)
        return out

    def pk(self, cosmo, k, a, massfunc, hbias, prof,
           covprof=None, prof_2=None, p_of_k_a=None,
           normprof=False, mdef=None, get_1h=True, get_2h=True):
        a_use = np.atleast_1d(a)
        k_use = np.atleast_1d(k)

        # Check inputs
        self._check_massfunc(massfunc)
        self._check_hbias(hbias)
        self._check_prof(prof)
        if (prof_2 is not None) and (not isinstance(prof_2, HaloProfile)):
            raise TypeError("prof_2 must be of type `HaloProfile` or `None`")
        if covprof is None:
            covprof = ProfileCovar()
        elif not isinstance(covprof, ProfileCovar):
            raise TypeError("covprof must be of type "
                            "`ProfileCovar` or `None`")
        # Power spectrum
        if isinstance(p_of_k_a, Pk2D):
            def pkf(sf): return p_of_k_a.eval(k_use, sf, cosmo)
        elif (p_of_k_a is None) or (p_of_k_a == 'linear'):
            def pkf(sf): return linear_matter_power(cosmo, k_use, sf)
        elif p_of_k_a == 'nonlinear':
            def pkf(sf): return nonlin_matter_power(cosmo, k_use, sf)
        else:
            raise TypeError("p_of_k_a must be `None`, \'linear\', "
                            "\'nonlinear\' or a `Pk2D` object")

        na = len(a_use)
        nk = len(k_use)
        out = np.zeros([na, nk])
        for ia, aa in enumerate(a_use):
            mf = self._hmf(massfunc, cosmo, aa, mdef=mdef)

            # Compute normalization
            if normprof:
                uk01 = prof.fourier(cosmo,
                                    self.precision['k_min'],
                                    self.mass, aa,
                                    mass_def=mdef)
                norm1 = 1. / self._u_k_from_arrays(mf, uk01)
            else:
                norm1 = 1.
            if prof_2 is None:
                norm2 = norm1
            else:
                if normprof:
                    uk02 = prof_2.fourier(cosmo,
                                          self.precision['k_min'],
                                          self.mass, aa,
                                          mass_def=mdef)
                    norm2 = 1. / self._u_k_from_arrays(mf, uk02)
                else:
                    norm2 = 1.
            norm = norm1 * norm2

            if get_2h:
                bf = self._hbf(hbias, cosmo, aa, mdef=mdef)
                # Compute first bias factor
                uk_1 = prof.fourier(cosmo, k_use, self.mass, aa,
                                    mass_def=mdef)
                bk_1 = self._b_k_from_arrays(mf, bf, uk_1)

                # Compute second bias factor
                if prof_2 is None:
                    bk_2 = bk_1
                else:
                    uk_2 = prof_2.fourier(cosmo, k_use, self.mass, aa,
                                          mass_def=mdef)
                    bk_2 = self._b_k_from_arrays(mf, bf, uk_2)

                # Compute power spectrum
                pk_2h = pkf(aa) * bk_1 * bk_2
            else:
                pk_2h = 0.

            if get_1h:
                # 1-halo term
                uk2 = covprof.fourier_covar(prof, cosmo, k_use,
                                            self.mass, aa,
                                            prof_2=prof_2,
                                            mass_def=mdef)
                pk_1h = self._u_k_from_arrays(mf, uk2)
            else:
                pk_1h = 0.

            # 2-halo term
            out[ia, :] = (pk_1h + pk_2h) * norm

        if np.ndim(a) == 0:
            out = np.squeeze(out, axis=0)
        if np.ndim(k) == 0:
            out = np.squeeze(out, axis=-1)
        return out
