

class HaloProfile(object):
    name = 'default'

    def __init__(self):
        pass

    def profile_real(self, cosmo, r, M, a):
        if getattr(self, '_profile_real', None):
            f_r = self._profile_real(cosmo, r, M, a)
        elif getattr(self, '_profile_fourier', None):
            f_r = _profile_fourier_to_real(self._profile_fourier,  # noqa
                                           cosmo, r, M, a)
        else:
            raise NotImplementedError("Profiles must have at least "
                                      " either a _profile_real or a "
                                      " _profile_fourier method.")
        norm = self.norm(cosmo, M, a)
        return norm * f_r

    def profile_fourier(self, cosmo, r, M, a):
        if getattr(self, '_profile_fourier', None):
            f_k = self._profile_fourier(cosmo, r, M, a)
        elif getattr(self, '_profile_real', None):
            f_k = _profile_real_to_fourier(self._profile_real,  # noqa
                                           cosmo, r, M, a)
        else:
            raise NotImplementedError("Profiles must have at least "
                                      " either a _profile_real or a "
                                      " _profile_fourier method.")
        norm = self.norm(cosmo, M, a)
        return norm * f_k
